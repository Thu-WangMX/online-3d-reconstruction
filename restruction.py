import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time

def main():
    # ... (前面的初始化代码保持不变) ...
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

    # ... (可视化窗口和变量初始化) ...
    vis = o3d.visualization.Visualizer()
    vis.create_window("Real-time Point Cloud Stitching", width=1280, height=720)
    
    pcd_global = o3d.geometry.PointCloud()
    rgbd_prev = None 
    transform_global = np.identity(4)
    is_first_frame = True
    frame_count = 0
    
    # 定义下采样的体素大小
    voxel_size = 0.02 # 2cm

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            rgbd_curr = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asanyarray(color_frame.get_data())),
                o3d.geometry.Image(np.asanyarray(depth_frame.get_data())),
                depth_scale=1.0/depth_scale, depth_trunc=3.0, convert_rgb_to_intensity=False)
            
            if rgbd_prev is None:
                # 处理第一帧
                pcd_curr = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_curr, o3d_intrinsic)
                pcd_global = pcd_curr.voxel_down_sample(voxel_size=voxel_size)
                rgbd_prev = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.asanyarray(color_frame.get_data())),
                    o3d.geometry.Image(np.asanyarray(depth_frame.get_data())),
                    depth_scale=1.0/depth_scale, depth_trunc=3.0, convert_rgb_to_intensity=False)
            else:
                # 处理后续帧
                option = o3d.pipelines.odometry.OdometryOption()
                # 可选：如果跟踪仍然容易失败，可以适当放宽参数
                # option.depth_diff_max = 0.07 
                success, transform_local, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd_curr, rgbd_prev, o3d_intrinsic, np.identity(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

                # 【诊断】打印每一帧的跟踪结果
                print(f"Frame {frame_count}: Odometry Success = {success}")

                if success:
                    transform_global = np.dot(transform_global, transform_local)
                    
                    # 【优化】先创建当前帧点云并降采样
                    pcd_curr = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_curr, o3d_intrinsic)
                    pcd_curr_down = pcd_curr.voxel_down_sample(voxel_size=voxel_size)
                    
                    # 变换降采样后的点云并添加到全局点云
                    pcd_curr_down.transform(transform_global)
                    pcd_global += pcd_curr_down
                    # 再次对全局点云进行降采样，以合并重叠区域
                    pcd_global = pcd_global.voxel_down_sample(voxel_size=voxel_size)

                rgbd_prev = rgbd_curr
            
            # --- 更新可视化 ---
            if is_first_frame:
                vis.add_geometry(pcd_global)
                is_first_frame = False
            else:
                vis.update_geometry(pcd_global)
            
            vis.poll_events()
            vis.update_renderer()
            frame_count += 1

    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    main()