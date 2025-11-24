#通过 Open3D 将 2D 图像转换为 3D 点云，并实现交互式可视化
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
## 配环境
def main():
    # -----------------------------------------------
    # 1. 初始化 RealSense 和 Open3D
    # -----------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)# 启动相机流，返回流配置文件
    
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()#获取深度缩放因子（depth_scale），用于将深度值转换为真实世界坐标
    align = rs.align(rs.stream.color)#将深度帧 “投影” 到彩色帧的坐标系中，使两者像素严格对齐

    # 【新增】1. 创建孔洞填充滤波器
    # 选项 '1' 表示基于邻近像素填充，效果比较自然
    hole_filling = rs.hole_filling_filter(1) #用于修复深度图中因反光、低纹理区域导致的空洞

    # -----------------------------------------------
    # 2. 设置 Open3D 可视化
    # -----------------------------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window('RealSense Live Point Cloud (Depth Completed)', width=848, height=480)
    
    pcd = o3d.geometry.PointCloud()#    创建一个空的点云对象
    is_first_frame = True # 用于判断是否是第一帧，以便添加几何体到可视化窗口

    try:
        while True:
            # -------------------------------------------
            # 3. 捕获并对齐帧
            # -------------------------------------------
            frames = pipeline.wait_for_frames()#确保获取到完整的深度 + 彩色帧对
            aligned_frames = align.process(frames) # 对齐深度和彩色帧
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # 【新增】4. 应用滤波器进行深度补全
            filled_depth_frame = hole_filling.process(depth_frame)#使用孔洞填充滤波器修复深度图中的空洞

            # -------------------------------------------
            # 5. 将图像转换为 Open3D 格式
            # -------------------------------------------
            intrinsic = color_frame.profile.as_video_stream_profile().get_intrinsics()#获取彩色相机内参（焦距、光心等）
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(#转换为Open3D内参格式（用于点云坐标计算）
                intrinsic.width, intrinsic.height, intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy)
            
            # 【将深度帧和彩色帧转换为Open3D的Image对象
            depth_image = o3d.geometry.Image(np.asanyarray(filled_depth_frame.get_data()))
            color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
            
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, 
                depth_image, 
                depth_scale=1.0/depth_scale,#将 RealSense 的缩放因子转换为 Open3D 所需的单位
                depth_trunc=3.0,## 深度截断距离（忽略3米外的点
                convert_rgb_to_intensity=False)## 不将RGB转换为灰度（保留彩色）

            # -------------------------------------------
            # 6.  # 基于RGBD图像和相机内参生成点云
            # -------------------------------------------
            temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d_intrinsic)
            
            pcd.points = temp_pcd.points## 将临时点云的坐标和颜色赋给全局点云对象
            pcd.colors = temp_pcd.colors
            
            # -------------------------------------------
            # 7. 更新可视化
            # -------------------------------------------
            if is_first_frame:
                vis.add_geometry(pcd)
                is_first_frame = False
            else:
                vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()
            
    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    main()