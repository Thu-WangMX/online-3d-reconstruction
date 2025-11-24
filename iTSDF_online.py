import open3d as o3d
import numpy as np
import os
import json
import time
import cv2
import pyrealsense2 as rs
import threading
import queue  # # 多线程处理
import matplotlib.pyplot as plt
import shutil # 用于复制文件

try:
    from natsort import natsorted
except ImportError:
    print("请先安装 natsort 库: pip install natsort")
    exit()

# 全局变量用于控制采集线程
capture_running = True #用于线程间通信，控制采集线程的结束
frame_queue = queue.Queue(maxsize=30)  # 设置帧数据队列大小防止内存溢出，是线程安全的队列，用于在采集线程和重建线程间传递数据

def capture_frames(output_folder, duration=None, max_frames=None):
    """实时采集RGB-D帧并保存到队列中"""
    global capture_running
    
    # 确保输出文件夹存在
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        # 创建 color 和 depth 子文件夹路径
        color_folder = os.path.join(output_folder, 'color')
        depth_folder = os.path.join(output_folder, 'depth')
        os.makedirs(os.path.join(output_folder, 'color'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'depth'), exist_ok=True)
        
    # 清空 color 文件夹
    for filename in os.listdir(color_folder):
        file_path = os.path.join(color_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    # 清空 depth 文件夹
    for filename in os.listdir(depth_folder):
        file_path = os.path.join(depth_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
    # 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # 启动流
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)  # 对齐深度和彩色帧
        
        # 保存相机内参
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsic_dict = {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'ppx': intrinsics.ppx,
            'ppy': intrinsics.ppy
        }
        
        # 将内参放入队列
        frame_queue.put(('intrinsics', intrinsic_dict))
        
        if output_folder:
            with open(os.path.join(output_folder, 'intrinsics.json'), 'w') as f:
                json.dump(intrinsic_dict, f, indent=4)
            print("相机内参已保存。")
        
        # 录制循环
        frame_count = 0
        start_time = time.time()
        
        print("开始实时采集和重建... 按 'q' 键退出并保存。")
        while capture_running and (duration is None or time.time() - start_time < duration) and (max_frames is None or frame_count < max_frames):
            # 获取帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)  # 对齐
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 保存图像（可选）
            if output_folder:
                cv2.imwrite(f'{output_folder}/color/{frame_count:05d}.png', color_image)
                cv2.imwrite(f'{output_folder}/depth/{frame_count:05d}.png', depth_image)
            
            # 将帧放入队列
            frame_queue.put(('frame', frame_count, color_image, depth_image))
            
            # 显示实时画面以供参考
            cv2.imshow('RealSense Live View - Press Q to Stop', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                capture_running = False
                break
            
            frame_count += 1
        
        print(f"\n采集结束，共采集 {frame_count} 帧图像")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        # 放入结束标志
        frame_queue.put(('end', None, None, None))

def process_frame(color_image, depth_image, intrinsic, use_image_downsample=1, downsample_factor=2, use_depth_filter=True, depth_scale=1000.0, depth_trunc=1.0):
    """处理单帧RGB-D数据，支持降采样以提高速度"""

    if depth_image.size > 0:  # 确保深度图有数据
        
        if use_depth_filter:
            # 中值滤波：去除椒盐噪声
            depth_image = cv2.medianBlur(depth_image, 3)  # 3x3 窗口大小
            depth_float = depth_image.astype(np.float32)#OpenCV 的双边滤波函数只支持 **8 位无符号整数（8u）或32 位浮点数（32f）** 格式的图像，但你的深度图可能是 16 位无符号整数（16u）格式（这是深度相机常见的原始格式）。
            # 双边滤波：在保留边缘的同时平滑图像，注意：双边滤波对深度图可能较慢，可根据性能需求调整参数
            depth_float = cv2.bilateralFilter(depth_float, 5, 50, 50)    
            depth_image = depth_float.astype(depth_image.dtype)
  

    # 降采样以提高处理速度
    if use_image_downsample and downsample_factor > 1:
        new_width = int(color_image.shape[1] / downsample_factor)
        new_height = int(color_image.shape[0] / downsample_factor)
        color_image = cv2.resize(color_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        depth_image = cv2.resize(depth_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # 创建新的相机内参对象
        new_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=new_width,
            height=new_height,
            fx=intrinsic.intrinsic_matrix[0, 0] / downsample_factor,
            fy=intrinsic.intrinsic_matrix[1, 1] / downsample_factor,
            cx=intrinsic.intrinsic_matrix[0, 2] / downsample_factor,
            cy=intrinsic.intrinsic_matrix[1, 2] / downsample_factor
        )
    else:
        new_intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsic)
    
    # 将 OpenCV 的 ndarray 转换为 Open3D 的 Image 格式
    color_o3d = o3d.geometry.Image(color_image)
    depth_o3d = o3d.geometry.Image(depth_image)
    
    # 创建RGBD图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
    
    return rgbd_image, new_intrinsic

def fast_keyframe_selection(rgbd_current, rgbd_previous, intrinsic, transform_previous,  use_pointcloud_downsample=True, voxel_size1=0.02, threshold=0.05, debug=False):
    """更快的关键帧选择方法，使用点云配准而不是RGBD里程计"""
    
    if rgbd_previous is None:
        if debug:
            print("第一帧，选为关键帧")
        return True, np.identity(4)
    
    # 从RGBD创建点云
    pcd_curr = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_current, intrinsic)
    pcd_prev = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_previous, intrinsic)
    
    # 检查点云是否有点
    if len(pcd_curr.points) < 100 or len(pcd_prev.points) < 100:
        if debug:
            print(f"警告: 点云点数过少，当前: {len(pcd_curr.points)}, 前一帧: {len(pcd_prev.points)}")
        return False, transform_previous
    
    if use_pointcloud_downsample:
    # 下采样点云以提高配准速度
        pcd_curr = pcd_curr.voxel_down_sample(voxel_size=voxel_size1)
        pcd_prev = pcd_prev.voxel_down_sample(voxel_size=voxel_size1)
    
    # 计算点云法线
    pcd_curr.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_prev.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 使用ICP进行快速配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_curr, pcd_prev, 0.05, transform_previous,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))

    # 计算变换矩阵的范数作为运动度量
    motion_norm = np.linalg.norm(reg_p2p.transformation[:3, 3])
    is_keyframe = motion_norm > threshold
    
    if is_keyframe:
        print(f"配准结果: fitness={reg_p2p.fitness:.4f}, inlier_rmse={reg_p2p.inlier_rmse:.4f}, motion={motion_norm:.4f}, 关键帧={is_keyframe}")
    
    return is_keyframe, reg_p2p.transformation

def set_custom_view(vis):
    ctr = vis.get_view_control()

    ctr.set_front([-1, 0.3, -2])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat([-0.2, 0.2, 0.2])
    ctr.set_zoom(0.3)#缩放

    # 获取相机参数
    params = ctr.convert_to_pinhole_camera_parameters()

    # 提取参数
    front = params.extrinsic[:3, 2]  # 相机Z轴方向（观察方向）
    up = -params.extrinsic[:3, 1]    # 相机Y轴负方向（上方向）

    # 计算相机位置
    camera_pos = -np.dot(params.extrinsic[:3, :3].T, params.extrinsic[:3, 3])

    # 计算注视点
    #lookat = camera_pos - front * ctr.get_distance()  #这行报错了
    lookat = camera_pos - front * 0.5  # 使用固定距离值
    return False, front, up, camera_pos, lookat

def realtime_reconstruction(output_folder=None, duration=None, max_frames=None):
    """实时三维重建主函数"""
    global capture_running

     # --- 新增：用户选择处理模式 --- 
    # 第一次输入：是否对depth做滤波
    while True:
        try:
            use_depth_filter = int(input("是否对depth做滤波？(0=不做, 1=做): "))
            if use_depth_filter in [0, 1]:
                break
            else:
                print("输入无效，请输入0或1")
        except ValueError:
            print("输入无效，请输入整数")
    
    # 第二次输入：是否对color/depth降采样
    while True:
        try:
            use_image_downsample = int(input("是否对color与depth降采样？(0=不做, 1=做): "))
            if use_image_downsample in [0, 1]:
                break
            else:
                print("输入无效，请输入0或1")
        except ValueError:
            print("输入无效，请输入整数")
    
    # 第三次输入：设置降采样因子（仅在use_downsample=1时需要）
    
    if use_image_downsample == 1:
        while True:
            try:
                downsample_factor = int(input("请输入需要的降采样因子（整数＞1）: "))
                if downsample_factor > 1:
                    break
                else:
                    print("降采样因子必须≥1")
            except ValueError:
                print("输入无效，请输入整数")
    else:
        downsample_factor = 1  # 不进行降采样
    
    # 第四次输入：是否对点云做降采样
    while True:
        try:
            use_pointcloud_downsample = int(input("是否对点云做降采样？(0=不做, 1=做): "))
            if use_pointcloud_downsample in [0, 1]:
                break
            else:
                print("输入无效，请输入0或1")
        except ValueError:
            print("输入无效，请输入整数")
    
    # 第五次输入：设置体素大小（仅在use_pointcloud_downsample=1时需要）
    if use_pointcloud_downsample == 1:
        while True:
            try:
                voxel_size = float(input("请输入需要的体素大小（默认0.02，建议0.01-0.04，越大，降采样程度越大）: "))
                if 0.005 <= voxel_size <= 0.1:
                    break
                else:
                    print("体素大小应在0.005-0.1之间")
            except ValueError:
                print("输入无效，请输入数字")
    else:
        voxel_size = 0.02  # 默认值（实际不会用到）

     # 第六次输入：设置运动阈值
    while True:
        try:
            motion_threshold = float(input("请输入运动阈值大小（默认0.05，越大，选择的关键帧越少，丢失的细节可能越多）: "))
            if 0.01 <= motion_threshold <= 0.5:
                break
            else:
                print("运动阈值大小应在0.01-0.5之间")
        except ValueError:
            print("输入无效，请输入数字")

    
    # --- 1. 设置参数 ---

    
    # 深度处理参数
    depth_scale = 1000.0   # 深度缩放因子，将深度值转换为米
    depth_trunc = 1.5      # 截断超过此距离的深度值
    
    # 关键帧选择参数
    use_adaptive_keyframe = True
    frame_interval = 3     # 检查关键帧的频率
    motion_threshold = motion_threshold  # 降低阈值，更容易选择关键帧
    
    # TSDF参数 - 针对洗手台场景优化
    voxel_length = 1.0 / 512.0  # 恢复到原始精细度
    sdf_trunc = 0.04           # 恢复到原始截断值
    
    # 只处理部分帧以提高速度
    process_every_n_frames = 1  # 处理每一帧，提高完整性
    
    # 调试选项
    debug_mode = True

    # 打印处理模式
    print(f"\n处理模式:")
    print(f"  深度图滤波: {'启用' if use_depth_filter else '禁用'}")
    print(f"  图像降采样: {'启用' if use_image_downsample else '禁用'} (因子={downsample_factor})")
    print(f"  点云降采样: {'启用' if use_pointcloud_downsample else '禁用'} (体素大小={voxel_size})")
    
    # --- 2. 初始化相机内参 ---
    # 等待从采集线程获取内参
    print("等待相机内参...")
    original_intrinsic = None
    
    # 启动采集线程
    capture_thread = threading.Thread(
        target=capture_frames, 
        args=(output_folder, duration, max_frames),
        daemon=True
    )
    capture_thread.start()
    
    # 等待获取内参
    while original_intrinsic is None:
        try:
            data = frame_queue.get(timeout=2.0)
            if data[0] == 'intrinsics':
                intrinsics = data[1]
                original_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    intrinsics['width'], intrinsics['height'], 
                    intrinsics['fx'], intrinsics['fy'], 
                    intrinsics['ppx'], intrinsics['ppy'])
                print(f"获取到相机内参: fx={original_intrinsic.intrinsic_matrix[0, 0]}, fy={original_intrinsic.intrinsic_matrix[1, 1]}, cx={original_intrinsic.intrinsic_matrix[0, 2]}, cy={original_intrinsic.intrinsic_matrix[1, 2]}")
            elif data[0] == 'end':
                print("未获取到相机内参，重建无法继续")
                return
        except queue.Empty:
            print("等待相机内参超时...")
            continue
    
    # --- 3. 初始化TSDF Volume ---
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
    transform_global = np.identity(4)
    rgbd_prev_keyframe = None
    prev_transform = np.identity(4)
    keyframe_indices = []
    success_count = 0
    
    # +++ 初始化可视化窗口 +++
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='实时三维重建', width=1024, height=768)

    # +++ 添加固定的初始坐标系 +++
    initial_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(initial_frame)
    geometry_display = initial_frame

    # +++ 添加动态的相机坐标系 +++
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    vis.add_geometry(camera_frame)

    set_custom_view(vis) #设置初始视角
    

    # # +++ 设置初始视角 +++
    # view_control = vis.get_view_control()
    # view_control.set_front([-0.3, -0.3, -1.0])
    # view_control.set_up([0, -1, 0])
    # view_control.set_lookat([0, 0, 0])
    # view_control.set_zoom(0.5)
    
    # --- 4. 实时重建循环 ---
    start_time = time.time()
    frame_count = 0
    last_update_time = time.time()
    frames_processed = 0
    
    print("开始实时重建...")
    while capture_running or not frame_queue.empty():
        try:
            # 从队列中获取帧（带超时）
            data = frame_queue.get(timeout=1.0)
            data_type = data[0]
            
            # 检查结束标志
            if data_type == 'end':
                print("接收到结束信号")
                break
            elif data_type != 'frame':
                continue  # 跳过非帧数据
                
            # 解包帧数据
            frame_index, color_image, depth_image = data[1], data[2], data[3]
            
            # 处理当前帧
            rgbd, current_intrinsic = process_frame(
                color_image, depth_image, original_intrinsic, use_image_downsample = use_image_downsample,
                downsample_factor = downsample_factor ,use_depth_filter = use_depth_filter,
                depth_scale =  depth_scale, depth_trunc =  depth_trunc)
            
            # 关键帧判断
            if not use_adaptive_keyframe or len(keyframe_indices) == 0 or (len(keyframe_indices) > 0 and frame_index % frame_interval == 0):
                is_keyframe, transform_local = fast_keyframe_selection(
                    rgbd, rgbd_prev_keyframe, current_intrinsic, prev_transform, use_pointcloud_downsample, voxel_size, motion_threshold, debug_mode)
                
                if is_keyframe:
                    transform_global = np.dot(transform_global, transform_local)
                    keyframe_indices.append(frame_index)
                    rgbd_prev_keyframe = rgbd
                    prev_transform = transform_local
                    
                    # 融合进 TSDF
                    volume.integrate(rgbd, current_intrinsic, np.linalg.inv(transform_global))
                    success_count += 1
                    print(f"已处理关键帧 {frame_index}, 位姿: {transform_global[:3, 3]}")
            
            frames_processed += 1
            
            # 定期更新可视化
            current_time = time.time()
            if current_time - last_update_time > 0.5:  # 每0.5秒更新一次
                try:
                    mesh = volume.extract_triangle_mesh()
                    mesh.compute_vertex_normals()
                    
                    vis.remove_geometry(geometry_display)
                    vis.add_geometry(mesh)
                    geometry_display = mesh
                    set_custom_view(vis) #设置视角
                    print(f"更新可视化: {frames_processed}帧已处理")

                    # +++ 关键修改：更新相机坐标系可视化 +++
                    # 从全局变换矩阵中提取相机位置和朝向
                    camera_pos = transform_global[:3, 3]  # 相机位置
                    camera_rot = transform_global[:3, :3]  # 相机旋转矩阵
                    
                    # 重置相机坐标系并应用位姿
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=camera_pos)
                    # 应用旋转（注意：Open3D变换矩阵为列主序，需确保矩阵格式正确）
                    camera_frame.rotate(camera_rot, center=camera_pos)
                    
                    # 先移除旧的相机坐标系，再添加新的
                    vis.remove_geometry(camera_frame_old if 'camera_frame_old' in locals() else None)
                    vis.add_geometry(camera_frame)
                    camera_frame_old = camera_frame  # 保存当前坐标系用于下次更新
                    set_custom_view(vis) #设置视角
 
                except Exception as e:
                    print(f"[警告] 提取网格失败：{e}")

                # +++ 确保两个坐标系都被更新 +++
                vis.update_geometry(initial_frame)
                vis.update_geometry(camera_frame)
                
                last_update_time = current_time
            
            # 刷新界面
            vis.poll_events()
            vis.update_renderer()
            
            frame_queue.task_done()
            
        except queue.Empty:
            # 队列为空但采集仍在进行，继续等待
            continue
            
    end_time = time.time()
    print(f"\n重建完成！总耗时: {end_time - start_time:.2f} 秒。")
    print(f"共处理 {success_count} 个关键帧，选择了 {len(keyframe_indices)} 个关键帧")
    print(f"关键帧索引: {keyframe_indices}")
    
    # --- 5. 后处理与可视化 ---
    print("正在提取最终的网格模型...")
    mesh = volume.extract_triangle_mesh()
    
    # 检查网格是否有顶点
    if len(mesh.vertices) == 0:
        print("警告: 提取的网格没有顶点，重建可能失败")
        # 尝试创建空网格的替代品
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    else:
        print(f"网格有 {len(mesh.vertices)} 个顶点和 {len(mesh.triangles)} 个三角形")
    
    # 网格清理
    mesh_clean = mesh.remove_degenerate_triangles()
    mesh_clean = mesh_clean.remove_duplicated_vertices()
    mesh_clean = mesh_clean.remove_unreferenced_vertices()
    mesh_clean.compute_vertex_normals()
    
    # 网格简化
    if len(mesh_clean.vertices) > 1e6:
        print("网格顶点过多，正在简化...")
        mesh_clean = mesh_clean.simplify_quadric_decimation(int(1e6))
    
    # 可视化
    if len(mesh_clean.vertices) > 0:
        output_file = "realtime_reconstruction_mesh.ply"
        o3d.io.write_triangle_mesh(output_file, mesh_clean)
        print(f"最终重建结果已保存到 '{output_file}'")
        
        # 显示最终网格
        vis.remove_geometry(geometry_display)
        vis.add_geometry(mesh_clean)
        set_custom_view(vis) #设置视角
        # 进入交互模式
        vis.run()
    else:
        print("由于网格为空，无法显示")
    
    # 关闭窗口
    vis.destroy_window()
    
    # 确保采集线程结束
    capture_running = False
    capture_thread.join(timeout=1.0)

if __name__ == "__main__":
    # 设置参数
    output_folder = 'realsense_data_realtime'  # 设置为None则不保存图像
    duration = None  # 采集持续时间（秒），设置为None则持续直到按q键
    max_frames = None  # 最大采集帧数，设置为None则不限制
    
    # 确保输出文件夹存在
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    realtime_reconstruction(output_folder, duration, max_frames)