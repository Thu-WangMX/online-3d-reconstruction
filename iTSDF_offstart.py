import open3d as o3d
import numpy as np
import os
import json
import time
import cv2
import matplotlib.pyplot as plt

try:
    from natsort import natsorted
except ImportError:
    print("请先安装 natsort 库: pip install natsort")
    exit()

def process_frame(color_file, depth_file, intrinsic, downsample_factor=2, depth_scale=1000.0, depth_trunc=1.0):
    """处理单帧RGB-D数据，支持降采样以提高速度"""
    # 读取图像
    color_cv = cv2.imread(color_file)
    depth_cv = cv2.imread(depth_file, -1)  # 以原始深度格式读取

    if depth_cv.size > 0:  # 确保深度图有数据

            # 中值滤波：去除椒盐噪声
            depth_cv = cv2.medianBlur(depth_cv, 3)  # 3x3 窗口大小
    #         depth_float = depth_cv.astype(np.float32)#OpenCV 的双边滤波函数只支持 **8 位无符号整数（8u）或32 位浮点数（32f）** 格式的图像，但你的深度图可能是 16 位无符号整数（16u）格式（这是深度相机常见的原始格式）。
    #         # 双边滤波：在保留边缘的同时平滑图像
    #         # 注意：双边滤波对深度图可能较慢，可根据性能需求调整参数
    #         depth_float = cv2.bilateralFilter(depth_float, 5, 50, 50)    
    #         depth_cv = depth_float.astype(depth_cv.dtype)

    if depth_cv.size == 0:
        print("警告: 深度图为空")
     
    
    # 降采样以提高处理速度
    if downsample_factor > 1:
        new_width = int(color_cv.shape[1] / downsample_factor)
        new_height = int(color_cv.shape[0] / downsample_factor)
        color_cv = cv2.resize(color_cv, (new_width, new_height), interpolation=cv2.INTER_AREA) #缩小图像推荐的插值方式
        depth_cv = cv2.resize(depth_cv, (new_width, new_height), interpolation=cv2.INTER_NEAREST) #用于深度图，避免插值产生非法值
        
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
    color_o3d = o3d.geometry.Image(color_cv)
    depth_o3d = o3d.geometry.Image(depth_cv)
    
    # 创建RGBD图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
    
    return rgbd_image, new_intrinsic

def fast_keyframe_selection(rgbd_current, rgbd_previous, intrinsic, transform_previous, threshold=0.15, debug=False):
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
    
    # 下采样点云以提高配准速度
    pcd_curr = pcd_curr.voxel_down_sample(voxel_size=0.02) #    0.02米的体素大小
    pcd_prev = pcd_prev.voxel_down_sample(voxel_size=0.02)
    
    # 计算点云法线
    pcd_curr.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_prev.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 使用ICP进行快速配准, ICP（Iterative Closest Point),得到当前帧与上一帧之间的变换矩阵
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_curr, pcd_prev, 0.05, transform_previous,#配准的最大距离阈值0.05
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),   #使用点到平面的误差模型进行优化（比点到点更精确）,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30,
            relative_fitness=1e-6,  # 更严格的收敛条件
            relative_rmse=1e-6))
    # reg_colored = o3d.pipelines.registration.registration_colored_icp(
    # pcd_curr, pcd_prev, 0.05, transform_previous,
    # o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    # )
    # transform_local = reg_colored.transformation  # 用颜色ICP的结果

    # 计算变换矩阵的范数作为运动度量
    motion_norm = np.linalg.norm(reg_p2p.transformation[:3, 3])#012行，第4列，即平移部分，计算欧几里得范数，得到平移距离
    is_keyframe = motion_norm > threshold
    
    if is_keyframe:
        print(f"配准结果: fitness={reg_p2p.fitness:.4f}, inlier_rmse={reg_p2p.inlier_rmse:.4f}, motion={motion_norm:.4f}, 关键帧={is_keyframe}")
    #.fitness: 匹配点占总点的比例，越大越接近；inlier_rmse：匹配点的平均误差（单位：米），越小越接近
    return is_keyframe, reg_p2p.transformation
    # motion_norm = np.linalg.norm(reg_colored.transformation[:3, 3])#012行，第4列，即平移部分，计算欧几里得范数，得到平移距离
    # is_keyframe = motion_norm > threshold
    
    # if is_keyframe:
    #     print(f"配准结果: fitness={reg_colored.fitness:.4f}, inlier_rmse={reg_colored.inlier_rmse:.4f}, motion={motion_norm:.4f}, 关键帧={is_keyframe}")
    # #.fitness: 匹配点占总点的比例，越大越接近；inlier_rmse：匹配点的平均误差（单位：米），越小越接近
    # return is_keyframe, reg_colored.transformation

def visualize_rgbd(rgbd_image):
    """可视化RGBD图像"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title('RGB image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

# 创建自定义视角设置函数
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
    #lookat = camera_pos - front * ctr.get_distance()#这行报错了
    return False,front, up, camera_pos

def main():
    # --- 1. 设置参数 ---
    data_folder = 'realsense_data_sz'
    #data_folder = 'realsense_data'
    # 图像降采样参数
    use_downsample = True
    downsample_factor = 2  # 2表示将图像尺寸减半
    
    # 深度处理参数
    depth_scale = 1000.0   # 深度缩放因子，将深度值转换为米
    depth_trunc = 1.5      # 截断超过此距离的深度值
    
    # 关键帧选择参数
    use_adaptive_keyframe = True
    frame_interval = 5     # 检查关键帧的频率
    motion_threshold = 0.05  # 降低阈值，更容易选择关键帧
    #motion_threshold = 0.03  # 降低阈值，更容易选择关键帧
    
    # TSDF参数 - 针对洗手台场景优化
    voxel_length = 1.0 / 512.0  # 恢复到原始精细度
    # sdf_trunc = 0.04           # 恢复到原始截断值
    #voxel_length = 1.0 / 768.0  # ~0.0013m（1.3mm），细节更丰富
    sdf_trunc = 0.02            # 2cm截断距离，减少过度融合，保留边缘

    
    # 只处理部分帧以提高速度
    process_every_n_frames = 1  # 处理每一帧，提高完整性
    
    # 调试选项
    debug_mode = True
    visualize_first_frame = False
 
    
    if not os.path.exists(data_folder):
        print(f"数据文件夹 '{data_folder}' 不存在。")
        return
        
    # --- 2. 加载相机内参 ---
    with open(os.path.join(data_folder, 'intrinsics.json')) as f:
        intrinsics = json.load(f)
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics['width'], intrinsics['height'], intrinsics['fx'], intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])
    
    print(f"相机内参: fx={o3d_intrinsic.intrinsic_matrix[0, 0]}, fy={o3d_intrinsic.intrinsic_matrix[1, 1]}, cx={o3d_intrinsic.intrinsic_matrix[0, 2]}, cy={o3d_intrinsic.intrinsic_matrix[1, 2]}")
    
    # 保存原始内参用于最终输出
    original_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d_intrinsic)

    # --- 3. 读取图像文件列表 ---
    color_files = natsorted([os.path.join(data_folder, 'color', f) for f in os.listdir(os.path.join(data_folder, 'color'))])
    depth_files = natsorted([os.path.join(data_folder, 'depth', f) for f in os.listdir(os.path.join(data_folder, 'depth'))])
    
    if len(color_files) != len(depth_files):
        print(f"错误: 颜色图像数量({len(color_files)})与深度图像数量({len(depth_files)})不匹配")
        return
    
    print(f"找到 {len(color_files)} 对RGB-D图像")
    
    # 只处理部分帧，这里取1实际上是处理所有帧
    color_files = color_files[::process_every_n_frames]
    depth_files = depth_files[::process_every_n_frames]
    
    # --- 4. 初始化TSDF Volume ---
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        # 新增：提高颜色融合权重（默认可能偏向深度，导致颜色细节丢失）
        #weight_function=o3d.pipelines.integration.TSDFWeightFunction.Uniform) # 或尝试其他权重函数)
        
    transform_global = np.identity(4)
    rgbd_prev_keyframe = None
    prev_transform = np.identity(4)
    keyframe_indices = []
    success_count = 0
    
    # +++ 【新增】初始化可视化窗口 +++
    vis = o3d.visualization.Visualizer() #创建可视化器对象
    vis.create_window(window_name='实时三维重建', width=1024, height=768) #创建显示窗口

    # +++ 【新增】添加初始坐标系用于参考 +++
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1) #创建坐标系网格（尺寸0.1）
    vis.add_geometry(mesh_frame) #将坐标系添加到可视化器
    geometry_display = mesh_frame  # 设置变量跟踪当前显示的几何体
    set_custom_view(vis) #设置初始视角

    # --- 5. 处理所有帧 ---
    start_time = time.time()
    
    for i in range(0, len(color_files), process_every_n_frames):
        frame_index = i  # 原始帧号

        if not use_adaptive_keyframe or (len(keyframe_indices) == 0 or (len(keyframe_indices) > 0 and (i % frame_interval == 0))):
            #关键帧选择条件：非自适应模式/首个关键帧/达到固定间隔

            # 此时才真正读取并处理图像
            color_file = color_files[i]
            depth_file = depth_files[i]
            rgbd, current_intrinsic = process_frame(
                color_file, depth_file, original_intrinsic,
                downsample_factor if use_downsample else 1,
                depth_scale, depth_trunc)

            # 关键帧判断
            is_keyframe, transform_local = fast_keyframe_selection(
                rgbd, rgbd_prev_keyframe, current_intrinsic, prev_transform, motion_threshold, debug_mode)

            if is_keyframe:
                transform_global = np.dot(transform_global, transform_local)
                keyframe_indices.append(frame_index)
                rgbd_prev_keyframe = rgbd
                prev_transform = transform_local

                # 融合进 TSDF
                volume.integrate(rgbd, current_intrinsic, np.linalg.inv(transform_global))
                success_count += 1
                print(f"已处理关键帧 {frame_index}, 位姿: {transform_global[:3, 3]}")

            #提取网格并更新可视化
                try: 
                    mesh = volume.extract_triangle_mesh()#从TSDF体积提取网格
                    mesh.compute_vertex_normals() #计算顶点法线（用于着色）
                
                    vis.remove_geometry(geometry_display)
                    vis.add_geometry(mesh)
                    geometry_display = mesh #更新当前显示的几何体引用
                    set_custom_view(vis) #设置视角
                except Exception as e:
                    print(f"[警告] 提取网格失败：{e}")

                # +++ 每次循环都刷新界面 +++
                vis.poll_events() #处理窗口事件（如关闭请求）
                vis.update_renderer() #更新渲染器显示

               
    end_time = time.time()  # 记录结束时间
    print(f"\n重建完成！总耗时: {end_time - start_time:.2f} 秒。")
    print(f"共处理 {success_count} 个关键帧，选择了 {len(keyframe_indices)} 个关键帧")
    print(f"关键帧索引: {keyframe_indices}")
    # --- 6. 后处理与可视化 ---
    print("正在提取最终的网格模型...")
    mesh = volume.extract_triangle_mesh()
    # pcd = volume.extract_point_cloud()
    # pcd.estimate_normals()
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # 检查网格是否有顶点
    if len(mesh.vertices) == 0:
        print("警告: 提取的网格没有顶点，重建可能失败")
    else:
        print(f"网格有 {len(mesh.vertices)} 个顶点和 {len(mesh.triangles)} 个三角形")
    
    # 网格清理
    mesh_clean = mesh.remove_degenerate_triangles()#    移除无效三角形
    mesh_clean = mesh_clean.remove_duplicated_vertices()#    # 移除重复顶点
    mesh_clean = mesh_clean.remove_unreferenced_vertices()#    # 移除未引用的顶点
    mesh_clean.compute_vertex_normals()#  在清理后计算法线！

    # 网格简化
    if len(mesh_clean.vertices) > 1e6:
        print("网格顶点过多，正在简化...")
        mesh_clean = mesh_clean.simplify_quadric_decimation(int(1e6))
    

    # 可视化
    if len(mesh_clean.vertices) > 0:
        output_file = "reconstruction_mesh.ply"
        o3d.io.write_triangle_mesh(output_file, mesh_clean)#    保存网格
        print(f"最终重建结果已保存到 '{output_file}'")
        print("显示最终重建的网格模型...")
        # 显示最终网格
        vis.remove_geometry(geometry_display) #移除当前显示的几何体
        vis.add_geometry(mesh_clean) #添加最终清理后的网格
        set_custom_view(vis)
        vis.run()  # 进入交互模式，可以鼠标操作视角
        
    else:
        print("由于网格为空，无法显示")

if __name__ == "__main__":
    main()