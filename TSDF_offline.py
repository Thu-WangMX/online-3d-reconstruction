import open3d as o3d
import numpy as np
import os
import json
import time

try:
    from natsort import natsorted
except ImportError:
    print("请先安装 natsort 库: pip install natsort")
    exit()#natsort用于按实际数字顺序（而非字符串顺序）排序图像文件（如00001.png在00010.png之前）

def main():
    # --- 1. 设置参数 ---
    data_folder = 'realsense_data' # 这里是你录制数据的文件夹
    #data_folder = 'realsense_data'
    # 【优化】设置关键帧间隔,你可以根据数据量调整这个值。
    keyframe_interval = 10
    
    if not os.path.exists(data_folder):
        print(f"数据文件夹 '{data_folder}' 不存在。请先运行 recorder.py 来采集数据。")
        return
        
    # --- 2. 加载相机内参，。相机内参用于将 2D 像素坐标转换为 3D 空间坐标，是 3D 重建的基础。 ---
    with open(os.path.join(data_folder, 'intrinsics.json')) as f:
        intrinsics = json.load(f)
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics['width'], intrinsics['height'], intrinsics['fx'], intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])

    # --- 3. 读取图像文件列表 ---
    color_files = natsorted([os.path.join(data_folder, 'color', f) for f in os.listdir(os.path.join(data_folder, 'color'))])
    depth_files = natsorted([os.path.join(data_folder, 'depth', f) for f in os.listdir(os.path.join(data_folder, 'depth'))])

    # --- 4. 【优化】使用 TSDF Volume 进行离线重建，将 3D 空间划分为体素，每个体素存储到最近表面的距离（SDF）和权重---
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,  # 体素大小，越小重建越精细，但内存占用越大
        sdf_trunc=0.04,#SDF 截断值，超出此距离的点不参与表面重建
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        # 启用八叉树结构（类似哈希表的动态内存管理）
        block_count=10000,  # 增加块数量以支持更大场景
        with_label=False    # 不使用语义标签以节省内存)
    )        
    transform_global = np.identity(4)## 全局变换矩阵（初始为单位矩阵），表示当前帧相对于初始帧的空间变换（位姿）
    rgbd_prev_keyframe = None # 上一关键帧的RGBD图像

    start_time = time.time()
    # 【优化】按关键帧间隔遍历文件
    keyframe_count = 0
    transform_local = np.identity(4) # 局部变换矩阵（初始为单位矩阵），表示当前帧相对于上一帧的空间变换（位姿）
    for i in range(0, len(color_files), keyframe_interval):
        print(f"正在处理第 {i+1}/{len(color_files)} 帧 (作为关键帧)...")
        keyframe_count += 1
        
        color_image = o3d.io.read_image(color_files[i])
        depth_image = o3d.io.read_image(depth_files[i])
        
        rgbd_curr_keyframe = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, depth_scale=1000.0, depth_trunc=1.0, convert_rgb_to_intensity=False)
            #depth_trunc=3.0：只保留距离小于 3 米的深度点
        if rgbd_prev_keyframe is None:
            # 处理第一个关键帧
            transform_global = np.identity(4)
            rgbd_prev_keyframe = rgbd_curr_keyframe
        else:
            # 使用 RGB-D Odometry 方法计算两帧之间的相对变换（transform_local计算两个关键帧之间的里程计
            option = o3d.pipelines.odometry.OdometryOption()#一个配置对象，用于指定 RGB-D Odometry 算法的一些参数
            success, transform_local, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbd_curr_keyframe, rgbd_prev_keyframe, o3d_intrinsic, transform_local,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
            #这是 Open3D 中用于计算 RGB-D Odometry 的雅可比矩阵的一个方法。
            # RGB-D Odometry 是一种通过比较连续帧之间的RGB和深度信息来估计相机运动的技术。提供了一种特定的方法来计算这个过程中的雅可比行列式，它结合了颜色和深度信息来更准确地估计两帧之间的变换
            if success:
                transform_global = np.dot(transform_global, transform_local)#更新全局变换矩阵
            #当成功计算出当前帧相对于上一帧的局部变换（transform_local）后，
            # 使用矩阵乘法将这个局部变换应用到全局变换矩阵（transform_global）
            rgbd_prev_keyframe = rgbd_curr_keyframe

        # 将当前关键帧融合到TSDF体中
        volume.integrate(rgbd_curr_keyframe, o3d_intrinsic, np.linalg.inv(transform_global))

    end_time = time.time()
    print(f"\n重建完成！总耗时: {end_time - start_time:.2f} 秒。")
    print(f"共处理 {keyframe_count} 个关键帧。")
    # --- 5. 后处理与可视化 ---
    print("正在提取最终的网格模型...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals() # 计算法线以获得更好的渲染效果
    
    # 可选：进行一些网格清理
    mesh_clean = mesh.remove_degenerate_triangles()
    mesh_clean = mesh_clean.remove_duplicated_vertices()
    mesh_clean = mesh_clean.remove_unreferenced_vertices()

    output_file = "reconstruction_mesh.ply"
    o3d.io.write_triangle_mesh(output_file, mesh_clean)
    print(f"最终重建结果已保存到 '{output_file}'")

    print("显示最终重建的网格模型...")
    o3d.visualization.draw_geometries([mesh_clean])

if __name__ == "__main__":
    main()