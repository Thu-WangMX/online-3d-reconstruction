import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def visualize_single_frame(frame_index=0):
    """
    可视化指定帧的彩色图、深度图、以及生成的单帧点云。
    :param frame_index: 要观察的图像帧的索引，默认为第一帧 (0)。
    """
    print(f"--- 开始观察第 {frame_index + 1} 帧的数据 ---")
    
    # --- 准备数据 ---
    dataset = o3d.data.SampleRedwoodRGBDImages()
    trajectory = o3d.io.read_pinhole_camera_trajectory(dataset.trajectory_log_path)
    intrinsic = trajectory.parameters[0].intrinsic

    # 读取指定帧的图像
    color_image = o3d.io.read_image(dataset.color_paths[frame_index])
    depth_image = o3d.io.read_image(dataset.depth_paths[frame_index])

    # --- 1. 使用 matplotlib 可视化2D图像 ---
    print("正在显示彩色图和深度图... 关闭图像窗口后将继续。")
    
    # 将Open3D图像转为Numpy数组用于显示
    color_numpy = np.asarray(color_image)
    depth_numpy = np.asarray(depth_image)

    # 创建一个图床，并排显示两个图像
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title(f'Frame #{frame_index} - Color Image')
    axes[0].imshow(color_numpy)
    axes[0].axis('off') # 关闭坐标轴

    axes[1].set_title(f'Frame #{frame_index} - Depth Image')
    # 使用伪彩图 (cmap='viridis') 显示深度，更直观
    im = axes[1].imshow(depth_numpy, cmap='viridis')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], shrink=0.8, label='Depth (raw value)')
    
    plt.show()

    # --- 2. 可视化单帧点云 ---
    print("正在生成并显示单帧点云... 关闭点云窗口后程序将结束。")
    
    # 创建RGBD图像对象
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_trunc=4.0, convert_rgb_to_intensity=False)
    
    # 从RGBD图像创建点云
    # 这一步需要相机内参，但不需要外参（位姿），因为我们是在相机的坐标系下观察
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic)

    # 翻转点云，让它看起来是正的（这是一个常见的可视化技巧）
    pcd.transform([[1, 0, 0, 0], 
                   [0, -1, 0, 0], 
                   [0, 0, -1, 0], 
                   [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd])
    
    print(f"--- 第 {frame_index + 1} 帧数据观察完毕 ---")

def main():
    """
    使用内置的 SampleRedwoodRGBDImages 数据集，从零开始进行三维重建。
    【终极版】修正了可视化函数的调用。
    """
    # 1. 加载内置数据集
    print("正在加载内置的 Redwood RGB-D 示例数据集...")
    dataset = o3d.data.SampleRedwoodRGBDImages()

    # 2. 从轨迹日志文件加载相机参数（包含内参和外参）
    print("正在加载相机轨迹（位姿及内参）...")
    trajectory = o3d.io.read_pinhole_camera_trajectory(dataset.trajectory_log_path)
    
    # 3. 从 trajectory 对象中获取相机内参
    intrinsic = trajectory.parameters[0].intrinsic

    # 4. 初始化用于融合的 TSDF Volume
    voxel_length = 4.0 / 512.0
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # 5. 遍历数据集中的每一张图，并将其融合到 TSDF Volume 中
    print("开始从RGB-D图像序列进行三维重建...")
    for i in range(len(dataset.depth_paths)):
        print(f"正在处理第 {i + 1}/{len(dataset.depth_paths)} 张图像...")

        color = o3d.io.read_image(dataset.color_paths[i])
        depth = o3d.io.read_image(dataset.depth_paths[i])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

        extrinsic = trajectory.parameters[i].extrinsic
        
        tsdf_volume.integrate(rgbd, intrinsic, extrinsic)

    # 6. 从融合后的 TSDF Volume 中提取出三角网格(Mesh)
    print("重建完成，正在提取网格模型...")
    mesh = tsdf_volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # 7. 可视化我们亲手创建的模型
    print("✅ 成功！显示我们从零开始重建的模型。")
    # 【修正】使用最简单、兼容性最强的 draw_geometries 函数
    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    visualize_single_frame(frame_index=3)
    main()