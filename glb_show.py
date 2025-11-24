import numpy as np
import open3d as o3d
from pygltflib import GLTF2
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import json

def visualize_glb(glb_path):
    """
    可视化GLB文件内容（支持网格、点云、深度图）
    """
    # 加载GLB文件
    gltf = GLTF2.load(glb_path)
    print(f"GLB文件 '{os.path.basename(glb_path)}' 加载成功")
    
    # 1. 尝试提取和可视化网格
    meshes = extract_meshes(gltf)
    if meshes:
        print(f"找到 {len(meshes)} 个网格")
        visualize_meshes(meshes)
        return
    
    # 2. 尝试提取深度图并转换为点云
    depth_image, camera_params = extract_depth_and_camera(gltf)
    if depth_image is not None:
        print(f"找到深度图 (尺寸: {depth_image.shape})")
        if camera_params:
            print(f"相机参数: {camera_params}")
            # 可视化深度图
            visualize_depth_map(depth_image)
            
            # 创建点云并可视化
            pointcloud = depth_to_pointcloud(depth_image, camera_params)
            visualize_pointcloud(pointcloud)
            return
        else:
            print("警告: 找到深度图但未找到相机参数")
            visualize_depth_map(depth_image)
            return
    
    # 3. 尝试提取普通图像
    images = extract_images(gltf)
    if images:
        print(f"找到 {len(images)} 张图像")
        visualize_images(images)
        return
    
    print("警告: 未找到可识别的网格、深度图或图像数据")

def extract_meshes(gltf):
    """从GLTF中提取网格数据"""
    meshes = []
    
    for mesh_index, mesh in enumerate(gltf.meshes):
        for primitive in mesh.primitives:
            # 提取顶点位置
            position_accessor = gltf.accessors[primitive.attributes.POSITION]
            position_buffer_view = gltf.bufferViews[position_accessor.bufferView]
            position_buffer = gltf.buffers[position_buffer_view.buffer]
            position_data = position_buffer.data[
                position_buffer_view.byteOffset + position_accessor.byteOffset:
                position_buffer_view.byteOffset + position_accessor.byteOffset + position_buffer_view.byteLength
            ]
            vertices = np.frombuffer(position_data, dtype=np.float32).reshape(-1, 3)
            
            # 提取索引
            indices = None
            if primitive.indices is not None:
                indices_accessor = gltf.accessors[primitive.indices]
                indices_buffer_view = gltf.bufferViews[indices_accessor.bufferView]
                indices_buffer = gltf.buffers[indices_buffer_view.buffer]
                indices_data = indices_buffer.data[
                    indices_buffer_view.byteOffset + indices_accessor.byteOffset:
                    indices_buffer_view.byteOffset + indices_accessor.byteOffset + indices_buffer_view.byteLength
                ]
                
                if indices_accessor.componentType == 5123:  # UNSIGNED_SHORT
                    indices = np.frombuffer(indices_data, dtype=np.uint16).reshape(-1)
                elif indices_accessor.componentType == 5125:  # UNSIGNED_INT
                    indices = np.frombuffer(indices_data, dtype=np.uint32).reshape(-1)
            
            # 创建Open3D网格
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            
            if indices is not None:
                triangles = indices.reshape(-1, 3)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            
            # 计算法线（用于更好的渲染）
            o3d_mesh.compute_vertex_normals()
            
            meshes.append(o3d_mesh)
    
    return meshes

def extract_depth_and_camera(gltf):
    """提取深度图和相机参数"""
    depth_image = None
    camera_params = {}
    
    # 提取图像
    for img in gltf.images:
        if img.uri or img.bufferView is not None:
            buffer_view = gltf.bufferViews[img.bufferView]
            buffer_data = gltf.buffers[buffer_view.buffer]
            data = buffer_data.data[buffer_view.byteOffset:buffer_view.byteOffset+buffer_view.byteLength]
            
            try:
                pil_image = Image.open(io.BytesIO(data))
                image_array = np.array(pil_image)
                
                # 假设第一个找到的图像是深度图
                if depth_image is None:
                    depth_image = image_array
                    print(f"提取到图像: 尺寸={image_array.shape}, 类型={image_array.dtype}")
            except Exception as e:
                print(f"图像读取失败: {e}")
    
    # 提取相机参数
    if gltf.cameras:
        cam = gltf.cameras[0]
        if cam.perspective:
            camera_params = {
                "aspect_ratio": cam.perspective.aspectRatio or 1.0,
                "yfov": cam.perspective.yfov or (np.pi / 3),
                "znear": cam.perspective.znear or 0.1,
                "zfar": cam.perspective.zfar or 100.0
            }
    
    return depth_image, camera_params

def extract_images(gltf):
    """提取所有图像"""
    images = []
    
    for img in gltf.images:
        if img.uri or img.bufferView is not None:
            buffer_view = gltf.bufferViews[img.bufferView]
            buffer_data = gltf.buffers[buffer_view.buffer]
            data = buffer_data.data[buffer_view.byteOffset:buffer_view.byteOffset+buffer_view.byteLength]
            
            try:
                pil_image = Image.open(io.BytesIO(data))
                images.append(np.array(pil_image))
            except Exception as e:
                print(f"图像读取失败: {e}")
    
    return images

def depth_to_pointcloud(depth_image, camera_params):
    """将深度图转换为点云"""
    # 处理深度图
    if depth_image.dtype == np.uint16:
        depth_data = depth_image.astype(np.float32) / 65535.0
    elif depth_image.dtype == np.uint8:
        depth_data = depth_image.astype(np.float32) / 255.0
    else:
        depth_data = depth_image.astype(np.float32)
    
    # 如果深度图是多通道，转换为单通道
    if len(depth_data.shape) == 3:
        if depth_data.shape[2] == 3:  # RGB
            depth_data = np.mean(depth_data, axis=2)
        elif depth_data.shape[2] == 4:  # RGBA
            depth_data = depth_data[:, :, 0]  # 取第一个通道
    
    # 计算相机内参
    H, W = depth_data.shape
    fy = H / (2 * np.tan(camera_params["yfov"] / 2))
    fx = fy * camera_params["aspect_ratio"]
    cx = W / 2
    cy = H / 2
    
    # 创建Open3D深度图
    depth_o3d = o3d.geometry.Image(depth_data)
    
    # 创建内参对象
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=W,
        height=H,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )
    
    # 生成点云
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        intrinsic,
        depth_scale=1.0,
        depth_trunc=camera_params["zfar"],
        stride=2
    )
    
    return pcd

def visualize_meshes(meshes):
    """可视化网格"""
    print("可视化网格...")
    o3d.visualization.draw_geometries(meshes, window_name="3D 网格可视化")

def visualize_pointcloud(pointcloud):
    """可视化点云"""
    print("可视化点云...")
    # 添加坐标轴
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # 估计点云法线（用于更好的渲染）
    pointcloud.estimate_normals()
    
    o3d.visualization.draw_geometries([pointcloud, coordinate_frame], 
                                      window_name="点云可视化",
                                      point_show_normal=True)

def visualize_depth_map(depth_image):
    """可视化深度图"""
    print("可视化深度图...")
    plt.figure(figsize=(10, 8))
    
    # 处理单通道深度图
    if len(depth_image.shape) == 2:
        plt.imshow(depth_image, cmap='viridis')
        plt.colorbar(label='深度值')
    # 处理RGB图像
    elif len(depth_image.shape) == 3:
        plt.imshow(depth_image)
    
    plt.title("深度图")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_images(images):
    """可视化所有图像"""
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    
    if num_images == 1:
        axes = [axes]
    
    for i, img in enumerate(images):
        ax = axes[i]
        if len(img.shape) == 2:  # 灰度图
            ax.imshow(img, cmap='gray')
        else:  # 彩色图
            ax.imshow(img)
        ax.set_title(f"图像 {i+1}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 使用示例
    glb_file = "/home/wmx/Downloads/glbscene_50_All_maskbFalse_maskwFalse_camTrue_skyFalse_predDepthmap_and_Camera_Branch.glb"
    
    if not os.path.exists(glb_file):
        print(f"错误: 文件 '{glb_file}' 不存在")
    else:
        visualize_glb(glb_file)