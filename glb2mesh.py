import numpy as np
import open3d as o3d
from pygltflib import GLTF2
from PIL import Image
import io
import os
import json

def glb_to_pointcloud(glb_path, output_ply):
    """
    将包含深度图和相机参数的GLB文件转换为点云网格(PLY)
    """
    # 加载GLB文件
    gltf = GLTF2.load(glb_path)
    
    # 提取深度图
    depth_image = None
    for img in gltf.images:
        if img.uri or img.bufferView is not None:
            buffer_view = gltf.bufferViews[img.bufferView]
            buffer_data = gltf.buffers[buffer_view.buffer]
            data = buffer_data.data[buffer_view.byteOffset:buffer_view.byteOffset+buffer_view.byteLength]
            
            # 使用PIL读取图像
            try:
                pil_image = Image.open(io.BytesIO(data))
                if depth_image is None:  # 使用第一个找到的图像
                    depth_image = np.array(pil_image)
                    print(f"深度图尺寸: {depth_image.shape}")
            except Exception as e:
                print(f"图像读取失败: {e}")
    
    if depth_image is None:
        raise ValueError("未在GLB文件中找到深度图")
    
    # 处理深度图 (假设是16位单通道)
    if depth_image.dtype == np.uint16:
        depth_data = depth_image.astype(np.float32) / 65535.0
    elif depth_image.dtype == np.uint8:
        depth_data = depth_image.astype(np.float32) / 255.0
    else:
        depth_data = depth_image.astype(np.float32)
    
    # 如果深度图是RGB格式，转换为灰度
    if len(depth_data.shape) == 3 and depth_data.shape[2] == 3:
        depth_data = np.mean(depth_data, axis=2)
    elif len(depth_data.shape) == 3 and depth_data.shape[2] == 4:
        depth_data = depth_data[:, :, 0]  # 取第一个通道
    
    # 提取相机参数
    camera_params = {
        "aspect_ratio": 1.0,
        "yfov": np.pi / 3,  # 默认60度视场角
        "znear": 0.1,
        "zfar": 100.0
    }
    
    if gltf.cameras:
        cam = gltf.cameras[0]
        if cam.perspective:
            camera_params["aspect_ratio"] = cam.perspective.aspectRatio or 1.0
            camera_params["yfov"] = cam.perspective.yfov or (np.pi / 3)
            camera_params["znear"] = cam.perspective.znear or 0.1
            camera_params["zfar"] = cam.perspective.zfar or 100.0
        print(f"相机参数: {camera_params}")
    
    # 计算内参矩阵
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
        stride=2  # 跳过点以加快处理速度
    )
    
    # 保存点云
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"点云已保存至: {output_ply}")
    
    # 额外保存相机参数
    with open(os.path.splitext(output_ply)[0] + "_camera.json", "w") as f:
        json.dump(camera_params, f, indent=2)
    
    return pcd

if __name__ == "__main__":
    input_glb = "/home/wmx/Downloads/glbscene_50_All_maskbFalse_maskwFalse_camTrue_skyFalse_predDepthmap_and_Camera_Branch.glb"
    output_ply = "output_pointcloud.ply"
    
    if not os.path.exists(input_glb):
        raise FileNotFoundError(f"输入文件不存在: {input_glb}")
    
    pointcloud = glb_to_pointcloud(input_glb, output_ply)
    
    # 可选: 可视化点云
    o3d.visualization.draw_geometries([pointcloud])
    print("点云可视化完成。")