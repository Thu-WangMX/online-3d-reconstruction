####可视化mesh文件
import open3d as o3d
import os

# 1. 定义要可视化的文件名
file_path = "mesh_with_normals.ply"

# 2. 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误：文件 '{file_path}' 在当前目录下不存在。")
    print("请确保您已经成功运行了重建脚本 (reconstruct.py)。")
else:
    # 3. 读取网格模型文件
    print(f"正在读取模型文件: {file_path}")
    mesh = o3d.io.read_triangle_mesh(file_path)

    # 4. 检查模型是否加载成功
    if not mesh.has_triangles():
        print(f"错误：文件 '{file_path}' 加载失败或文件内容为空。")
    else:
        print("加载成功！正在打开可视化窗口...")
        print("您可以在窗口中使用鼠标进行交互 (旋转/缩放/平移)。")
        print("按 'q' 键或关闭窗口即可退出。")
        
        # 5. 可视化模型
        o3d.visualization.draw_geometries([mesh])