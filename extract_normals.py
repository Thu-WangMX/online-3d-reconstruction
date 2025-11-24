import open3d as o3d
import numpy as np
import os
import json

def get_rotation_matrix_from_vectors(vec1, vec2):
    """ 计算从 vec1 旋转到 vec2 的旋转矩阵 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-6: # 如果向量方向相同或相反
        return np.identity(3) if c > 0 else -np.identity(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.identity(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def main(mesh_path="reconstruction_mesh.ply"):
    # 1. 加载模型并确保有法向量
    print(f"--- 步骤 1: 加载并准备模型 ---")
    if not os.path.exists(mesh_path):
        print(f"错误：文件 '{mesh_path}' 不存在。")
        return
        
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    print("模型加载完毕。")

    # 【新增】创建一个世界坐标系的模型
    # size 参数可以根据您的模型大小进行调整，这里设置为0.2米（20厘米）
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    # 2. 启动第一个窗口，用于交互式选择顶点
    print("\n--- 步骤 2: 请在弹出的窗口中选择任意数量的点 ---")
    print("操作指南:")
    print(" - 鼠标左键 + 拖动: 旋转视图")
    print(" - 鼠标滚轮: 缩放视图")
    print(" - 按住 [Shift] + 鼠标左键点击: 选择点 (点会变红)")
    print(" - 选择完毕后，请按键盘上的 [q] 键或关闭窗口。")
    
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window("Step 1: Select Points (SHIFT+Click), then press 'q'", 1024, 768)
    
    # 【修改】将模型和坐标系都添加到可视化窗口
    vis.add_geometry(mesh)
    vis.add_geometry(world_frame)
    
    vis.run() 
    
    # 3. 获取选择结果
    picked_points = vis.get_picked_points()
    vis.destroy_window()

    if not picked_points:
        print("\n您没有选择任何点。程序结束。")
        return

    # 4. 为每一个被选择的点创建法向量并准备保存数据
    print(f"\n--- 步骤 3: 可视化每个点的法向量 ---")
    print(f"您一共选择了 {len(picked_points)} 个顶点。")

    all_vertices = np.asarray(mesh.vertices)
    all_normals = np.asarray(mesh.vertex_normals)
    
    # 【修改】创建一个列表来存放所有要显示的几何体，预先放入主模型和世界坐标系
    geometries_to_draw = [mesh, world_frame]
    
    # 【修改】直接打开 .jsonl 文件准备写入
    output_filename = "selected_points_data.jsonl"
    with open(output_filename, 'w') as f:
        # 遍历所有被选中的点
        for point in picked_points:
            idx = point.index
            coord = all_vertices[idx]
            normal = all_normals[idx]
            
            # 【修改】创建不含 "index" 的字典
            point_data = {
                "coordinate": coord.tolist(),
                "normal": normal.tolist()
            }
            
            # 【修改】将字典转换为JSON字符串并写入文件，每条记录占一行
            f.write(json.dumps(point_data) + '\n')

            # 为可视化创建箭头 (这部分逻辑不变)
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.005, cone_radius=0.01,
                cylinder_height=0.1, cone_height=0.04
            )
            arrow.paint_uniform_color([1.0, 0.0, 0.0]) # 红色
            
            rotation = get_rotation_matrix_from_vectors([0, 0, 1], normal)
            arrow.rotate(rotation, center=[0, 0, 0])
            arrow.translate(coord)
            geometries_to_draw.append(arrow)

    print(f"已成功将 {len(picked_points)} 个点的数据逐行保存到 '{output_filename}'")

    # 遍历所有被选中的点
    for point in picked_points:
        idx = point.index
        coord = all_vertices[idx]
        normal = all_normals[idx]
        
        # 为当前点创建一个箭头
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005, cone_radius=0.01,
            cylinder_height=0.1, cone_height=0.04
        )
        arrow.paint_uniform_color([1.0, 0.0, 0.0])
        
        rotation = get_rotation_matrix_from_vectors([0, 0, 1], normal)
        arrow.rotate(rotation, center=[0, 0, 0])
        arrow.translate(coord)
        
        geometries_to_draw.append(arrow)

    # 在一个新的窗口中同时显示模型、世界坐标系和所有法向量箭头
    print("\n正在打开最终结果窗口...")
    o3d.visualization.draw_geometries(geometries_to_draw, 
                                      window_name="Step 2: Visualizing Individual Normal Vectors")

if __name__ == "__main__":
    main()