import numpy as np
import json
import os
from scipy.spatial.transform import Rotation

def get_rotation_matrix_from_vectors(vec1, vec2):
    """ 计算从 vec1 旋转到 vec2 的旋转矩阵 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-6:
        return np.identity(3) if c > 0 else -np.identity(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.identity(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def calculate_robot_pose(point_in_base, normal_in_base):
    """
    根据基座坐标系下的点坐标和法向量，计算机器人末端的6D位姿。
    返回值为一个包含6个元素的列表 [x, y, z, rx, ry, rz]。
    """
    position_xyz = point_in_base
    tool_z_axis_initial = np.array([0.0, 0.0, 1.0])
    target_direction = -normal_in_base
    rotation_matrix = get_rotation_matrix_from_vectors(tool_z_axis_initial, target_direction)
    r = Rotation.from_matrix(rotation_matrix)
    rotation_vector_xyz = r.as_rotvec()
    pose_6d = np.concatenate((position_xyz, rotation_vector_xyz)).tolist()
    return pose_6d

if __name__ == '__main__':
    input_filename = "transformed_points_data.json"
    # 【修改】定义新的输出文件名和格式
    output_filename = "robot_target_poses.jsonl"
    
    if not os.path.exists(input_filename):
        print(f"错误：输入文件 '{input_filename}' 不存在。")
        exit()

    # --- 用户需要替换的矩阵 ---
    T_B_E0 = np.array([
        [ 0.707, -0.707,  0.0,    0.1],
        [ 0.707,  0.707,  0.0,    0.2],
        [ 0.0,    0.0,    1.0,    0.5],
        [ 0.0,    0.0,    0.0,    1.0]
    ])
    T_E_C = np.array([
        [ 0.0,  0.0,  1.0,  0.1],
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0]
    ])
    
    # --- 核心计算与文件写入 ---
    with open(input_filename, 'r') as f_in, open(output_filename, 'w') as f_out:
        data_in_base_frame = json.load(f_in)
        
        print(f"--- 开始为 {len(data_in_base_frame)} 个点计算机器人目标位姿 ---")
        
        for i, data_point in enumerate(data_in_base_frame):
            p_base = np.array(data_point['coordinate_base'])
            n_base = np.array(data_point['normal_base'])

            # 计算6D位姿
            target_pose = calculate_robot_pose(p_base, n_base)
            
            # 【修改】将计算出的列表直接转换为JSON字符串并写入新行
            f_out.write(json.dumps(target_pose) + '\n')
            
            print(f"已处理 [点 {i+1}]，位姿: {target_pose}")

    print(f"\n--- 处理完成 ---")
    print(f"所有机器人的目标位姿已逐行保存到 '{output_filename}' 文件中。")