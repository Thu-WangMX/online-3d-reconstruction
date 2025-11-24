import numpy as np
import json
import os

def transform_point_and_normal(point_data, T_B_M):
    """
    将模型坐标系下的点和法向量，变换到机器人基座坐标系下。

    Args:
        point_data (dict): 包含 'coordinate' 和 'normal' 的字典。
        T_B_M (np.ndarray): 4x4的、从模型到基座的总变换矩阵。

    Returns:
        tuple: (变换后的点坐标, 变换后的法向量)
    """
    # 提取旋转部分 R_B_M，用于变换法向量
    R_B_M = T_B_M[:3, :3]

    # --- 变换点的坐标 ---
    p_M = np.array(point_data['coordinate'])
    p_M_h = np.append(p_M, 1) # 转换为齐次坐标
    p_B_h = T_B_M @ p_M_h    # 进行变换
    p_B = p_B_h[:3]          # 转回三维坐标

    # --- 变换法向量 ---
    n_M = np.array(point_data['normal'])
    n_B = R_B_M @ n_M # 只使用旋转部分
    n_B = n_B / np.linalg.norm(n_B) # 归一化

    return p_B, n_B

if __name__ == '__main__':
    # =======================================================================
    # ==                            用户需要替换的部分                          ==
    # =======================================================================
    
    input_filename = "selected_points_data.jsonl"

    # --- 1. 初始机器人位姿 T_B_E0 ---
    # 您在开始采集第一帧图像时，从机器人控制器读取的末端位姿。
    T_B_E0 = np.array([
        [ 0.707, -0.707,  0.0,    0.1],
        [ 0.707,  0.707,  0.0,    0.2],
        [ 0.0,    0.0,    1.0,    0.5],
        [ 0.0,    0.0,    0.0,    1.0]
    ])

    # --- 2. 手眼标定矩阵 T_E_C ---
    # 您通过手眼标定得到的，从机械臂末端 {E} 到相机 {C} 的固定变换。
    T_E_C = np.array([
[ 0.99950632, -0.03131929, -0.00249517, -0.03133564 ],
[ 0.03130265, 0.99948913, -0.00645112, -0.12496923 ],
[ 0.00269594, 0.00636983, 0.99997608, -0.11041862 ],
[ 0.00000000, 0.00000000, 0.00000000, 1.00000000 ]
    ])
    
    # =======================================================================
    # ==                          核心计算与结果输出                          ==
    # =======================================================================
    
    if not os.path.exists(input_filename):
        print(f"错误：输入文件 '{input_filename}' 不存在。")
        exit()

    # 先计算出总的变换矩阵
    T_B_M = T_B_E0 @ T_E_C
    print("--- 变换矩阵已准备就绪 ---")
    print("总变换矩阵 T_B_M (从模型到基座) =\n", T_B_M)
    
    transformed_data_list = []
    
    print(f"\n--- 开始处理文件 '{input_filename}' 中的每个点 ---")
    
    with open(input_filename, 'r') as f:
        for i, line in enumerate(f):
            # 将每一行的JSON字符串解析为Python字典
            point_in_model_frame = json.loads(line)
            
            # 调用函数进行变换
            point_in_base, normal_in_base = transform_point_and_normal(point_in_model_frame, T_B_M)
            
            # 打印单次变换结果
            print(f"\n[点 {i+1}]")
            print(f"  原始坐标 (模型系): {np.round(point_in_model_frame['coordinate'], 4)}")
            print(f"  变换后坐标 (基座系): {np.round(point_in_base, 4)}")
            print(f"  原始法向量 (模型系): {np.round(point_in_model_frame['normal'], 4)}")
            print(f"  变换后法向量 (基座系): {np.round(normal_in_base, 4)}")
            
            # 将变换后的结果存入列表，准备写入新文件
            transformed_data_list.append({
                "coordinate_base": point_in_base.tolist(),
                "normal_base": normal_in_base.tolist()
            })

    # 将所有变换后的结果保存到一个新的JSON文件中
    output_filename = "transformed_points_data.json"
    with open(output_filename, 'w') as f:
        json.dump(transformed_data_list, f, indent=4)

    print(f"\n--- 处理完成 ---")
    print(f"所有 {len(transformed_data_list)} 个点的变换结果已保存到 '{output_filename}' 文件中。")