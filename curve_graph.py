# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # 导入 3D 绘图模块
# import os # 用于处理文件路径

# # --- 恢复的读取和提取函数 (无缩放功能) ---
# def read_and_extract_xyz_concise(file_path):
#     """简洁地读取 JSON 并提取 XYZ，假设格式正确，不进行缩放"""
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     xyz_coords_list = [ [float(item[0]), float(item[1]), float(item[2])] for item in data ]
#     xyz_coords_np = np.array(xyz_coords_list)
#     return xyz_coords_np

# # --- 可视化函数 (更新了主轨迹标签) ---
# def visualize_curve_and_points_concise(main_xyz, specific_xyz=None, control_points_xyz=None):
#     """简洁地可视化主曲线、特定点和控制点"""
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # 绘制主轨迹曲线 (来自 waypoints.json, 未缩放)
#     if main_xyz is not None and main_xyz.shape[0] > 0:
#         ax.plot(main_xyz[:, 0], main_xyz[:, 1], main_xyz[:, 2], label='curve') # 标签更新

#     # 绘制特定点 (来自代码中定义的六个点, 未缩放)
#     if specific_xyz is not None and specific_xyz.shape[0] > 0:
#         ax.scatter(specific_xyz[:, 0],
#                    specific_xyz[:, 1],
#                    specific_xyz[:, 2],
#                    c='r',          # 颜色设为红色
#                    marker='X',      # 标记风格设为 'X'
#                    s=100,          # 点的大小设为 100
#                    label='sample')   # 添加图例标签

#     # 绘制控制点 (来自 control_points.json, 已缩放)
#     if control_points_xyz is not None and control_points_xyz.shape[0] > 0:
#         ax.scatter(control_points_xyz[:, 0],
#                    control_points_xyz[:, 1],
#                    control_points_xyz[:, 2],
#                    c='g',          # 颜色设为绿色
#                    marker='o',      # 标记风格设为 'o' (圆点)
#                    s=50,           # 点的大小设为 50
#                    label='control_points') # 添加图例标签

#     ax.legend() # 显示图例
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Trajectory Visualization') # 标题更新
#     plt.show()

# # --- 主程序入口 ---
# if __name__ == "__main__":
#     # --- 定义文件路径 ---
#     waypoints_file_path = 'waypoints.json' # 您的主轨迹 JSON 文件名
#     control_points_file_path = 'control_points.json' # 您的控制点 JSON 文件名

#     # --- 直接定义您提供的六个特定位姿 (6 维列表) ---
#     # 这些点的XYZ坐标将按原样使用，不进行缩放
#     specific_poses_6d = [
#         [-0.5889062871727464, 0.14207100001274142, 0.10909034872102628, -1.0638352686876693, -2.4384189463589703, 1.058334478952947],
#         [-0.5291665789054791, 0.07473299091874429, 0.09846134885204899, -1.5094866649007754, -2.1202624269741235, 0.7542283048423432],
#         [-0.521117506082712, 0.038036479623482077, 0.09310892355592701, -1.3526321861180215, -1.9749666035862192, 0.726031360404872],
#         [-0.5316238426446698, -0.03963773513948222, 0.09959057051572551, -2.015914909180084, -1.2496676523730124, 0.6237208761766714],
#         [-0.5874866142637598, -0.12744952146270894, 0.1042381133003337, -2.128909473583844, -0.537201359130647, 0.4224845030575986],
#         [-0.6622061099267336, -0.13101117947745206, 0.12282400250617741, -2.3927513655715917, -0.47977820649984443, -0.22550492498460406]
#     ]

#     # --- 直接从这六个特定位姿中提取它们的 XYZ 坐标 (不缩放) ---
#     specific_xyz_coords = np.array([ [float(p[0]), float(p[1]), float(p[2])] for p in specific_poses_6d ])

#     main_xyz = np.empty((0, 3)) # 初始化为空数组
#     control_points_xyz = np.empty((0,3)) # 初始化为空数组

#     try:
#         # 1. 读取并提取主轨迹的 XYZ (来自 waypoints.json) - 不缩放
#         print(f"尝试读取并提取主轨迹数据从: {waypoints_file_path} ...")
#         if os.path.exists(waypoints_file_path):
#             main_xyz = read_and_extract_xyz_concise(waypoints_file_path)
#             print(f"成功提取 {main_xyz.shape[0]} 个主轨迹点 (未使用缩放)。")
#         else:
#             print(f"警告: 主轨迹文件 {waypoints_file_path} 未找到。将不绘制主轨迹。")

#         # 2. 读取并提取控制点的 XYZ (来自 control_points.json) - 然后进行缩放
#         print(f"\n尝试读取并提取控制点数据从: {control_points_file_path} ...")
#         if os.path.exists(control_points_file_path):
#             control_points_xyz_raw = read_and_extract_xyz_concise(control_points_file_path)
#             control_points_xyz = control_points_xyz_raw / 1000.0 # 在此处应用缩放
#             print(f"成功提取 {control_points_xyz_raw.shape[0]} 个控制点，并已缩放 (除以1000)。")
#         else:
#             print(f"警告: 控制点文件 {control_points_file_path} 未找到。将不绘制控制点。")

#         # 3. 调用可视化函数
#         print("\n开始可视化...")
#         visualize_curve_and_points_concise(
#             main_xyz,
#             specific_xyz=specific_xyz_coords,
#             control_points_xyz=control_points_xyz
#         )
#         print("\n可视化完成 (如果窗口弹出)。")

#     except json.JSONDecodeError as e:
#         print(f"错误: JSON 解码失败。请检查文件格式是否正确。详细: {e}")
#     except (IndexError, TypeError, ValueError) as e:
#         print(f"错误: 文件中的数据格式与预期不符。请检查文件内容是否为列表，且内部为列表/元组(长度>=3，可转换为数字)。详细: {e}")
#     except Exception as e:
#         print(f"发生未知错误: {e}")

#     print("\n程序运行结束.")
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 导入 3D 绘图模块
import os # 用于处理文件路径
from scipy.spatial.transform import Rotation # <<< 新增导入：用于旋转向量转换

# --- 已有的读取XYZ函数 (无缩放功能, 用于 control_points.json) ---
def read_and_extract_xyz_concise(file_path):
    """简洁地读取 JSON 并提取 XYZ，假设格式正确，不进行缩放"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    xyz_coords_list = [ [float(item[0]), float(item[1]), float(item[2])] for item in data if len(item) >=3 ]
    xyz_coords_np = np.array(xyz_coords_list)
    return xyz_coords_np

# --- 新增：读取6D位姿函数 (用于 waypoints.json) ---
def read_and_extract_6d_poses(file_path):
    """读取 JSON 并提取6D位姿 (x, y, z, rx, ry, rz)，假设格式正确。"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    # 假设每个 item 是一个包含至少6个数字元素的列表/元组
    pose_list = []
    for item_idx, item in enumerate(data):
        if len(item) >= 6:
            try:
                pose_list.append([
                    float(item[0]), float(item[1]), float(item[2]), # 位置
                    float(item[3]), float(item[4]), float(item[5])  # 旋转向量
                ])
            except (ValueError, TypeError) as e:
                print(f"警告: 在文件 {file_path} 的第 {item_idx+1} 项转换6D位姿数据时出错: {item}. 错误: {e}. 跳过此项。")
        else:
            print(f"警告: 在文件 {file_path} 的第 {item_idx+1} 项数据不足6个元素: {item}. 跳过此项。")
    pose_np = np.array(pose_list)
    return pose_np

# --- 旧的可视化函数 (可选保留，如果仍需单独绘制无姿态的曲线) ---
# def visualize_curve_and_points_concise(main_xyz, specific_xyz=None, control_points_xyz=None):
#     """简洁地可视化主曲线、特定点和控制点"""
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     # ... (旧代码内容) ...
#     plt.show()


# --- 新增：可视化轨迹及姿态的函数 ---
def visualize_trajectory_with_orientations(
    main_poses_6d=None,
    specific_xyz_for_marker=None, # 用于标记 specific_poses_6d 中的位置点
    control_points_xyz=None,
    skip_frames_main=10,
    axis_length_main=0.05
):
    """
    可视化主轨迹（带姿态）、特定标记点和控制点。
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    all_points_for_scaling = [] # 用于后续的坐标轴范围设定

    # 1. 绘制主轨迹曲线 (来自 waypoints.json 的位置) 及其姿态
    if main_poses_6d is not None and main_poses_6d.shape[0] > 0:
        main_positions = main_poses_6d[:, :3]
        main_rotation_vectors = main_poses_6d[:, 3:6]
        all_points_for_scaling.append(main_positions)

        ax.plot(main_positions[:, 0], main_positions[:, 1], main_positions[:, 2], label='Main Trajectory (Waypoints)', color='blue', zorder=1)

        # 绘制姿态坐标系
        for i in range(0, main_positions.shape[0], skip_frames_main):
            pos = main_positions[i]
            rot_vec = main_rotation_vectors[i]
            try:
                r_matrix = Rotation.from_rotvec(rot_vec).as_matrix()
            except ValueError as e: # 更具体的异常捕获
                print(f"警告: 无法转换旋转向量 {rot_vec} (在主轨迹索引 {i} 处)。可能为零向量或无效。错误: {e}. 跳过此姿态绘制。")
                r_matrix = np.eye(3) # 使用单位矩阵作为默认值
            except Exception as e:
                print(f"警告: 转换旋转向量 {rot_vec} (在主轨迹索引 {i} 处) 时发生未知错误: {e}. 跳过此姿态绘制。")
                r_matrix = np.eye(3)


            x_axis, y_axis, z_axis = r_matrix[:, 0], r_matrix[:, 1], r_matrix[:, 2]

            # ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], length=axis_length_main, color='r', normalize=True, zorder=3)
            # ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], length=axis_length_main, color='g', normalize=True, zorder=3)
            ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], length=axis_length_main, color='b', normalize=True, zorder=3)
    else:
        # 即使没有数据，也添加标签以保持图例一致性
        ax.plot([], [], [], label='Main Trajectory (Waypoints)', color='blue')


    # 2. 绘制特定点 (来自代码中定义的六个点, 仅XYZ位置)
    if specific_xyz_for_marker is not None and specific_xyz_for_marker.shape[0] > 0:
        all_points_for_scaling.append(specific_xyz_for_marker)
        ax.scatter(specific_xyz_for_marker[:, 0],
                   specific_xyz_for_marker[:, 1],
                   specific_xyz_for_marker[:, 2],
                   c='purple', # 更改颜色以便区分
                   marker='X',
                   s=120, # 稍微调大一点
                   label='Specific Hardcoded Points (XYZ)',
                   zorder=4) # 确保在最上层


    # 3. 绘制控制点 (来自 control_points.json, 已缩放)
    if control_points_xyz is not None and control_points_xyz.shape[0] > 0:
        all_points_for_scaling.append(control_points_xyz)
        ax.scatter(control_points_xyz[:, 0],
                   control_points_xyz[:, 1],
                   control_points_xyz[:, 2],
                   c='green',
                   marker='o',
                   s=60, # 稍微调大一点
                   label='Control Points (Scaled)',
                   zorder=2)

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory and Pose Visualization')

    # 自动调整坐标轴范围
    if all_points_for_scaling:
        all_points_np = np.vstack(all_points_for_scaling)
        min_vals = np.min(all_points_np, axis=0) - axis_length_main
        max_vals = np.max(all_points_np, axis=0) + axis_length_main
        
        centers = (min_vals + max_vals) / 2.0
        ranges = max_vals - min_vals
        max_plot_range = np.max(ranges) if ranges.size > 0 else 1.0 # 处理 ranges 为空的情况

        ax.set_xlim(centers[0] - max_plot_range / 2, centers[0] + max_plot_range / 2)
        ax.set_ylim(centers[1] - max_plot_range / 2, centers[1] + max_plot_range / 2)
        ax.set_zlim(centers[2] - max_plot_range / 2, centers[2] + max_plot_range / 2)
    
    try: # For matplotlib >= 3.3 for equal aspect ratio in 3D
        ax.set_box_aspect([1,1,1])
    except AttributeError:
        # Fallback for older Matplotlib (manual approximation for equal aspect)
        xlim = ax.get_xlim(); ylim = ax.get_ylim(); zlim = ax.get_zlim()
        max_abs_range = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]]).max() / 2.0
        mid_x = np.mean(xlim); mid_y = np.mean(ylim); mid_z = np.mean(zlim)
        ax.set_xlim(mid_x - max_abs_range, mid_x + max_abs_range)
        ax.set_ylim(mid_y - max_abs_range, mid_y + max_abs_range)
        ax.set_zlim(mid_z - max_abs_range, mid_z + max_abs_range)

    plt.show()


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 定义文件路径 ---
    waypoints_file_path = 'waypoints.json' # 您的主轨迹 JSON 文件名
    control_points_file_path = 'control_points.json' # 您的控制点 JSON 文件名

    # --- 直接定义您提供的六个特定位姿 (6 维列表) ---
    specific_poses_6d = [
[0.05692644290448469, 0.2602563835537891, 0.8034632565750283, 0.32381349852397345, -0.6454097411460297, 0.0],
[0.014939612416600478, 0.2841053039770923, 0.7935892771587907, 0.5737649899117029, -0.27094764963096457, 0.0],
[-0.09836321476819548, 0.27753627678318105, 0.7974945668947201, 0.748414907524227, 0.4234326374441012, 0.0],
[-0.18097206071795188, 0.19433410953001679, 0.8091947761143872, 0.2168698428218372, 0.7741347627311207, 0.0],
[-0.04199290164065913, 0.29959109602916134, 0.8016592280534424, 0.8223697524499909, 0.09568174352979462, 0.0],
[-0.15250512338900224, 0.24261978120223515, 0.8073660968905223, 0.5307016268628446, 0.6882114554536762, 0.0],
[-0.18783957472407367, 0.1364594997586289, 0.8159020450247432, -0.21803643976929382, 0.7094666623414327, 0.0]

    ]
    specific_xyz_coords_for_marker = np.array([ [float(p[0]), float(p[1]), float(p[2])] for p in specific_poses_6d ])

    main_poses_data = np.empty((0, 6)) # 初始化为空的6D数组
    control_points_xyz = np.empty((0,3)) # 初始化为空数组

    try:
        # 1. 读取并提取主轨迹的 6D位姿 (来自 waypoints.json) - 不缩放
        print(f"尝试读取并提取主轨迹6D位姿数据从: {waypoints_file_path} ...")
        if os.path.exists(waypoints_file_path):
            main_poses_data = read_and_extract_6d_poses(waypoints_file_path) # <<< 使用新的读取函数
            if main_poses_data.ndim == 2 and main_poses_data.shape[1] == 6:
                 print(f"成功提取 {main_poses_data.shape[0]} 个主轨迹6D位姿 (未使用缩放)。")
            elif main_poses_data.shape[0] == 0: # read_and_extract_6d_poses 返回空数组如果所有行都有问题
                 print(f"未能从 {waypoints_file_path} 提取任何有效的6D位姿数据。")
            else: # Should not happen if read_and_extract_6d_poses works as intended
                 print(f"警告: 从 {waypoints_file_path} 读取的数据格式不符合预期的 Nx6 格式，实际为: {main_poses_data.shape}。")
                 main_poses_data = np.empty((0, 6)) # 重置为空
        else:
            print(f"警告: 主轨迹文件 {waypoints_file_path} 未找到。将不绘制主轨迹。")

        # 2. 读取并提取控制点的 XYZ (来自 control_points.json) - 然后进行缩放
        print(f"\n尝试读取并提取控制点数据从: {control_points_file_path} ...")
        if os.path.exists(control_points_file_path):
            control_points_xyz_raw = read_and_extract_xyz_concise(control_points_file_path) # 旧函数仍适用
            if control_points_xyz_raw.shape[0] > 0:
                control_points_xyz = control_points_xyz_raw / 1000.0 # 在此处应用缩放
                print(f"成功提取 {control_points_xyz_raw.shape[0]} 个控制点，并已缩放 (除以1000)。")
            else:
                print(f"未能从 {control_points_file_path} 提取任何有效的控制点数据。")

        else:
            print(f"警告: 控制点文件 {control_points_file_path} 未找到。将不绘制控制点。")

        # 3. 调用新的可视化函数
        print("\n开始可视化...")
        skip_count_main = 1
        if main_poses_data.shape[0] > 20 : # 如果点很多，隔点绘制姿态
            skip_count_main = max(1, main_poses_data.shape[0] // 20) # 目标是绘制大约20个坐标系
        skip_count_main = 20
        visualize_trajectory_with_orientations(
            main_poses_6d=main_poses_data,
            specific_xyz_for_marker=specific_xyz_coords_for_marker,
            control_points_xyz=control_points_xyz,
            skip_frames_main=skip_count_main, # 动态调整跳帧
            axis_length_main=0.03 # 调整坐标轴的显示长度，根据你的场景大小
        )
        print("\n可视化完成 (如果窗口弹出)。")

    except json.JSONDecodeError as e:
        print(f"错误: JSON 解码失败。请检查文件格式是否正确。详细: {e}")
    except (IndexError, TypeError) as e: # ValueError 已在读取函数中部分处理
        print(f"错误: 文件中的数据格式或内容与预期不符。详细: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

    print("\n程序运行结束.")
