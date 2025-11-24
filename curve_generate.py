from curve_nurbs import NURBSS
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
from pyquaternion import Quaternion
def rotation_vector_to_quaternion_scipy(rot_vec_list):
    """
    使用 scipy 将长度为3的列表（旋转向量）转换为 numpy-quaternion 对象。

    参数:
    rot_vec_list (list): 长度为3的列表 [rx, ry, rz]，表示旋转向量。

    返回:
    numpy.quaternion: 表示旋转的四元数对象。
    """
    # SciPy 的 Rotation.from_rotvec 需要一个 numpy 数组
    rot_vec_np = np.array(rot_vec_list, dtype=float)

    # 从旋转向量创建 Rotation 对象
    rotation = R.from_rotvec(rot_vec_np)

    # 将其转换为四元数 (x, y, z, w 顺序)
    quat_scipy = rotation.as_quat()

    # 转换为Quaternion 对象 (w, x, y, z 顺序)
    # 注意 scipy 的 as_quat() 返回的是 [x, y, z, w]
    # numpy.quaternion 的构造函数是 (w, x, y, z)
    return Quaternion(quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2])
# --- 读取数据 ---
# 每次读取一行,读取全部的姿态
def read_poses_jsonl_line_by_line(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip(): # 确保行不为空
                pose = json.loads(line) # 将JSON字符串转回Python list
                print(f"读取到的单行姿态: {pose}")
                poses.append(pose)
    return poses

filename_jsonl = "robot_target_poses.jsonl"
all_read_poses_jsonl = read_poses_jsonl_line_by_line(filename_jsonl)
row = len(all_read_poses_jsonl)
#存储nurbss生成的数据
B_Arr = list()
Quat = list()
arr = list()
rotation_vectors = list()
quat = list()
for pose in all_read_poses_jsonl:
    arr.append(pose[:3])
    rotation_vectors.append(pose[3:])
print(arr)

for vector in rotation_vectors:
    quat.append(rotation_vector_to_quaternion_scipy(vector))
print("111111")
NURBSS(row,arr,quat,B_Arr,Quat)

print("111111")
waypoint_num = len(B_Arr)
waypoints = list()
print(waypoint_num)
for i in range(waypoint_num):
    q_scipy = R.from_quat([Quat[i].x,Quat[i].y,Quat[i].z,Quat[i].w]) #xyzw
    waypoint = B_Arr[i] +  list(q_scipy.as_rotvec())
    waypoints.append(waypoint)

# 指定要写入的 JSON 文件名
file_path = 'waypoints.json'
print("111111")

try:
    with open(file_path, 'w') as f:
        json.dump(waypoints, f, indent=4)  # 使用 indent=4 可以使 JSON 文件更易读
    print(f"数据已成功写入到文件: {file_path}")
except TypeError as e:
    print(f"写入 JSON 文件时发生 TypeError: {e}")
    print("请确保 waypoints 列表中的所有元素都是 JSON 可序列化的。")
except Exception as e:
    print(f"写入 JSON 文件时发生错误: {e}")