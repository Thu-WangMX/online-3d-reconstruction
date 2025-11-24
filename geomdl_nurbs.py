from geomdl import fitting
from geomdl.visualization import VisMPL
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 整理后的三维型值点（假设每组前3个值为 x, y, z）
points = [
[0.06766290552030202, 0.2715512478770816, 0.8181063155650825],
[0.03546209023499848, 0.30180016289061073, 0.817081877688405],
[-0.0715658276710901, 0.3058724211794731, 0.8184530764446402],
[-0.03309817063941522, 0.312613954188525, 0.8168111385905102],
[-0.10173149299161978, 0.29134307892961286, 0.8173995506760574],
[-0.15945437838755572, 0.2465387757921505, 0.8172775651171711],
[0.008542787903894733, 0.3113049819283563, 0.8172173295059708],
[-0.18423422035189957, 0.20808560140127547, 0.817103596706813],
[-0.18320579355615113, 0.16428633531241235, 0.8080400702574776],
[-0.1601161559407691, 0.13192375285868876, 0.7968369820088341]
]

# 使用 geomdl 进行曲线拟合（反求控制点和节点向量）
degree = 3  # 三次 B 样条
curve = fitting.approximate_curve(points, degree)

# 输出节点向量和控制点
print("自动生成的节点向量:", curve.knotvector)
print("计算得到的控制点数量:", len(curve.ctrlpts))
print("前3个控制点坐标:", curve.ctrlpts[:3])

# 提高曲线采样密度
curve.delta = 0.0001  # 采样间隔，值越小采样点越多
curve_points = curve.evalpts

# 可视化结果
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制型值点（原始数据）
ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points],
           c='red', s=80, marker='o', label='型值点')

# 绘制控制点
ax.scatter([p[0] for p in curve.ctrlpts], [p[1] for p in curve.ctrlpts],
           [p[2] for p in curve.ctrlpts], c='green', s=50, marker='x', label='控制点')

# 绘制生成的曲线
ax.plot([p[0] for p in curve_points], [p[1] for p in curve_points],
        [p[2] for p in curve_points], 'b-', linewidth=2, label='NURBS 拟合曲线')

# 添加数据标签（显示型值点编号）
for i, point in enumerate(points):
    ax.text(point[0], point[1], point[2], f'P{i}', fontsize=9)

ax.set_xlabel('X 轴')
ax.set_ylabel('Y 轴')
ax.set_zlabel('Z 轴')
ax.set_title('NURBS 曲线拟合结果')
ax.legend()
plt.grid(True)
plt.show()