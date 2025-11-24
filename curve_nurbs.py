import numpy as np
from scipy import interpolate
import math
# import quaternion as quat
from pyquaternion import Quaternion
import json
#quat的输入数据格式也需要使用numpy的quaternion
def NURBSS(row, arr, quat, B_Arr, Quat):
    # 初始化数组
    D_X = [0.0] * row
    D_Y = [0.0] * row
    D_Z = [0.0] * row
    U = [0.0] * (row + 4)
    P_X = [0.0] * row
    P_Y = [0.0] * row
    P_Z = [0.0] * row
    W = [0.0] * row
    p = [0.0]*6
    # 获取型值点的位置坐标，机械臂在曲面采样的点或者是感知发给机械臂的
    for i in range(row):
        D_X[i] = 1000 * arr[i][0]
        D_Y[i] = 1000 * arr[i][1]
        D_Z[i] = 1000 * arr[i][2]

    # 计算B样条控制点
    counter_BA(D_X, D_Y, D_Z, row, P_X, P_Y, P_Z, U, W)
    print(U)
    control_points = list()
    for i in range(len(P_X)):
        control_points.append([P_X[i], P_Y[i], P_Z[i]])

    # 定义要保存的文件路径和名称
    file_name = 'control_points.json'
    # file_path = os.path.join('data', file_name) # 如果想保存到 data 子文件夹

    file_path = file_name # 示例保存到当前目录

    # 使用 try...except 块处理文件写入可能发生的错误
    try:
        # 'w' 模式表示写入文件，如果文件已存在会覆盖
        # encoding='utf-8' 是推荐的文件编码
        # indent=4 使 JSON 文件有缩进，更易读
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(control_points, f, indent=4)

        print(f"\ncontrol_points 数据已成功写入到文件: {file_path}")

    except TypeError as e:
        # 如果 control_points 中的元素包含非 JSON 可序列化的对象，会发生 TypeError
        print(f"写入 JSON 文件时发生 TypeError: {e}")
        print("请确保 control_points 列表中的所有元素（即每个 [X, Y, Z] 子列表）只包含 JSON 可序列化的数据类型（如数字、字符串、布尔值、None、列表、字典）。")
    except Exception as e:
        # 捕获其他可能的异常，如权限问题等
        print(f"写入 JSON 文件时发生错误: {e}")
    # 生成参数u
    pro = 0.001
    u = np.linspace(0, 1, int(1.0 / pro) + 1).tolist()

    n = len(u)
    s_m = [0.0]
    Tdd = []
    s = []

    # 计算弧长参数
    for i in range(n - 1):
        arcs = ComputeArcLengthOfBspline(row, P_X, P_Y, P_Z, U, u[i], u[i + 1])
        s_m.append(s_m[i] + arcs)
    St = s_m[-1]
    for val in s_m:
        s.append(val / St)
    s_m.clear()

    # 建立u-s五次多项式模型
    ret = polyFit(s, u, 5,p)
 
    # T型速度插补
    v0 = 0.0
    v1 = 0.0
    s0 = 0.0
    s1 = 1.0
    print('St',St)
    vmax = 10.0 / St#应该是毫米
    amax = 5.0 / St
    td = 0.1 # 时间间隔
    Ts = list()
    V_T = [0.0]*3
    n = TSpeedCruveMethod_1(s0, s1, v0, v1, vmax, amax, td,Ts,V_T)
    
    print('V_T',V_T)

    # 获取B样条离散点
    Tdd = []
    u_i = []
    x_Arr = [0.0] * 3

    for i in range(len(Ts)):
        ui = Ts[i][1]
        if ui > 1.0:
            ui = 1.0
        u_i.append(ui)
        Bspline(P_X, P_Y, P_Z, U, row, ui, x_Arr)
        
        Tdd.append([x / 1000.0 for x in x_Arr])

        # --- 修改开始：直接生成B样条曲线上的离散点 ---
    
    # # 定义要在曲线上生成的点的数量
    # num_curve_points = 1000  # 您可以根据需要的曲线平滑度调整这个值
    #                        # 原代码中 pro=0.001 会生成约1001个点用于弧长计算

    # # 生成用于评估B样条曲线的参数u值
    # # 假设U是标准 clamped knot vector, u的有效范围是 [0, 1]
    # # (counter_BA 中 U[0:4]=0.0 和 U[row:row+4]=1.0 保证了这一点，假设degree=3)
    # u_values_for_curve = np.linspace(0.0, 1.0, num_curve_points)

    # Tdd = []  # 用于存储最终的曲线点 (scaled down)
    # x_Arr = [0.0] * 3  # 用于从 Bspline 函数接收单点坐标 (scaled up)

    # print(f"\nGenerating {num_curve_points} points on the fitted B-spline curve...") # 英文提示

    # for u_param in u_values_for_curve:
    #     # 使用 counter_BA 计算得到的控制点 P_X, P_Y, P_Z 和节点矢量 U
    #     # row 是型值点/控制点的数量
    #     Bspline(P_X, P_Y, P_Z, U, row, u_param, x_Arr)
        
    #     # 将计算得到的点 (x_Arr 是在放大1000倍的尺度上的) 缩小1000倍并存储
    #     Tdd.append([x_coord / 1000.0 for x_coord in x_Arr])
    
    # print(f"Generated {len(Tdd)} points for the curve.") # 英文提示

    # 姿态插补
    Sr_m = [0.0]
    for i in range(row - 1):
        arcS = ComputeArcLengthOfBspline(row, P_X, P_Y, P_Z, U, W[i], W[i + 1])
        Sr_m.append(Sr_m[i] + arcS)

    Sr = [Sr / Sr_m[-1] for Sr in Sr_m]
    si = list() #里面存储的也是numpy的quaternion 格式为wxyz
    getSi(quat,si)

    for i in range(len(u_i)):
        for j in range(row - 1):
            if Sr[j] <= u_i[i] <= Sr[j + 1]:
                qi1 = quat[j]
                qi2 = quat[j + 1]
                si1 = si[j]
                si2 = si[j + 1]
                pra = (u_i[i] - Sr[j]) / (Sr[j + 1] - Sr[j])
                break
        Qi = squad(qi1, qi2, si1, si2, pra)
        Quat.append(Qi) #期望姿态和期望位置
        B_Arr.append([Tdd[i][0], Tdd[i][1], Tdd[i][2]])
    

    return 0
#给定型值点反算3次非均匀B样条控制点(自适应分割点)
#X,Y,Z输入坐标的数组，b_x,b_y,b_z,U输出数组，row输入数组的行数，row是型值点个数
#b_x,b_y,b_z为反算的样条控制点，U是节点矢量
def counter_BA(X, Y, Z, row, b_x, b_y, b_z, U, W):
    n = row
    N = np.zeros((2, 4), dtype=np.float64)
    Dydata = 0.0
    # 计算逻辑
    L = np.zeros((n - 1, 1), dtype=np.float64)
    w = np.zeros((n, 1), dtype=np.float64)
    du = np.zeros((n + 3, 1), dtype=np.float64)
    Lmd = np.zeros((n - 1, 1), dtype=np.float64)
    p = np.zeros((n, 1), dtype=np.float64)
    mu = np.zeros((n - 1, 1), dtype=np.float64)
    d_x = np.zeros((n, 1), dtype=np.float64)
    d_y = np.zeros((n, 1), dtype=np.float64)
    d_z  = np.zeros((n, 1), dtype=np.float64)
    b_Xm = np.zeros((n, 1), dtype=np.float64)
    b_Ym = np.zeros((n, 1), dtype=np.float64)
    b_Zm = np.zeros((n, 1), dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
    A_inve = np.zeros((n, n), dtype=np.float64)

    #累积弦长参数化，U为节点矢量，并将其归一化
    dx = np.diff(X)  # X[i+1] - X[i]
    dy = np.diff(Y)
    dz = np.diff(Z)
    distances = np.sqrt(dx**2 + dy**2 + dz**2) #第i个点和第i+1个点的距离
    L[:n-1, 0] = distances #型值点矢量
    w[0][0] = 0

    for i in range(1,n):
        w[i][0] = w[i - 1][0] + L[i - 1][0]
    total_length = w[n-1][0]
    if total_length < 1e-8:  # 避免零长度
        total_length = 1.0  
    for i in range(n):
        w[i][0] = w[i][0] /w[n-1][0]
    print("w[n][0]",w[n-1][0])
    # for i in range(n):
    #     w[i][0] = i / (n-1) if n > 1 else 0.0
    
    U[0:4] = [0.0, 0.0, 0.0, 0.0]
    j = 2
    for i in range(4, n):
        U[i] = w[j][0]  # NumPy 的二维索引写法
        j += 1
    # for i in range(4, n):
    #     # 直接使用归一化后的参数值作为内部节点
    #     U[i] = w[i-3][0]  # 修正索引，避免越界    
    U[n:n+4] = [1.0, 1.0, 1.0, 1.0]
    #利用3次B样条反算控制点，求解线性方程组得到d
    t1 = w[1][0]
    t2 = w[n - 2][0]

    for i in range(n + 3):
        du[i][0] = U[i + 1] - U[i]

    for i in range(2, n - 2):
        # 计算 Lmd[i][0]
        numerator = du[i + 2][0] * du[i + 2][0]
        denominator = du[i][0] + du[i + 1][0] + du[i + 2][0]
        Dydata = numerator / denominator
        Lmd[i][0] = Dydata

        # 计算 p[i][0]
        part1 = (du[i + 2][0] * (du[i][0] + du[i + 1][0])) / (du[i][0] + du[i + 1][0] + du[i + 2][0])
        part2 = (du[i + 1][0] * (du[i + 2][0] + du[i + 3][0])) / (du[i + 1][0] + du[i + 2][0] + du[i + 3][0])
        Dydata = part1 + part2
        p[i][0] = Dydata

        # 计算 mu[i][0]
        numerator = du[i + 1][0] * du[i + 1][0]
        denominator = du[i + 1][0] + du[i + 2][0] + du[i + 3][0]
        Dydata = numerator / denominator
        mu[i][0] = Dydata

        # 计算 d_x[i][0]
        Dydata = (du[i + 1][0] + du[i + 2][0]) * X[i]
        d_x[i][0] = Dydata

        # 计算 d_y[i][0]
        Dydata = (du[i + 1][0] + du[i + 2][0]) * Y[i]
        d_y[i][0] = Dydata

        # 计算 d_z[i][0]
        Dydata = (du[i + 1][0] + du[i + 2][0]) * Z[i]
        d_z[i][0] = Dydata
    # N[0][0]
    N[0][0] = ((U[4] - t1) / du[3][0]) ** 3

    # N[0][1]
    term1 = (t1 - U[3]) / du[3][0]
    term2 = ((U[4] - t1) / du[3][0]) ** 2
    term3 = (U[5] - t1) / (du[3][0] + du[4][0])
    term4 = (U[4] - t1) / du[3][0] + (U[5] - t1) / (du[3][0] + du[4][0])
    N[0][1] = term1 * (term2 + term3 * term4)

    # N[0][2]
    term1 = (t1 - U[3]) ** 2 / (du[3][0] * (du[3][0] + du[4][0]))
    term2 = (U[4] - t1) / du[3][0]
    term3 = (U[5] - t1) / (du[3][0] + du[4][0])
    term4 = (U[6] - t1) / (du[3][0] + du[4][0] + du[5][0])
    N[0][2] = term1 * (term2 + term3 + term4)

    # N[0][3]
    N[0][3] = ((t1 - U[3]) ** 3) / (du[3][0] * (du[3][0] + du[4][0]) * (du[3][0] + du[4][0] + du[5][0]))

    # N[1][0]
    N[1][0] = ((U[n] - t2) ** 3) / (du[n-1][0] * (du[n-2][0] + du[n-1][0]) * (du[n-3][0] + du[n-2][0] + du[n-1][0]))

    # N[1][1]
    term1 = ((U[n] - t2) ** 2) / (du[n-1][0] * (du[n-2][0] + du[n-1][0]))
    term2 = (t2 - U[n-3]) / (du[n-3][0] + du[n-2][0] + du[n-1][0])
    term3 = (t2 - U[n-2]) / (du[n-2][0] + du[n-1][0])
    term4 = (t2 - U[n-1]) / du[n-1][0]
    N[1][1] = term1 * (term2 + term3 + term4)

    # N[1][2]
    term1 = (U[n] - t2) / du[n-1][0]
    term2 = (t2 - U[n-2]) / (du[n-2][0] + du[n-1][0])
    term3 = term2 + (t2 - U[n-1]) / du[n-1][0]
    term4 = ((t2 - U[n-1]) / du[n-1][0]) ** 2
    N[1][2] = term1 * (term2 * term3 + term4)

    # N[1][3]
    N[1][3] = ((t2 - U[n-1]) / du[n-1][0]) ** 3

    Lmd[1][0] = mu[2][0] * N[0][0]
    p[1][0] = mu[2][0] * N[0][1] - Lmd[2][0] * N[0][3]
    mu[1][0] = mu[2][0] * N[0][2] - p[2][0] * N[0][3]
    Lmd[n-2][0] = Lmd[n-3][0] * N[1][1] - p[n-3][0] * N[1][0]
    p[n-2][0] = Lmd[n-3][0] * N[1][2] - mu[n-3][0] * N[1][0]
    mu[n-2][0] = Lmd[n-3][0] * N[1][3]
    d_x[1][0] = mu[2][0] * X[1] - (du[3][0] + du[4][0]) * N[0][3] * X[2]
    d_y[1][0] = mu[2][0] * Y[1] - (du[3][0] + du[4][0]) * N[0][3] * Y[2]
    d_z[1][0] = mu[2][0] * Z[1] - (du[3][0] + du[4][0]) * N[0][3] * Z[2]
    d_x[n-2][0] = Lmd[n-3][0] * X[n-2] - (du[n-2][0] + du[n-1][0]) * N[1][0] * X[n-3]
    d_y[n-2][0] = Lmd[n-3][0] * Y[n-2] - (du[n-2][0] + du[n-1][0]) * N[1][0] * Y[n-3]
    d_z[n-2][0] = Lmd[n-3][0] * Z[n-2] - (du[n-2][0] + du[n-1][0]) * N[1][0] * Z[n-3]

    
    d_x[0][0] = X[0]
    d_x[n-1][0] = X[n-1]

    d_y[0][0] = Y[0]
    d_y[n-1][0] = Y[n-1]

    d_z[0][0] = Z[0]
    d_z[n-1][0] = Z[n-1]

    p[0][0] = 1
    p[n-1][0] = 1
    mu[0][0] = 0

    for i in range(n - 2):
        Lmd[i][0] = Lmd[i + 1][0]

    Lmd[n-2][0] = 0

    for i in range(n):
        for j in range(n):
            if i > 0 and j == i - 1:
                A[i][j] = Lmd[i - 1][0]
            elif j > 0 and j == i + 1:
                A[i][j] = mu[i][0]
            elif j == i:
                A[i][j] = p[i][0]
            else:
                A[i][j] = 0
    A_inve = np.linalg.inv(A)
    b_Xm = A_inve @ d_x
    b_Ym = A_inve @ d_y
    b_Zm = A_inve @ d_z

    #解得的控制点坐标存储到一维数组 b_x, b_y, b_z 中
    for i in range(n):
        b_x[i] = b_Xm[i][0]
        b_y[i] = b_Ym[i][0]
        b_z[i] = b_Zm[i][0]
        W[i] = w[i][0] #第 i 个原始数据点所对应的、经过归一化的累积弦长参数值。
    return 0

def ComputeArcLengthOfBspline(row, X, Y, Z, U, a, b):
    f0, f1, f2, f3, f4, s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 微调 a 和 b，避免超出 [0, 1] 范围
    if a != 0:
        a = (a * 1.0e8 - 1.0) / 1.0e8
    if b != 1:
        b = (b * 1.0e8 - 1.0) / 1.0e8

    # 将区间 [a, b] 分成 4 份，共 5 个点
    h = (b - a) / 4
    x0 = a
    x1 = a + h
    x2 = x1 + h
    x3 = x2 + h
    x4 = b

    # 计算各点的切线向量模长
    f0 = ComTanVecBspline(row, X, Y, Z, U, x0)
    f1 = ComTanVecBspline(row, X, Y, Z, U, x1)
    f2 = ComTanVecBspline(row, X, Y, Z, U, x2)
    f3 = ComTanVecBspline(row, X, Y, Z, U, x3)
    f4 = ComTanVecBspline(row, X, Y, Z, U, x4)

    # 使用 Simpson's 1/3 规则计算弧长
    s = 2 * h * (7 * f0 + 32 * f1 + 12 * f2 + 32 * f3 + 7 * f4) / 45

    return s

def ComTanVecBspline(row, X, Y, Z, U, u):
    p = 3  # 固定阶数
    n = row  # 控制点数量

    # 初始化结果数组
    result = np.zeros(3)  # 存储切线向量的三个分量 (X, Y, Z)
    f = 0.0  # 切线向量的长度

    # 使用 NumPy 数组代替动态分配的指针
    T_X = np.zeros(n - 1)
    T_Y = np.zeros(n - 1)
    T_Z = np.zeros(n - 1)
    T_U = np.zeros(n + p - 1)

    # 计算 T_X, T_Y, T_Z
    for i in range(n - 1):
        delta_X = X[i + 1] - X[i]
        delta_U = U[i + p + 1] - U[i + 1]
        T_X[i] = p * delta_X / delta_U
        T_Y[i] = p * (Y[i + 1] - Y[i]) / delta_U
        T_Z[i] = p * (Z[i + 1] - Z[i]) / delta_U

    # 填充 T_U
    for i in range(n + p - 1):
        T_U[i] = U[i + 1]

    # 调用 TanVecBspline 函数（假设已实现）
    TanVecBspline(n - 1, T_X, T_Y, T_Z, T_U, u, result)

    # 计算切线向量的模长
    f = np.sqrt(np.sum(result**2))

    return f

def TanVecBspline(row, X, Y, Z, U, u, result):
    k = 2  # 固定值
    n = row  # 控制点数量
    C = 0  # 参数 u 所在的区间索引

    # 找到参数 u 所在的区间 [U[i], U[i+1]]
    for i in range(k, n):
        if U[i + 1] >= u and u >= U[i]:
            C = i + 1

    x = y = z = 0.0

    # 计算切线向量的分量
    for m in range(C - k, C + 1):  # 注意：C+1 是为了包含 m <= C
        N_ = TanVecB_basis(u, U, m - 1)  # 计算基函数
        x += X[m - 1] * N_
        y += Y[m - 1] * N_
        z += Z[m - 1] * N_

    # 存储结果
    result[0] = x
    result[1] = y
    result[2] = z
def TanVecB_basis(u, U, i):
    # 初始化 3x3 矩阵（全0）
    N = np.zeros((3, 3))

    # 设置初始值（第0列）
    if U[i] <= u <= U[i + 1]:
        N[0, 0] = 1
    else:
        N[0, 0] = 0

    if U[i + 1] <= u <= U[i + 2]:
        N[1, 0] = 1
    else:
        N[1, 0] = 0

    if U[i + 2] <= u <= U[i + 3]:
        N[2, 0] = 1
    else:
        N[2, 0] = 0

    # 计算递归公式
    for j in range(1, 3):  # j: 1, 2
        for k in range(3 - j):  # k: 0, 1 (当 j=1 时)
            deta1 = U[i + j + k] - U[i + k]
            deta2 = U[i + j + k + 1] - U[i + k + 1]

            # 避免除以0
            A = (u - U[i + k]) / deta1 if deta1 != 0 else 0
            B1 = (U[i + j + k + 1] - u) / deta2 if deta2 != 0 else 0

            # 递归计算基函数
            N[k, j] = A * N[k, j - 1] + B1 * N[k + 1, j - 1]

    # 强制设置某些位置为0
    N[2, 1] = 0
    N[2, 2] = 0
    N[1, 2] = 0

    # 返回最终结果（N[0, 2]）
    return N[0, 2]

def polyFit(x, y, k, A):
    n = len(x)

    # 初始化矩阵
    Y = np.zeros((n, 1))
    U = np.zeros((n, k + 1))
    trans_U = np.zeros((k + 1, n))
    M_U = np.zeros((k + 1, k + 1))
    Inv_U = np.zeros((k + 1, k + 1))
    TM_U = np.zeros((k + 1, n))
    M_Y = np.zeros((k + 1, 1))

    # 填充 Y 矩阵
    for i in range(n):
        Y[i, 0] = y[i]

    # 填充 U 矩阵
    for i in range(n):
        for j in range(k + 1):
            U[i, j] = x[i] ** j

    # 转置 U 得到 trans_U
    trans_U = U.T

    # 计算 M_U = trans_U @ U
    M_U = trans_U @ U

    # 计算 M_U 的逆矩阵
    Inv_U = np.linalg.inv(M_U)

    # 计算 TM_U = Inv_U @ trans_U
    TM_U = Inv_U @ trans_U

    # 计算 M_Y = TM_U @ Y
    M_Y = TM_U @ Y

    # 将结果写入 A 数组
    for j in range(k + 1):
        A[j] = M_Y[j, 0]

    return k + 1  # 与原 C++ 函数保持一致

def TSpeedCruveMethod_1(q0, q1, v0, v1, vmax, amax, td, result, V_T):
    v_temp = math.sqrt((2.0 * amax * (q1 - q0) + (v1 * v1 + v0 * v0)) / 2)
    vlim = min(v_temp, vmax)

    Ta = (vlim - v0) / amax
    Sa = v0 * Ta + amax * Ta * Ta / 2
    Tv = (q1 - q0 - (vlim * vlim - v0 * v0) / (2 * amax) - (v1 * v1 - vlim * vlim) / (2 * -amax)) / vlim
    Sv = vlim * Tv
    Td = (vlim - v1) / amax
    Sd = vlim * Td - amax * Td * Td / 2
    T = Ta + Tv + Td

    # 更新 V_T（必须是长度 >= 3 的可变列表）
    V_T[0] = Ta
    V_T[1] = Tv
    V_T[2] = Td

    k = 1
    Tdd = []

    while k * td < T:
        time = td * k
        t = time  # 当前时间

        if t >= 0 and t < Ta:
            q = q0 + v0 * t + amax * t * t / 2
            qd = v0 + amax * t
            qdd = amax
        elif t >= Ta and t < Ta + Tv:
            q = q0 + Sa + vlim * (t - Ta)
            qd = vlim
            qdd = 0
        elif t >= Ta + Tv and t <= T:
            delta_t = t - Ta - Tv
            q = q0 + Sa + Sv + vlim * delta_t - amax * delta_t * delta_t / 2
            qd = vlim - amax * delta_t
            qdd = -amax
        else:
            q = q1
            qd = 0
            qdd = 0

        Tdd = [time, q, qd, qdd]
        result.append(Tdd)
        k += 1

    return k - 1  # 返回实际使用的时间步数
def find_knot_interval(U, u, k=3):
    """查找参数 u 所在的节点区间 [U[i], U[i+1]]"""
    # U 长度应为 n + k + 1（n:控制点数量，k:次数）
    n = len(U) - k - 1  
    if u >= U[-1]:
        return n - 1  # 强制最后一个有效区间
    low, high = k, n  # clamped B样条的有效区间是 [U[k], U[n]]
    while high - low > 1:
        mid = (low + high) // 2
        if U[mid] > u:
            high = mid
        else:
            low = mid
    return low
def Bspline(X, Y, Z, U, row, u, Arr):
    k = 3  # 三次样条
    n = row

    # 初始化矩阵 T (k+1 行 × n+3 列)
    T = np.zeros((k + 1, n + 3))

    # 找到 u 所在的区间 [U[i], U[i+1]]
    i = 0
    for Count in range(k, n):
        if U[Count + 1] >= u and u >= U[Count]:
            T[0, Count] = 1.0
            i = Count
    PP = i
    #PP = find_knot_interval(U, u, k=3)
    # 递归填充矩阵 T
    for j in range(1, k + 1):  # 阶数从 1 到 k
        for Count in range(PP - k, PP + 1):  # Count 的范围
            for i in range(Count, Count + k + 1 - j):  # i 的范围
                denom1 = U[i + j] - U[i]
                denom2 = U[i + j + 1] - U[i + 1]

                if denom1 == 0:
                    if denom2 == 0:
                        T[j, i] = 0.0
                    else:
                        T[j, i] = (U[i + j + 1] - u) / denom2 * T[j - 1, i + 1]
                else:
                    if denom2 == 0:
                        T[j, i] = (u - U[i]) / denom1 * T[j - 1, i]
                    else:
                        T[j, i] = (
                            (u - U[i]) / denom1 * T[j - 1, i] +
                            (U[i + j + 1] - u) / denom2 * T[j - 1, i + 1]
                        )

    # 计算最终的点坐标
    x = y = z = 0.0
    # for m in range(PP - k, PP):  # m 的范围
    #     x += X[m] * T[k, m]
    #     y += Y[m] * T[k, m]
    #     z += Z[m] * T[k, m]
    for m in range(PP - k, PP + 1):  # m 的范围
        x += X[m] * T[k, m]
        y += Y[m] * T[k, m]
        z += Z[m] * T[k, m]

    # 存储结果
    Arr[0] = x
    Arr[1] = y
    Arr[2] = z

    return 0
# def Bspline(X, Y, Z, U, row, u, Arr):
#     k = 3  # 三次样条
#     n = row  # 控制点数量（索引 0~n-1）
#     # 查找 u 所在区间
#     PP = find_knot_interval(U, u, k)
#     # De Boor 算法计算基函数权重
#     alpha = np.zeros(k+1)
#     alpha[0] = 1.0
#     for j in range(1, k+1):
#         # 计算左分母
#         left_den = U[PP + 1 - k + j] - U[PP + 1 - k]
#         # 计算右分母
#         right_den = U[PP + j + 1] - U[PP + 1]
#         # 临时数组保存上一层结果
#         prev_alpha = alpha.copy()
#         alpha[0] = 0.0
#         for i in range(j):
#             # 左权重
#             if left_den > 1e-8:
#                 left_w = (u - U[PP + 1 - k + i]) / left_den
#             else:
#                 left_w = 0.0
#             # 右权重
#             if right_den > 1e-8:
#                 right_w = (U[PP + 1 + i] - u) / right_den
#             else:
#                 right_w = 0.0
#             alpha[i+1] = left_w * prev_alpha[i] + right_w * prev_alpha[i+1]
#     # 计算曲线点
#     x = y = z = 0.0
#     for i in range(k+1):
#         ctrl_idx = PP - k + i
#         if 0 <= ctrl_idx < n:
#             x += X[ctrl_idx] * alpha[i]
#             y += Y[ctrl_idx] * alpha[i]
#             z += Z[ctrl_idx] * alpha[i]
#     Arr[0], Arr[1], Arr[2] = x, y, z

# def Bspline(X, Y, Z, U, row, u, Arr):
#     k = 3  # 三次样条
#     n = row # n 通常指的是控制点的数量，或者最大索引+1。如果 row 是控制点数量, 那么控制点索引是 0 到 row-1

#     # 初始化矩阵 T (k+1 行 × n+3 列) - 这里的列数 n+3 假设 n 是控制点数, 节点数是 n+k+1
#     # 实际上，T 矩阵的维度可以更灵活，只需要存储计算过程中需要的值。
#     # 更常见的 De Boor 算法实现是直接计算基函数值，而不是填充一个大的 T 矩阵。
#     # 但我们先基于您当前的 T 矩阵结构来分析。
#     # 假设 T[j, i] 存储的是 N_{i,j}(u) 基函数的值。
#     T = np.zeros((k + 1, n + k + 1)) # 节点矢量U的长度通常是 n+k+1，所以T的第二维最大索引可能是 n+k

#     # 找到 u 所在的区间 [U[i], U[i+1]]
#     # 这个查找方法对于 u=1.0 的情况可能需要特别处理，因为它可能落在最后一个区间的右边界
#     # 并且 PP 的值应该是使得 U[PP] <= u < U[PP+1] 的那个 PP
#     # 如果 u == U[n] (最后一个有效区间的右端点，通常是1.0对于clamped样条)，
#     # 那么 PP 应该取 n-1，这样 U[n-1] <= u <= U[n] (假设最后一个控制点是P_{n-1})
#     PP = 0
#     # 控制点数量为 n (即 X, Y, Z 的长度是 n, 索引从 0 到 n-1)
#     # 节点矢量 U 的长度是 n + k + 1
#     # 有效的 u 范围是 U[k] 到 U[n]
#     if u == U[n]: # 特殊处理 u 等于最后一个有效节点的情况 (通常是1.0)
#         PP = n - 1 # 最后一个基函数 N_{n-1,k}(u) 将非零
#     else:
#         for i_knot in range(k, n): # 遍历可能的节点区间起始索引
#             if U[i_knot] <= u < U[i_knot + 1]:
#                 PP = i_knot
#                 break
    
#     # --- De Boor 算法的更标准实现方式 ---
#     # 我们需要计算 N_{i,0}(u), ..., N_{i,k}(u)
#     # 通常会用一个局部数组 alpha[k+1]
#     # alpha[j] = N_{PP-k+j, k}(u)
    
#     # 以下是尝试修正您现有 T 矩阵填充逻辑的思路，但这部分非常复杂且易错
#     # 您原代码中 T[0, Count] = 1.0 的逻辑是 De Boor 算法计算0次基函数 N_{i,0} 的基础
#     # 如果 U[Count] <= u < U[Count+1]，则 N_{Count,0}(u) = 1，否则为0。
#     # 您的 T[0,Count] 似乎是直接对应这个 Count，如果Count是PP，则 T[0,PP]=1

#     # 重新审视 T 矩阵的填充：
#     # T[j, i] 应该是 N_{i, j}(u)
#     # 0次基函数 (j=0):
#     for i_idx in range(n + k): # 遍历所有可能的0次基函数（对应每个节点区间）
#         if U[i_idx] <= u < U[i_idx+1]: # 注意这里是严格小于右边界，除非 u 是最后一个点
#              T[0, i_idx] = 1.0
#         elif u == U[n] and i_idx == n-1 : # 特殊处理 u = U[n] (通常是1.0)
#              T[0, n-1] = 1.0 # N_{n-1,0}(1.0)=1 if U[n-1] <= 1.0 < U[n] (with U[n]=1.0, U[n+1]=1.0 for clamped)
#         else:
#              T[0, i_idx] = 0.0
    
#     # 递归计算更高次的基函数 (j 从 1到k)
#     for j_degree in range(1, k + 1):      # j_degree 是当前计算的基函数次数
#         for i_ctrl_idx in range(n + k - j_degree): # i_ctrl_idx 是基函数 N_{i, j_degree} 的索引 i
#             denom1 = U[i_ctrl_idx + j_degree] - U[i_ctrl_idx]
#             term1 = 0.0
#             if denom1 != 0:
#                 term1 = (u - U[i_ctrl_idx]) / denom1 * T[j_degree - 1, i_ctrl_idx]

#             denom2 = U[i_ctrl_idx + j_degree + 1] - U[i_ctrl_idx + 1]
#             term2 = 0.0
#             if denom2 != 0:
#                 term2 = (U[i_ctrl_idx + j_degree + 1] - u) / denom2 * T[j_degree - 1, i_ctrl_idx + 1]
            
#             T[j_degree, i_ctrl_idx] = term1 + term2

#     # 计算最终的点坐标
#     # 对于参数 u，有贡献的基函数是 N_{PP-k,k}(u), N_{PP-k+1,k}(u), ..., N_{PP,k}(u)
#     # 对应的控制点是 P_{PP-k}, P_{PP-k+1}, ..., P_{PP}
#     # 假设 X, Y, Z 的长度是 'row' (即 n 个控制点, 索引 0 to n-1)
#     # PP 是 u 所在主要区间的起始节点索引 (U[PP] <= u < U[PP+1])
#     # (或者 PP 是使得 N_{PP,k}(u) 可能是主要贡献者的控制点索引的“中心点”)

#     x = y = z = 0.0
#     # 循环应该遍历那些对当前 u 非零的基函数所对应的控制点
#     # 对于阶数为k的B样条，在参数u处，最多有 k+1 个 N_{i,k}(u) 非零。
#     # 这些非零基函数的索引 i 的范围是 [PP-k, PP] (使用上述修正后的PP定义)
    
#     # 修正后的循环 (m 是控制点的索引):
#     for m_ctrl_idx in range(max(0, PP - k), min(n, PP + 1)): # 确保 m_ctrl_idx 在有效控制点索引 [0, n-1] 内
#                                                         # 并且是可能对u有贡献的控制点
#         # T[k, m_ctrl_idx] 应该是 N_{m_ctrl_idx, k}(u) 的值
#         # 检查 T 矩阵的第二维是否确实对应控制点索引 m_ctrl_idx
#         if m_ctrl_idx < n : # 确保不越界访问 X, Y, Z
#             basis_val = T[k, m_ctrl_idx] # 获取 N_{m_ctrl_idx, k}(u)
#             x += X[m_ctrl_idx] * basis_val
#             y += Y[m_ctrl_idx] * basis_val
#             z += Z[m_ctrl_idx] * basis_val

#     # 存储结果
#     Arr[0] = x
#     Arr[1] = y
#     Arr[2] = z

#     return 0

# #C++代码实现
# def getSi(q_list, s_list):
#     M = len(q_list)
#     for i in range(M):
#         if i == 0:
#             s_list.append(q_list[0])
#         elif i == M - 1:
#             s_list.append(q_list[-1])
#         else:
#             # 计算 logQuat(q[i] * q[i-1]) 和 logQuat(q[i] * q[i+1])
#             log1 = logQuat(q_list[i] * q_list[i - 1])
#             log2 = logQuat(q_list[i] * q_list[i + 1])
#             # 对系数进行线性组合
#             temp = -(log1 + log2) / 4
#             # 计算最终结果
#             s_list.append(q_list[i] * expQuat(temp))
#     return 0

def getSi(q_list, s_list):
    """
    计算SQUAD插值所需的控制四元数 s_i。
    使用 numpy.quaternion 库的内置运算。

    参数:
    - q_list: 包含关键帧单位四元数 (quat.quaternion 对象) 的列表。
    - s_list: 一个空列表，此函数将用计算出的 s_i 四元数填充它。
              s_i 与 q_i 一一对应。
    """
    M = len(q_list)

    if M == 0:
        return 0 # 没有关键帧，s_list 保持为空

    # 清空 s_list 以防外部传入非空列表（或者由调用者保证）
    # s_list.clear() # 如果需要确保 s_list 从空开始

    for i in range(M):
        if M == 1: #只有一个点的情况
            s_list.append(q_list[0])
            break # 直接结束循环
        
        # 端点条件
        if i == 0:
            s_list.append(q_list[0])
        elif i == M - 1:
            s_list.append(q_list[M - 1])
        else:
            # 中间点 s_i 的计算
            # s_i = q_i * exp( -(log(q_i^{-1} * q_{i-1}) + log(q_i^{-1} * q_{i+1})) / 4 )
            
            q_i = q_list[i]
            q_i_minus_1 = q_list[i-1]
            q_i_plus_1 = q_list[i+1]

            # 1. 计算相对旋转
            # q_i_inv = q_i.inverse() # numpy.quaternion 有 .inverse() 方法
            # delta_prev = q_i_inv * q_i_minus_1
            # delta_next = q_i_inv * q_i_plus_1
            
            # 或者更直接（q_i.inverse() * A 等价于 q_i.conj() * A 如果q_i是单位四元数，
            # 但 .inverse() 更通用且安全）
            # numpy-quaternion 乘法 * 是 Hamilton product.
            # (A * B^{-1}) 表示 B 旋转后应用 A 旋转。
            # 我们需要从 q_i 到 q_{i-1} 的旋转，即 delta_prev * q_i = q_{i-1} => delta_prev = q_{i-1} * q_i.inverse()
            # 或者，如果 q_i * delta_prev = q_{i-1} => delta_prev = q_i.inverse() * q_{i-1} (常用定义)

            # 标准定义中，log的参数是使得 q_i * result = q_{i-1} (或 q_{i+1}) 的那个旋转。
            # 即 q_i^{-1} * q_{i-1}
            
            q_i_inv = q_i.inverse
            delta_prev = q_i_inv * q_i_minus_1 
            delta_next = q_i_inv * q_i_plus_1

            # 2. 计算对数
            # numpy.quaternion.log(q) 返回一个新的纯四元数 (w=0)，其向量部分 v 代表 angle*axis
            log_delta_prev = Quaternion.log(delta_prev)
            log_delta_next = Quaternion.log(delta_next)

            # 3. 对数求和并缩放
            # log_delta_prev 和 log_delta_next 都是纯四元数 (标量部分为0)
            # 它们的和也是纯四元数
            sum_of_logs = log_delta_prev + log_delta_next
            
            # exp 的参数
            # sum_of_logs 是纯四元数，除以标量后仍然是纯四元数
            arg_for_exp = -sum_of_logs / 4.0

            # 4. 计算指数
            # numpy.quaternion.exp(p) 当 p 是纯四元数 (p.w=0, p.vec=v) 时,
            # 返回 (cos|v|, v/|v| * sin|v|)
            exp_component = Quaternion.exp(arg_for_exp)

            # 5. 计算 s_i
            s_i = q_i * exp_component
            s_list.append(s_i)
            
    
def squad(q1: Quaternion, q2: Quaternion, s1: Quaternion, s2: Quaternion, t: float) -> Quaternion:
    """
    Squad 四元数插值
    - q1, q2: 主路径上的两个点
    - s1, s2: 辅助方向四元数
    - t: 插值参数，范围 [0, 1]
    - return: 插值结果
    """
    # 第一次 SLERP（主路径）
    slerp1 = slerp(q1, q2, t)
    # # 第二次 SLERP（辅助方向）
    slerp2 = slerp(s1, s2, t)
    # 最终 SLERP，参数为 2 * t * (1 - t)
    t_final = 2 * t * (1 - t)
    squ = slerp(slerp1, slerp2, t_final)
    return squ


def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """
    四元数球面线性插值（Spherical Linear Interpolation）
    - q1, q2: 单位四元数
    - t: 插值参数，范围 [0, 1]
    - return: 插值结果（单位四元数）
    """
    # 计算点积（cosθ）
    cos_a = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

    # 判断是否接近线性插值（角度很小）
    if abs(cos_a) > 0.9995:
        a_t = 1.0 - t
        b_t = t
    else:
        # 计算 sinθ 和角度 θ
        sin_a = np.sqrt(1.0 - cos_a * cos_a)
        a = np.arctan2(sin_a, cos_a)
        a_t = np.sin((1.0 - t) * a) / sin_a
        b_t = np.sin(t * a) / sin_a

    # 线性插值（四元数加权和）
    result = q1 * a_t + q2 * b_t

    return result