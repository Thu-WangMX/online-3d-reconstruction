import pyrealsense2 as rs
import numpy as np
import cv2
# 配置深度和彩色流
pipeline = rs.pipeline()
config = rs.config()
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
#config.enable_stream(rs.stream.accel)  # 加速度计
#config.enable_stream(rs.stream.gyro)   # 陀螺仪

# 开始流
pipeline.start(config)

try:
    print("RealSense 相机已启动，采集数据中...")
    
    for i in range(30):  # 跳过前30帧
        pipeline.wait_for_frames()
    
    frames = pipeline.wait_for_frames()
    
    # 获取深度帧
    depth_frame = frames.get_depth_frame()
    if depth_frame:
        print(f"深度图尺寸: {depth_frame.width}x{depth_frame.height}")
    
    # 获取彩色帧
    color_frame = frames.get_color_frame()
    if color_frame:
        print(f"彩色图尺寸: {color_frame.width}x{color_frame.height}")
    
    # 获取IMU数据
    accel_frame = frames.first_or_default(rs.stream.accel)
    gyro_frame = frames.first_or_default(rs.stream.gyro)
    
    if accel_frame:
        accel_data = accel_frame.as_motion_frame().get_motion_data()
        print(f"加速度: x={accel_data.x:.3f}, y={accel_data.y:.3f}, z={accel_data.z:.3f}")
    
    if gyro_frame:
        gyro_data = gyro_frame.as_motion_frame().get_motion_data()
        print(f"陀螺仪: x={gyro_data.x:.3f}, y={gyro_data.y:.3f}, z={gyro_data.z:.3f}")

finally:
    pipeline.stop()
    print("相机已停止")





# # 1. 初始化 RealSense 管道
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置 RGB 流
# pipeline.start(config)

# # 2. 获取一帧 RGB 图像
# frames = pipeline.wait_for_frames()
# color_frame = frames.get_color_frame()
# image_bgr = np.asanyarray(color_frame.get_data())  # 转成 numpy 数组（BGR 格式）
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # 转 RGB

# # 3. 显示图像
# cv2.imshow("RealSense RGB", image_rgb)
# cv2.waitKey(0)
# pipeline.stop()
# cv2.destroyAllWindows()