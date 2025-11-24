import pyrealsense2 as rs
import numpy as np
import cv2

# 全局变量来存储深度相关的数据
depth_frame_global = None
depth_intrinsics_global = None

def mouse_callback(event, x, y, flags, param):
    """
    OpenCV 鼠标回调函数
    当鼠标在窗口上被点击时，此函数会被调用
    """
    global depth_frame_global, depth_intrinsics_global

    # 检查是否是鼠标左键单击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        # 确保全局的深度帧和内参有效
        if depth_frame_global is None or depth_intrinsics_global is None:
            print("深度数据尚未准备好，请稍后再试。")
            return

        # 获取点击像素点的深度值
        depth_value = depth_frame_global.get_distance(x, y)

        # 如果深度值为0，表示该点没有有效的深度信息
        if depth_value == 0:
            print(f"像素点 ({x}, {y}) 处无有效深度数据，无法计算三维坐标。")
            return

        # 使用逆投影（deprojection）计算三维坐标
        # rs2_deproject_pixel_to_point 需要像素坐标[x, y]和深度值
        point_3d = rs.rs2_deproject_pixel_to_point(
            depth_intrinsics_global, [x, y], depth_value
        )
        
        # point_3d 是一个包含 [X, Y, Z] 坐标的列表
        print(f"像素点 ({x}, {y}) -> 相机坐标系 (X, Y, Z): "
              f"[{point_3d[0]:.4f}, {point_3d[1]:.4f}, {point_3d[2]:.4f}] 米")


def main():
    global depth_frame_global, depth_intrinsics_global

    # --- 1. 初始化 RealSense Pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()

    # 获取可用的设备列表
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # 打印设备信息
    print(f"正在使用设备: {device.get_info(rs.camera_info.name)}")
    print(f"    固件版本: {device.get_info(rs.camera_info.firmware_version)}")
    print(f"    序列号: {device.get_info(rs.camera_info.serial_number)}")

    # 配置数据流
    # 您可以根据需要修改分辨率
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # --- 2. 启动 Pipeline ---
    profile = pipeline.start(config)

    # 获取深度传感器的内参（Intrinsics）
    # 内参包含了相机的焦距、主点等信息，是进行2D到3D转换所必需的
    depth_profile = profile.get_stream(rs.stream.depth)
    depth_intrinsics_global = depth_profile.as_video_stream_profile().get_intrinsics()

    # 创建一个对齐对象，将深度帧对齐到彩色帧
    align_to = rs.stream.color
    align = rs.align(align_to)

    # --- 3. 创建 OpenCV 窗口并设置鼠标回调 ---
    window_name = 'RealSense - Click to get 3D coordinates'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n程序已启动。在视频窗口中单击鼠标左键以获取该点的三维坐标。")
    print("按 'q' 键退出程序。")

    try:
        while True:
            # 等待一组成对的帧：深度和彩色
            frames = pipeline.wait_for_frames()

            # 将深度帧对齐到彩色帧的视口
            aligned_frames = align.process(frames)

            # 获取对齐后的帧
            depth_frame_global = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame_global or not color_frame:
                continue

            # --- 4. 将图像数据转换为 NumPy 数组 ---
            color_image = np.asanyarray(color_frame.get_data())

            # --- 5. 显示图像 ---
            cv2.imshow(window_name, color_image)

            # 等待按键，如果按下 'q' 或 ESC，则退出循环
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
    finally:
        # --- 6. 停止并清理 ---
        print("\n正在关闭程序...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已关闭。")

if __name__ == "__main__":
    main()