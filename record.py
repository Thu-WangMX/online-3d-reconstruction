####保存RealSense相机的彩色图像和深度图像
import pyrealsense2 as rs
import numpy as np
import cv2 # 使用OpenCV进行图像保存和显示
import os
import json

def main():
    # --- 1. 设置输出文件夹 ---
    output_folder = 'realsense_data_test'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, 'color'))
        os.makedirs(os.path.join(output_folder, 'depth'))
    else:
        print(f"文件夹 '{output_folder}' 已存在。请先删除或重命名该文件夹。")
        return

    # --- 2. 初始化 RealSense ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)#   启动流
    align = rs.align(rs.stream.color)# 对齐深度和彩色帧

    # --- 3. 保存相机内参 ---
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intrinsic_dict = {
        'width': intrinsics.width,
        'height': intrinsics.height,
        'fx': intrinsics.fx,
        'fy': intrinsics.fy,
        'ppx': intrinsics.ppx,
        'ppy': intrinsics.ppy
    }
    with open(os.path.join(output_folder, 'intrinsics.json'), 'w') as f:
        json.dump(intrinsic_dict, f, indent=4)
    print("相机内参已保存。")

    # --- 4. 录制循环 ---
    frame_count = 0
    try:
        print("开始录制... 按 'q' 键退出并保存。")
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)# 对齐
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # 获取图像数据
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 保存图像
            cv2.imwrite(f'{output_folder}/color/{frame_count:05d}.png', color_image)
            cv2.imwrite(f'{output_folder}/depth/{frame_count:05d}.png', depth_image)
            
            print(f"已保存第 {frame_count} 帧")
            frame_count += 1

            # 显示实时画面以供参考
            cv2.imshow('RealSense Live View - Press Q to Stop', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n录制结束，共保存 {frame_count} 帧图像到 '{output_folder}' 文件夹。")

if __name__ == "__main__":
    main()