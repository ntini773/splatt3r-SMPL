'''
Date: 2024-06-01 14:09:13
LastEditors: DreamTale
LastEditTime: 2024-06-03 12:12:13
FilePath: /DataCrawler/mt_download.py
'''
import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

# global args


import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor



import os
import cv2

def extract_frames_from_videos(input_dir, out_dir, frame_interval=1):
    if not os.path.isdir(input_dir):
        print(f"Invalid input directory: {input_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]

    if not video_files:
        print(f"No .mp4 files found in {input_dir}")
        return

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        v_name = os.path.splitext(video_file)[0]

        dir_path = os.path.join(input_dir, v_name)

        if os.path.exists(dir_path):
            continue
        os.makedirs(dir_path, exist_ok=True)


        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"No frames found in video: {video_file}")
            cap.release()
            continue

        frame_idx = 0
        saved_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                save_name = f"frame_{frame_idx:05d}.jpg" # {v_name}_
                save_path = os.path.join(dir_path, save_name)
                try:
                    cv2.imwrite(save_path, frame)
                    saved_idx += 1
                except Exception as e:
                    print(f"Failed to save frame {frame_idx} of {video_file}: {e}")

            frame_idx += 1

        cap.release()
        print(f"Extracted {saved_idx} frames from {video_file}.")

# 示例调用
extract_frames_from_videos(
    'data_input/driven/target_images/',
    'data_input/driven/target_images/'
)
