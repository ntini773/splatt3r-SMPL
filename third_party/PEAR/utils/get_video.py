import imageio
import os
import glob
import re

def sort_by_number(paths):
    """
    Sort file paths by the last number in the filename.
    """
    def extract_number(path):
        nums = re.findall(r'\d+', path)
        return int(nums[-1]) if nums else -1

    return sorted(paths, key=extract_number)

def images_to_video(img_dir, output_path, fps=25):
    """
    使用 imageio.mimwrite 将文件夹中的图片合成为视频
    """
    img_files = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )
    img_files = sort_by_number(img_files)

    if len(img_files) == 0:
        raise ValueError("❌ 没有找到任何图片！")

    mesh_images = []

    for f in img_files:
        img = imageio.imread(f)

        # 如果图片来自 OpenCV，可能是 BGR，需要转成 RGB
        if img.ndim == 3 and img.shape[2] == 3:
            pass  # 如果你确认是 RGB，可删除这一行
            # img = img[..., ::-1]

        mesh_images.append(img)

    imageio.mimwrite(
        output_path,
        mesh_images,
        fps=fps
    )

    print(f"✅ 视频已保存到 {output_path}, 共 {len(mesh_images)} 帧，fps={fps}")



