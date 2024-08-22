# import os

# def get_filenames_without_extensions(folder_path):
#     """Return a set of filenames without extensions in the given folder"""
#     return set(os.path.splitext(filename)[0] for filename in os.listdir(folder_path))

# def delete_mismatch_files(folder1_path, folder2_path):
#     """Delete files from folder2 that don't have a matching filename in folder1"""
#     folder1_filenames = get_filenames_without_extensions(folder1_path)
#     folder2_filenames = get_filenames_without_extensions(folder2_path)

#     mismatch_files = [filename for filename in os.listdir(folder2_path) if os.path.splitext(filename)[0] not in folder1_filenames]

#     if mismatch_files:
#         print(f"Deleting {len(mismatch_files)} mismatch files from {folder2_path}:")
#         for filename in mismatch_files:
#             file_path = os.path.join(folder2_path, filename)
#             os.remove(file_path)
#             print(f"Deleted {file_path}")
#     else:
#         print(f"No mismatch files found in {folder2_path}")

# if __name__ == "__main__":
#     folder1_path = "./311366_Cracks and Potholes in Road/images/val"
#     folder2_path = "./311366_Cracks and Potholes in Road/labels/val"

#     delete_mismatch_files(folder1_path, folder2_path)

import cv2
import os

# Set the directory path and the output video file path
dir_path = './RCP/images/train'
output_video_path = './video.mp4'

# Get the list of image files in the directory
image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')]

# Sort the image files in alphabetical order
image_files.sort()

# Set the video codec and frame rate
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15

# Get the dimensions of the first image
img = cv2.imread(os.path.join(dir_path, image_files[0]))
height, width, _ = img.shape

# Create a video writer object
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Iterate over the image files and write them to the video
for file in image_files:
    img = cv2.imread(os.path.join(dir_path, file))
    video_writer.write(img)

# Release the video writer object
video_writer.release()