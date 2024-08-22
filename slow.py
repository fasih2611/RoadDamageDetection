import cv2

def slow_down_video(video_path, output_path, slow_down_factor):
    """
    Slow down a video by duplicating frames.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to the output video file.
        slow_down_factor (int): Factor by which to slow down the video.
    """
    # Read the input video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Slow down the video by duplicating frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Write the original frame
        out.write(frame)

        # Duplicate frames to slow down the video
        for _ in range(slow_down_factor - 1):
            out.write(frame)

    # Release resources
    cap.release()
    out.release()

# Example usage
video_path = './sample_video.mp4'
output_path = 'slowed.mp4'
slow_down_factor = 3  # Slow down the video by a factor of 2

slow_down_video(video_path, output_path, slow_down_factor)