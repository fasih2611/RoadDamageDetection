import cv2

def resize_video(input_file, output_file):
    """
    Resize a video to 640x640 pixels.

    Args:
        input_file (str): Path to the input video file.
        output_file (str): Path to the output video file.
    """
    cap = cv2.VideoCapture(input_file)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        resized_frame = cv2.resize(frame, (640, 480))

        out.write(resized_frame)

    cap.release()
    out.release()

input_file = './runs/segment/predict6/video.avi'
output_file = 'output.mp4'
resize_video(input_file, output_file)