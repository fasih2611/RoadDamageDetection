import cv2
from ultralytics import YOLO, solutions
import time

model = YOLO("./runs/detect/train7/weights/POT-YOLO.pt").to('cpu')

cap = cv2.VideoCapture("./sample_video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(0, 600), (1400, 600)]

video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=1,
)

frame_id = 0
start_time = time.time()

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    frame_id += 1

    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)

    elapsed_time = time.time() - start_time
    fps = frame_id / elapsed_time

    # Create a beautiful FPS counter
    fps_text = f"FPS: {fps:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    color = color = (255, 191, 0) # Green color
    text_size, _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
    text_x = 10
    text_y = 30
    cv2.rectangle(im0, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 10, text_y + 5), (0, 0, 0), -1)
    cv2.putText(im0, fps_text, (text_x + 5, text_y), font, font_scale, color, thickness)

    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()