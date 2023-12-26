import pyrealsense2 as rs
import cv2
import numpy as np
import matplotlib.pyplot as plt
from stitching import Stitcher

stitcher = Stitcher()
# Create pipeline and config objects for the two cameras
pipeline1 = rs.pipeline()
pipeline2 = rs.pipeline()

config1 = rs.config()
config1.enable_device('317422071363')  # Replace with the actual serial number
config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

config2 = rs.config()
config2.enable_device('317622075394')  # Replace with the actual serial number
config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipelines
pipeline1.start(config1)
pipeline2.start(config2)

try:
    while True:
        # Read frames from the first camera
        frames1 = pipeline1.wait_for_frames()
        color_frame1 = frames1.get_color_frame()
        if not color_frame1:
            continue

        # Read frames from the second camera
        frames2 = pipeline2.wait_for_frames()
        color_frame2 = frames2.get_color_frame()
        if not color_frame2:
            continue

        # Convert frames to numpy arrays
        image1 = np.asanyarray(color_frame1.get_data())
        image2 = np.asanyarray(color_frame2.get_data())
        aligned = np.hstack((image1,image2))
        #panorama = stitcher.stitch([image1,image2])
        #cv2.imwrite("Panorama.png",panorama)
        cv2.imshow("heehee",aligned)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release the window gracefully when the window is closed
        if cv2.getWindowProperty('Stitched Image', cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    # Stop streaming for both pipelines
    pipeline1.stop()
    pipeline2.stop()
    cv2.destroyAllWindows()
