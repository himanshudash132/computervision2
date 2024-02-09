import cv2
from ultralytics import YOLO
import time

#Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')

# Open the video file
video_path = "traffic.mp4"
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
     # Read a frame from the video
     success, frame =cap.read()

     if success:
        # start = time.perf_counter()
        #Run YOLOv8 inference on the frame
        results = model(frame)

        # end = time.perf_counter()
        # total_time = end - start
        # fps = 1 / total_time
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        #Display the annotated frame
        # cv2.putText(annotated_frame, f"FPS: {int(fps)}")
        cv2.imshow('YOLOv8 Inference',annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
     else:
          # Break the loop if the end of the video is reached
          break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()               