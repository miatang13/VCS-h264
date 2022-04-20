import numpy as np
import cv2

cap = cv2.VideoCapture('../videos/corgi_short.mp4')

# Define the codec and create VideoWriter object
# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
fps = 20.0
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.mp4',fourcc, fps, (854,480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    print("Flipping frame")
    frame = cv2.flip(frame, 0)
    # write the flipped frame
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release everything if job is finished
print("Releasing everything. Job finished. ")
cap.release()
out.release()

cv2.destroyAllWindows()