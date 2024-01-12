import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
temp = '/home/ubuntu/pidnet-marcus/debug/the_ultimate_test_video.mp4'
# '/home/ubuntu/pidnet-marcus/output/20230323_batches1_8_linlin/20230328_pidnet_l_linlin_4gpu/video_test_dir/the_ultimate_test_video.mp4'
cap = cv2.VideoCapture(temp)
cap.release()
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

# error: [mov,mp4,m4a,3gp,3g2,mj2 @ 0x2e38e00] moov atom not found
# Unable to stop the stream: Inappropriate ioctl for device
# Error opening video stream or file
