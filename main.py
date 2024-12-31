import cv2
import os
from mediapipe.python.solutions import drawing_utils
import mediapipe.python as mp
import argparse
from mediapipe.python.solutions import pose 
from utils.drawing_utils import draw_landmarks
from utils.body_parts import LeftSideBodyParts as L, RightSideBodyParts as R, FullBody as F

# Initialize Parser
parser = argparse.ArgumentParser()

# Add Arguments
parser.add_argument("-p", "--Path", help="Path to your video file")
parser.add_argument("-o", "--Output", default="output", help="Path to output dir")

# Read arguments from command line
args = parser.parse_args()

my_pose = pose.Pose(model_complexity=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Initialize MediaPipe Drawing.
mp_drawing = drawing_utils
  
UpperBodyConnections = [R.right_bicep, R.right_forearm, R.right_torso, L.left_bicep, L.left_forearm, L.left_torso]
ArcConnections = [list(R.right_forearm)+list(R.right_bicep), list(L.left_forearm)+list(L.left_bicep)]
print(ArcConnections)

# Open the video file or webcam. 'data\\back\SeatedCableRow.mp4'
cap = cv2.VideoCapture(args.Path)  # Replace 'input_video.mp4' with 0 to use the webcam.

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 

result = cv2.VideoWriter(f'{args.Output}/{os.path.basename(args.Path)}.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, size) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("ret not found")
        break
    
    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose.
    results = my_pose.process(rgb_frame)
    
    # Draw the pose annotation on the frame.
    if results.pose_landmarks:
        draw_landmarks(
            frame, results.pose_landmarks, UpperBodyConnections, ArcConnections)
    
    # Display the frame.
    cv2.imshow('Pose Detection', frame)
    # cv2.waitKey(0)
    result.write(frame)
    # Press 'q' to exit the loop.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object.
cap.release()
result.release() 
# img = cv2.imread("girl.jpg")


cv2.destroyAllWindows()
my_pose.close()