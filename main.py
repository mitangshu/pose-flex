import cv2
from mediapipe.python.solutions import drawing_utils
import mediapipe.python as mp
from mediapipe.python.solutions import pose 
from utils.drawing_utils import draw_landmarks
from utils.body_parts import LeftSideBodyParts as L, RightSideBodyParts as R, FullBody as F

my_pose = pose.Pose(model_complexity=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Initialize MediaPipe Drawing.
mp_drawing = drawing_utils




frozenset([                             (5, 6),(4, 5),          (1, 2), (2, 3),
                                    (6, 8),                                     (3, 7),
                                                    (0, 4),(0, 1),
           


                                                        (9, 10),
                                



                    (16, 22),                                                                   (15, 21),
           (18, 20),(16, 20),    (14, 16),  (12, 14),   (11, 12),   (11, 13),    (13, 15),      (15, 19), (17, 19),
                (16, 18),                                                                       (15, 17),
                
                



                
                                            (12, 24),   (23, 24),   (11, 23),
                                            
                                            
                                            (24, 26),               (23, 25),


                                            (26, 28),               (25, 27),


                                            (28, 30),               (27, 31),

                                    (28, 32), (30, 32),            (27, 29), (29, 31),
                                            ])    
UpperBodyConnections = [R.right_bicep, R.right_forearm, R.right_torso, L.left_bicep, L.left_forearm, L.left_torso]
ArcConnections = [list(R.right_forearm)+list(R.right_bicep), list(L.left_forearm)+list(L.left_bicep)]
print(ArcConnections)

# Open the video file or webcam.
cap = cv2.VideoCapture('data\\back\SeatedCableRow.mp4')  # Replace 'input_video.mp4' with 0 to use the webcam.

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
    # Press 'q' to exit the loop.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object.
cap.release()

# img = cv2.imread("girl.jpg")


cv2.destroyAllWindows()
my_pose.close()