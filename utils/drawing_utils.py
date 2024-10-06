import dataclasses
import math
from typing import List, Mapping, Optional, Tuple, Union
from itertools import filterfalse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates, DrawingSpec
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import location_data_pb2

from utils.body_parts import FullBody

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

mitDrawing_Spec: DrawingSpec = DrawingSpec
mitDrawing_Spec.circle_radius=2
mitDrawing_Spec.color=WHITE_COLOR
mitDrawing_Spec.thickness=1

def get_edge(line: List, intersection: List):
    """
    Find the edge point of the line

    Args:
      line: list of (x, y) coordinates
      intersection: list of (x, y) coordinate

    Returns:
      edge: list of (x, y) coordinate
    """

    if line[:2]==intersection:
      return line[2:]
    else:
      return line[:2]

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    Args:
      a, b, c: tuples of (x, y) coordinates
    Returns:
      Angle in degrees
    """
    # figure out the direction of coordinates
    if a[0] > c[0]:
       a, c = c, a
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang



def line_intersection(line1: List, line2: List):
    """
    Find intersection between two lines
    
    Args:
      line1: list of (x, y) coordinates
      line2: list of (x, y) coordinates

    Raises:
      Exception: if lines dont intersect with one another

    Returns:
      Intersection: the point of intersection between the lines
      point1: edge of line 1 which is not intersecting with line 2
      point2: edge of line 2 which is not intersecting with line 1
    """
    xdiff = (line1[0] - line1[2], line2[0] - line2[2])
    ydiff = (line1[1] - line1[3], line2[1] - line2[3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('Lines do not intersect')

    d = (det((line1[0], line1[1]), (line1[2], line1[3])), det((line2[0], line2[1]), (line2[2], line2[3])))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    intersection = [int(x), int(y)]
    point1 = get_edge(line1, intersection)
    point2 = get_edge(line2, intersection)
    return intersection, point1, point2



def draw_arc(image: np.ndarray,
    idx_2_coordinates: dict,
    joints: List[Tuple[int, int, int, int]] = None,
    ):
  """
  Draw an arc representing the angle between three points.
  Args:
    image: numpy array, the image to draw on
    idx_2_coordinates: dict of pose landmarks
    joints: list of (x, y) coordinates of joints
  """

  try:
    line1 = [list(idx_2_coordinates[i]) for i in joints[:2]]
    line2 = [list(idx_2_coordinates[i]) for i in joints[2:]]
    
    # print(line1, line2)
    line1 = line1[0]+line1[1]
    line2 = line2[0]+line2[1]
    
    intersection, point1, point2 = line_intersection(line1, line2)

    angle1 = math.atan2(point1[1] - intersection[1],  point1[0]- intersection[0])
    angle2 = math.atan2(point2[1] - intersection[1],  point2[0] - intersection[0])

    # Convert angles to degrees
    angle1 = np.degrees(angle1)
    angle2 = np.degrees(angle2)
    # print(angle1, angle2)
    
    # Calculate the internal angle between the two lines 
    arc_angle = calculate_angle(point1, intersection, point2)
    print("initial arc angle", arc_angle)
    
    # # Define radius for the arc
    radius = int(math.sqrt(abs(line1[3]-line1[1])**2 + abs(line1[2]-line1[0])**2)//3)
    radius = 20
    
    start_angle, end_angle = angle1, angle2
    if abs(end_angle - start_angle) > 180:
        if start_angle < end_angle:
            start_angle += 360
        else:
            end_angle += 360

    cv2.ellipse(image, intersection, (radius, radius), 0,start_angle, end_angle, (0, 255, 0), -1)
    cv2.putText(image, f"{round(arc_angle)}", (intersection[0], intersection[1]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2)
  except:
    pass



  



def draw_landmarks(image: np.ndarray, landmark_list: landmark_pb2.NormalizedLandmarkList, connections: Optional[List[Tuple[int, int]]] = None,
    arc_connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Optional[Union[DrawingSpec, Mapping[int, DrawingSpec]]] = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = mitDrawing_Spec(),
    is_drawing_landmarks: bool = True,
):
  """Draws the landmarks and the connections on the image.

  Args:
    image: A three channel BGR image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: Either a DrawingSpec object or a mapping from hand
      landmarks to the DrawingSpecs that specifies the landmarks' drawing
      settings such as color, line thickness, and circle radius. If this
      argument is explicitly set to None, no landmarks will be drawn.
    connection_drawing_spec: Either a DrawingSpec object or a mapping from hand
      connections to the DrawingSpecs that specifies the connections' drawing
      settings such as color and line thickness. If this argument is explicitly
      set to None, no landmark connections will be drawn.
    is_drawing_landmarks: Whether to draw landmarks. If set false, skip drawing
      landmarks, only contours will be drawed.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel BGR.
      b) If any connetions contain invalid landmark index.
  """
  if not landmark_list:
    return
  if image.shape[2] != _BGR_CHANNELS:
    raise ValueError('Input image must contain three channel bgr data.')
  
  image_rows, image_cols, _ = image.shape
  
  idx_to_coordinates = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
  if connections:
    num_landmarks = len(landmark_list.landmark)
    # Draws the connections if the start and end landmarks are both visible.
    connection_points = set()
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      connection_points.add(start_idx)
      connection_points.add(end_idx)
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        drawing_spec = connection_drawing_spec[connection] if isinstance(
            connection_drawing_spec, Mapping) else connection_drawing_spec
        cv2.line(image, idx_to_coordinates[start_idx],
                 idx_to_coordinates[end_idx], BLUE_COLOR,
                 drawing_spec.thickness)
  if arc_connections:
    # Visualizing the angle between joints and drawing an arc between them
    draw_arc(image, idx_to_coordinates, arc_connections[1])
    draw_arc(image, idx_to_coordinates, arc_connections[0])
  
  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  if is_drawing_landmarks and landmark_drawing_spec:
    for idx, landmark_px in idx_to_coordinates.items():
      if idx in connection_points:
        drawing_spec = landmark_drawing_spec[idx] if isinstance(
            landmark_drawing_spec, Mapping) else landmark_drawing_spec
        # White circle border
        circle_border_radius = max(drawing_spec.circle_radius + 1,
                                    int(drawing_spec.circle_radius * 1.2))
        cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                    drawing_spec.thickness)
        # Fill color into the circle
        cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                    #  drawing_spec.color, 
                    WHITE_COLOR, drawing_spec.thickness)
        
    # return idx_to_coordinates