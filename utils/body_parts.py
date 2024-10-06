import dataclasses
from typing import Tuple

@dataclasses.dataclass
class BodyPart:
    landmarks: Tuple[int, int]

@dataclasses.dataclass
class RightSideBodyParts:
    right_bicep: BodyPart= (12, 14)
    right_forearm: BodyPart=(14, 16)
    right_torso: BodyPart=(12, 24)
    right_thigh: BodyPart=(24, 26)
    right_shin: BodyPart= (26, 28)

    

@dataclasses.dataclass
class LeftSideBodyParts:
    left_bicep: BodyPart= (11, 13)
    left_forearm: BodyPart=(13, 15)
    left_torso: BodyPart=(11, 23)
    left_thigh: BodyPart=(23, 25)
    left_shin: BodyPart= (25, 27)

@dataclasses.dataclass
class FullBody:
    left_side: LeftSideBodyParts= LeftSideBodyParts()
    right_side: RightSideBodyParts= RightSideBodyParts()
    chest: BodyPart=(11, 12)



