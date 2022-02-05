import cv2
import mediapipe as mp
import socket
import time
import numpy as np

import json
from types import SimpleNamespace


class AllBodyPoint:
    def __init__(self):
        self.wristR = None
        self.elbowR = None
        self.upperArmR = None

    def assign_upperArmR(self, x, y, z):
        self.upperArmR = ElementVector(x, y, z)

    def assign_ElbowR(self, x, y, z):
        self.elbowR = ElementVector(x, y, z)

    def assign_WristR(self, x, y, z):
        self.wristR = ElementVector(x, y, z)


class ElementVector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def default_method(item):
    if isinstance(item, object) and hasattr(item, '__dict__'):
        return item.__dict__
    else:
        raise TypeError


def calculate_angle(a, b, c):
    # calc angle with 3S coordinate
    v1 = a - b
    v2 = c - b
    print(v1 @ v2)
    angel = np.arccos((v1 @ v2) / (np.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]) * np.sqrt(
        v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])))
    return angel * 180 / np.pi


def calculateVector(landmarks,origin,index):
    vector_packer = Test_Vector()
    vector_packer.x = landmarks[index].x - landmarks[origin].x
    vector_packer.y = landmarks[index].y - landmarks[origin].y
    vector_packer.z = landmarks[index].z - landmarks[origin].z
    return vector_packer


class Test_Coordinate:
    pass


class Test_Vector:
    pass

def main():
    print("aa")
    a = np.array([1, 0, 0])
    b = np.array([0, 0, 0])
    c = np.array([0, 1, 0])
    print(c[0])
    print(a @ b)
    print(calculate_angle(a, b, c))

    fullBodyPoint3D = AllBodyPoint()
    fullBodyPoint3D.assign_WristR(4, 4, 4)
    fullBodyPoint3D.assign_ElbowR(3, 3, 3)
    fullBodyPoint3D.assign_upperArmR(2, 2, 2)

    json_str = json.dumps(fullBodyPoint3D, default=default_method, indent=2)
    print(json_str)

    testc1 = Test_Coordinate()
    testc1.x = 1
    testc1.y = 1
    testc1.z = 1

    testc2 = Test_Coordinate()
    testc2.x = 3
    testc2.y = 5
    testc2.z = 7

    landmarks = [testc1,testc2]

    print(landmarks)
    print(landmarks[1].x)

    # Calculate Vector given OriginPoint And IndexPoint
    vector_package = calculateVector(landmarks,0,1)
    print(vector_package.x, vector_package.y, vector_package.z)



    return 0


if __name__ == "__main__":
    main()



