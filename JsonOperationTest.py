import cv2
import mediapipe as mp
import socket
import time
import numpy as np

import json
from types import SimpleNamespace


class AllBodyVector:
    def __init__(self):
        self.wristL = None
        self.elbowL = None
        self.upperArmL = None

    def assign_upperArmL(self, landmarks):
        id_left_shoulder = 11
        id_left_elbow = 13
        self.upperArmL = calculateVector(landmarks, id_left_shoulder, id_left_elbow)

    def assign_ElbowL(self, landmarks):
        id_left_elbow = 13
        id_left_wrist = 15
        self.elbowL = calculateVector(landmarks, id_left_elbow, id_left_wrist)

    def assign_WristL(self, landmarks):
        id_left_wrist = 15
        emulatedPoint = calcEmulatedPoint(landmarks, "HandTip")
        self.wristL = calculateVector_Manual(landmarks[15], emulatedPoint)

    def assign_UpperBody(self, landmarks):
        required_points = {"id_left_shoulder": 11, "id_right_shoulder": 12, "id_right_hip": 24, "id_left_hip": 23}
        emulatedPoint = calcEmulatedPoint(landmarks, "UpperBody", required_points)
        self.wristL = calculateVector_Manual(landmarks[15], emulatedPoint)


class ElementVector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Vector:
    pass


def calculateVector(landmarks, origin, index):
    vector_packer = Vector()
    vector_packer.x = landmarks[index].x - landmarks[origin].x
    vector_packer.y = landmarks[index].y - landmarks[origin].y
    vector_packer.z = landmarks[index].z - landmarks[origin].z
    return vector_packer


def calculateVector_Manual(origin, index):
    #  In case of emulated point, you can use it for manual point describe
    vector_packer = Vector()
    vector_packer.x = index.x - origin.x
    vector_packer.y = index.y - origin.y
    vector_packer.z = index.z - origin.z
    return vector_packer


def calculateMiddlePoint(point1, point2):
    vector_packer = Vector()
    vector_packer.x = (point1.x + point2.x)/2
    vector_packer.y = (point1.y + point2.y)/2
    vector_packer.z = (point1.z + point2.z)/2
    return vector_packer


def calcEmulatedPoint(landmarks, mode, requiredpoints):
    if mode == "HandTip":
        pass

    if mode == "UpperBody":
        # Calculate Upper Body Vector. In Unity It is Spine or Hip.
        left_shoulder = landmarks[requiredpoints["id_left_shoulder"]]
        right_shoulder = landmarks[requiredpoints["id_right_shoulder"]]
        right_hip = landmarks[requiredpoints["id_right_hip"]]
        left_hip = landmarks[requiredpoints["id_left_hip"]]
        # Calc middle point of given two points
        centre_hip = calculateMiddlePoint(left_shoulder,right_shoulder)
        centre_shoulder = calculateMiddlePoint(left_hip,right_hip)
        return calculateVector_Manual(centre_hip,centre_shoulder)


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



    testc1 = Test_Coordinate()
    testc1.x = 1
    testc1.y = 1
    testc1.z = 1

    testc2 = Test_Coordinate()
    testc2.x = 3
    testc2.y = 5
    testc2.z = 7

    landmarks = [testc1,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2,testc2]

    print(landmarks)
    print(landmarks[1].x)

    # Calculate Vector given OriginPoint And IndexPoint
    vector_package = calculateVector(landmarks,0,1)
    print(vector_package.x, vector_package.y, vector_package.z)

    fullBodyPoint3D = AllBodyVector()
    # fullBodyPoint3D.assign_WristL(4, 4, 4)
    fullBodyPoint3D.assign_ElbowL(landmarks)
    fullBodyPoint3D.assign_upperArmL(landmarks)

    json_str = json.dumps(fullBodyPoint3D, default=default_method, indent=2)
    print(json_str)



    return 0


if __name__ == "__main__":
    main()



