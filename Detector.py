import sys

import cv2
import mediapipe as mp
import socket
import time
import numpy as np
import json
from types import SimpleNamespace
import os

class AllBodyVector:
    def __init__(self):
        self.wristL = None
        self.elbowL = None
        self.upperArmL = None
        self.upperBody = None

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
        emulatedPointVector = calcEmulatedPoint(landmarks, "UpperBody", required_points)
        self.upperBody = emulatedPointVector


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

    sys.exit("Invalid mode at calcEmulatedPoint")


def default_method(item):
    if isinstance(item, object) and hasattr(item, '__dict__'):
        return item.__dict__
    else:
        raise TypeError


def calculate_angle(a, b, c):
    # calc angle with 3S coordinate
    v1 = a - b
    v2 = c - b
    angel = np.arccos((v1 @ v2) / (np.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]) * np.sqrt(
        v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])))
    return angel * 180 / np.pi


def main():
    id_left_shoulder = 11
    id_left_elbow = 13
    id_left_wrist = 15

    serv_address = ('127.0.0.1', 8890)

    # ソケットを作成する
    sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
    print('create socket')

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            try:
                lp = results.pose_world_landmarks.landmark  # Landmark Pack
            except AttributeError:
                print("Detection Failed")
                message = "Detection Failed"
                allBodyVectorPacks = AllBodyVector()
                landmark_packed_json = json.dumps(allBodyVectorPacks, default=default_method, indent=2)
                send_len = sock.sendto(landmark_packed_json.encode('utf-8'), serv_address)
                continue  # Skip this non-detected loop

            # Calculate angle of elbow Left
            a = np.array([lp[id_left_shoulder].x, lp[id_left_shoulder].y, lp[id_left_shoulder].z])
            b = np.array([lp[id_left_elbow].x, lp[id_left_elbow].y,
                          lp[id_left_elbow].z])
            c = np.array([lp[id_left_wrist].x, lp[id_left_wrist].y,
                          lp[id_left_wrist].z])
            # print(calculate_angle(a, b, c))
            # print(f"\r{lp[id_left_wrist].x * 10:.3f}, {lp[id_left_wrist].y * 10:.3f}, {lp[id_left_wrist].z * 10:.3f}",end="")

            allBodyVectorPacks = AllBodyVector()
            allBodyVectorPacks.assign_ElbowL(lp)
            allBodyVectorPacks.assign_upperArmL(lp)
            allBodyVectorPacks.assign_UpperBody(lp)

            landmark_packed_json = json.dumps(allBodyVectorPacks, default=default_method, indent=2)
            print(landmark_packed_json)
            send_len = sock.sendto(landmark_packed_json.encode('utf-8'), serv_address)

            # print('###########################################')
            # cv2.waitKey(0)
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == "__main__":
    main()
