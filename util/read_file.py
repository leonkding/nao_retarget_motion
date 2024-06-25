import numpy as np

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

def read_npz(file_path):
    try:
        f = np.load(file_path)
        body_pose = f['body_pose'] 
        hand_pose = f['hand_pose']
        face_pose = f['face_pose']
        return body_pose, hand_pose, face_pose

    except():
        print("Can not open file!")