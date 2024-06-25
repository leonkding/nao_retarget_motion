"""
@author: ziluo
time: 2024/06/26
"""

import time
import numpy as np
import math
from pybullet_utils import transformations
import pybullet
import pybullet_data as pd
from qibullet import SimulationManager
import pandas as pds

import util.retarget_config_h1 as config
from util.read_txt import read_txt_file


H1_joint_label = config.H1_joint_label

def quaternion_from_euler(r,p,y)

    q = transformations.quaternion_from_euler(r, p, y)
    return q


def quaternion_to_angle_axis(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    # 计算旋转角度 theta
    theta = 2 * math.acos(w)
    # 计算旋转轴的系数
    sin_theta_over_2 = math.sqrt(1 - w*w)
    # 如果sin(theta/2)非常小，那么轴方向可能无法准确计算
    if sin_theta_over_2 > 0.0001:
        axis_x = x / sin_theta_over_2
        axis_y = y / sin_theta_over_2
        axis_z = z / sin_theta_over_2
    else:
        # 当旋转角度很小的时候，旋转轴不重要，可以取任意单位向量
        axis_x = 0
        axis_y = 0
        axis_z = 1
    return theta, (axis_x, axis_y, axis_z)

def get_joint_limits(robot):
    num_joints = pybullet.getNumJoints(robot)
    joint_limit_low = []
    joint_limit_high = []

    for i in range(num_joints):
        joint_info = pybullet.getJointInfo(robot, i)
        joint_name = joint_info[1].decode('utf-8')

        if joint_name in config.MOTOR_NAMES:
          joint_limit_low.append(joint_info[8])
          joint_limit_high.append(joint_info[9])

    return joint_limit_low, joint_limit_high


def set_pose(robot, pose):
    num_joints = pybullet.getNumJoints(robot)
    root_pos = get_root_pos(pose)
    root_rot = get_root_rot(pose)

    pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

    j_pose_idx = len(root_pos) + len(root_rot)
    for j in range(num_joints):
        j_info = pybullet.getJointInfo(robot, j)
        j_state = pybullet.getJointStateMultiDof(robot, j)

        j_name = j_info[1].decode('utf-8')
        #  = j_info[3]
        j_pose_size = len(j_state[0])
        j_vel_size = len(j_state[1])

        if j_name in config.MOTOR_NAMES:
            j_pose = pose[j_pose_idx:(j_pose_idx + j_pose_size)]
            j_vel = np.zeros(j_vel_size)
            pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)
            j_pose_idx += 1

    return


def set_maker_pos(marker_pos, marker_ids):
    num_markers = len(marker_ids)
    assert(num_markers == marker_pos.shape[0])

    for i in range(num_markers):
        curr_id = marker_ids[i]
        curr_pos = marker_pos[i]

        pybullet.resetBasePositionAndOrientation(curr_id, curr_pos, DEFAULT_ROT)

    return


def process_ref_joint_pos_data(joint_pos):
    proc_pos = joint_pos.copy()
    num_pos = joint_pos.shape[0]

    for i in range(num_pos):
        curr_pos = proc_pos[i]
        curr_pos = QuaternionRotatePoint(curr_pos, REF_COORD_ROT)
        curr_pos = QuaternionRotatePoint(curr_pos, REF_ROOT_ROT)
        curr_pos = curr_pos * config.REF_POS_SCALE + REF_POS_OFFSET  # change pose size for different skeleton
        proc_pos[i] = curr_pos

    return proc_pos


def retarget_root_pose(ref_joint_pos, get_motion_name, init_rotation):
    pelvis_pos = ref_joint_pos[REF_HIP_JOINT_ID]
    neck_pos = ref_joint_pos[REF_NECK_JOINT_ID]

    left_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[1]]
    right_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[0]]
    left_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[3]]
    right_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[2]]

    up_dir = neck_pos - pelvis_pos  # vector difference
    up_dir = up_dir / np.linalg.norm(up_dir)  # unitize

    delta_shoulder = left_shoulder_pos - right_shoulder_pos
    delta_hip = left_hip_pos - right_hip_pos
    dir_shoulder = delta_shoulder / np.linalg.norm(delta_shoulder)
    dir_hip = delta_hip / np.linalg.norm(delta_hip)    # unitize

    # left_dir = 0.66 * (dir_shoulder + dir_hip)
    left_dir = dir_hip

    forward_dir = np.cross(up_dir, left_dir)
    forward_dir = forward_dir / np.linalg.norm(forward_dir)

    left_dir = np.cross(forward_dir, up_dir)
    left_dir = left_dir / np.linalg.norm(left_dir)  # unitize

    rot_mat = np.array([[forward_dir[0], left_dir[0], up_dir[0], 0],
                      [forward_dir[1], left_dir[1], up_dir[1], 0],
                      [forward_dir[2], left_dir[2], up_dir[2], 0],
                      [0, 0, 0, 1]])
    if get_motion_name == 'walk':
        root_pos = pelvis_pos
        root_pos[2] -= 0.10
    elif get_motion_name == 'jump':
        root_pos = pelvis_pos
        root_pos[2] -= 0.12   # let robot move upward 0.12

        global lock, limit_root_low
        if lock == 0:
            limit_root_low = root_pos[2]+0.06
            lock = 1
        if root_pos[2] <= limit_root_low:
            root_pos[2] = limit_root_low
    else:
        print("This motion is not implemented.")
        exit(0)
    # root_pos = 0.16 * (left_shoulder_pos + right_shoulder_pos + left_hip_pos + right_hip_pos)
    root_rot = transformations.quaternion_from_matrix(rot_mat)
    root_rot = transformations.quaternion_multiply(root_rot, init_rotation)
    root_rot = root_rot / np.linalg.norm(root_rot)

    return root_pos, root_rot


# compute the pose of nao in every frame
def retarget_pose(human_pose):
    target_pose = np.zeros(len(H1_joint_label))
    target_pose_mask = np.ones(len(H1_joint_label))
    # left and right hip
    left_hip_pose = human_pose[0]
    right_hip_pose = human_pose[1]
    target_pose[0] = left_hip_pose[2]
    target_pose[1] = left_hip_pose[0]
    target_pose[2] = left_hip_pose[1]
    target_pose[5] = right_hip_pose[2]
    target_pose[6] = right_hip_pose[0]
    target_pose[7] = right_hip_pose[1]

    # left and right hip
    left_shoulder_pose = human_pose[15]
    right_shoulder_pose = human_pose[16]
    target_pose[11] = left_shoulder_pose[1]
    target_pose[12] = left_shoulder_pose[0]
    target_pose[13] = left_shoulder_pose[2]
    target_pose[15] = right_shoulder_pose[1]
    target_pose[16] = right_shoulder_pose[0]
    target_pose[17] = right_shoulder_pose[2] 

    # left and right knee
    left_knee_pose = human_pose[3]
    left_knee_pose, _ = quaternion_to_angle_axis(quaternion_from_euler(left_knee_pose))
    right_knee_pose = human_pose[4]
    right_knee_pose, _ = quaternion_to_angle_axis(quaternion_from_euler(right_knee_pose))
    target_pose[3] = left_knee_pose
    target_pose[8] = right_knee_pose

    # left and right ankle
    left_ankle_pose = human_pose[6]
    left_ankle_pose, _ = quaternion_to_angle_axis(quaternion_from_euler(left_ankle_pose))
    right_ankle_pose = human_pose[7]
    right_ankle_pose, _ = quaternion_to_angle_axis(quaternion_from_euler(right_ankle_pose))
    target_pose[4] = left_ankle_pose
    target_pose[9] = right_ankle_pose

    # left and right elbow
    left_elbow_pose = human_pose[17]
    left_elbow_pose, _ = quaternion_to_angle_axis(quaternion_from_euler(left_elbow_pose))
    right_elbow_pose = human_pose[18]
    right_elbow_pose, _ = quaternion_to_angle_axis(quaternion_from_euler(right_elbow_pose))
    target_pose[14] = left_elbow_pose
    target_pose[18] = right_elbow_pose

    # torso
    target_pose_mask[10] = 0
    #elbow_pose = human_pose[10]
    #target_pose[10] = left_elbow_pose[1]

    return target_pose, target_pose_mask


def update_camera(robot):
    base_pos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])
    [yaw, pitch, dist] = pybullet.getDebugVisualizerCamera()[8:11]
    pybullet.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
    return


def load_ref_data(MOTION_NAME, JOINT_POS_FILENAME, FRAME_START, FRAME_END):
    joint_pos_data = pds.read_csv(JOINT_POS_FILENAME)
    joint_pos_data.dropna(axis='columns', how='any')

    start_frame = 0 if (FRAME_START is None) else FRAME_START
    end_frame = joint_pos_data.shape[0] if (FRAME_END is None) else FRAME_END
    joint_pos_data = joint_pos_data[start_frame:end_frame]
    joint_pos_data = joint_pos_data.values[:, 2:REF_jOINT_NUM*3+2]
    if MOTION_NAME == 'walk':
        for frame in range(joint_pos_data.shape[0]):
            joint_pos_data[frame][1 * 3 + 2] -= 0.1   # rightshoulder move outward 0.1  (0,1,2)=(y,x,z)
            joint_pos_data[frame][4 * 3 + 2] += 0.1   # leftshoulder
            joint_pos_data[frame][2 * 3 + 1] -= 0.13  # rightforearm
            joint_pos_data[frame][2 * 3 + 0] -= 0.02  # rightforearm
            joint_pos_data[frame][3 * 3 + 1] -= 0.1   # rightarm
            joint_pos_data[frame][5 * 3 + 1] -= 0.13  # leftforearm
            joint_pos_data[frame][5 * 3 + 0] += 0.02  # leftforearm
            joint_pos_data[frame][6 * 3 + 1] -= 0.1   # leftarm
    elif MOTION_NAME == 'jump':
        for frame in range(joint_pos_data.shape[0]):
            joint_pos_data[frame][1 * 3 + 0] -= 0.1   # rightshoulder
            joint_pos_data[frame][4 * 3 + 0] += 0.1   # leftshoulder
            joint_pos_data[frame][2 * 3 + 1] -= 0.13  # rightforearm
            joint_pos_data[frame][2 * 3 + 0] -= 0.02  # rightforearm
            joint_pos_data[frame][3 * 3 + 1] -= 0.1   # rightarm
            joint_pos_data[frame][5 * 3 + 1] -= 0.13  # leftforearm
            joint_pos_data[frame][5 * 3 + 0] += 0.02  # leftforearm
            joint_pos_data[frame][6 * 3 + 1] -= 0.1   # leftarm

    return joint_pos_data


def retarget_motion(robot, joint_pos_data, motion_name, init_rotat):
    num_frames = joint_pos_data.shape[0]

    for f in range(num_frames):
        ref_joint_pos = joint_pos_data[f]
        ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
        ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)

        curr_pose = retarget_pose(robot, config.DEFAULT_JOINT_POSE, ref_joint_pos, motion_name, init_rotat)
        set_pose(robot, curr_pose)

        if f == 0:
            pose_size = curr_pose.shape[-1]
            new_frames = np.zeros([num_frames, pose_size])

        new_frames[f] = curr_pose

    return new_frames


def output_motion(frames, out_filename, frame_duration):
    with open(out_filename, "a+") as f:
        f.write("{\n")
        f.write("\"LoopMode\": \"Wrap\",\n")
        f.write("\"FrameDuration\": " + str(frame_duration) + ",\n")
        f.write("\"EnableCycleOffsetPosition\": true,\n")
        f.write("\"EnableCycleOffsetRotation\": true,\n")
        f.write("\n")

        f.write("\"Frames\":\n")

        f.write("[")
        for i in range(frames.shape[0]):
            curr_frame = frames[i]

            if i != 0:
                f.write(",")
            f.write("\n  [")

            for j in range(frames.shape[1]):
                curr_val = curr_frame[j]
                if j != 0:
                    f.write(", ")
                f.write("%.5f" % curr_val)

            f.write("]")

        f.write("\n]")
        f.write("\n}")

    return


def main(body_pose, hand_pose, face_poset_data):

    len_motion, num_body_pose = body_pose.shape
    target_poses = []

    for i in range(len_motion):
        current_body_pose = np.reshape(body_pose[i], (-1, 3))
        target_pose, target_pose_mask = retarget_pose(current_body_pose)
        target_poses.append(target_pose)
    
    target_poses = np.stack(target_poses, 0)


if __name__ == "__main__":
    body_pose, hand_pose, face_pose = read_txt_file("path")
    main(body_pose, hand_pose, face_pose)

