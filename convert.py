SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    # "spine2",
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
    # "left_hand",
    # "right_hand",
]

SMPL_TO_MOCAP = {
    0: 0,  # pelvis 0
    1: 43,  # left_hip 1
    2: 47,  # right_hip 2
    3: 1,  # spine1 3
    4: 44,  # left_knee 4
    5: 48,  # right_knee 5
    # 6: not present spine2
    7: 45,  # left_ankle 6
    8: 49,  # right_ankle 7
    9: 2,  # spine3 8
    10: 46,  # left_foot 9
    11: 50,  # right_foot 10
    12: 3,  # neck 11
    13: 5,  # left_collar 12
    14: 24,  # right_collar 13
    15: 4,  # head 14
    16: 6,  # left_shoulder 15
    17: 25,  # right_shoulder 16
    18: 7,  # left_elbow 17
    19: 26,  # right_elbow 18
    20: 8,  # left_wrist 19
    21: 27,  # right_wrist 20
    # IGNORE hand
    # 22: 15, # left_hand
    # 23: 24 # right_hand
}
SMPL_PARENTS = {
    0: -1,  # pelvis (root, no parent)
    1: 0,  # left_hip -> pelvis
    2: 0,  # right_hip -> pelvis
    3: 0,  # spine1 -> pelvis
    4: 1,  # left_knee -> left_hip
    5: 2,  # right_knee -> right_hip
    7: 4,  # left_ankle -> left_knee
    8: 5,  # right_ankle -> right_knee
    9: 3,  # spine3 -> spine1
    # 10: 7,  # left_foot -> left_ankle
    # 11: 8,  # right_foot -> right_ankle
    12: 9,  # neck -> spine3
    13: 12,  # left_collar -> neck
    14: 12,  # right_collar -> neck
    15: 12,  # head -> neck
    16: 13,  # left_shoulder -> left_collar
    17: 14,  # right_shoulder -> right_collar
    18: 16,  # left_elbow -> left_shoulder
    19: 17,  # right_elbow -> right_shoulder
    20: 18,  # left_wrist -> left_elbow
    21: 19,  # right_wrist -> right_elbow
}
