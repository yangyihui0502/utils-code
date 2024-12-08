MOTIVE_NUM_JOINTS = 21

MOTIVE_JOINTS_NAMES = [
    'Hips', # 0
    'Spine', # 1
    'Spine1', # 2
    'Neck', # 3
    'Head', # 4
    'LeftShoulder', # 5
    'LeftArm', # 6
    'LeftForeArm', # 7 
    'LeftHand', # 8
    'RightShoulder', # 9
    'RightArm', # 10
    'RightForeArm', # 11
    'RightHand', # 12
    'LeftUpLeg', # 13
    'LeftLeg', # 14
    'LeftFoot', # 15
    'LeftToeBase', # 16
    'RightUpLeg', # 17
    'RightLeg', # 18
    'RightFoot', # 19
    'RightToeBase', # 20
]

MOTIVE_VR3JOINTS = [
    "Head",
    "LeftHand",
    "RightHand",
]

MOTIVE_JOINTS_PARENTS = [
    -1, 0, 1, 2, 3,
              2, 5, 6, 7,
              2, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19,
]

MOTIVE_ALIGNMENT = [
    0, 0, 1, 0, 1,
             0, 1, 1, 1,
             0, 1, 1, 1,
       0, 1, 1, 1,
       0, 1, 1, 1,
]

MOTIVE_KEYJOINTS = [
    0, # Hips, 0
    1, # Spine, 1
    4, # Head, 2
    7, # LeftForeArm, 3
    8, # LeftHand, 4
    11, # RightForeArm, 5
    12, # RightHand, 6
    14, # LeftLeg, 7
    15, # LeftFoot, 8
    18, # RightLeg, 9
    19, # RightFoot, 10
]

KEYJOINTS11_PARENTS = [
    -1, 0, 1,
           1, 3,
           1, 5,
       0, 7, 
       0, 9,
]

SMPL_NUM_JOINTS = 24

SMPL_JOINTS_NAMES = [
    "pelvis",         # 0
    "left_hip",       # 1
    "right_hip",      # 2
    "spine1",         # 3
    "left_knee",      # 4
    "right_knee",     # 5
    "spine2",         # 6
    "left_ankle",     # 7
    "right_ankle",    # 8
    "spine3",         # 9
    "left_foot",      # 10
    "right_foot",     # 11
    "neck",           # 12
    "left_collar",    # 13
    "right_collar",   # 14
    "head",           # 15
    "left_shoulder",  # 16
    "right_shoulder", # 17
    "left_elbow",     # 18
    "right_elbow",    # 19
    "left_wrist",     # 20
    "right_wrist",    # 21
    "left_hand", 
    "right_hand"
]

SMPL_JOINTS_PARENTS = [
    -1, 0, 0, 0, 
    1, 2, 3, 4, 5, 6, 7, 8, # 0-11
    9, 9, 9, 
    12, 13, 14, 16, 17, 18, 19, 20, 21
]

SMPL_ALIGNMENT = [
    0, 0, 0, 0, 
    1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 
    1, 1, 1, 1, 1, 1, 1, 1, 1
]

SMPL_KEYJOINTS = [
    0, # pelvis
    3, # spine1
    4, # left_knee
    5, # right_knee
    7, # left_ankle
    8, # right_ankle
    15, # head
    18, # left_elbow
    19, # right_elbow
    20, # left_wrist
    21, # right_wrist
]

# for fast indexing, skeleton group
SG = {
    'motive': {
        'parents': MOTIVE_JOINTS_PARENTS,
        'njoints': MOTIVE_NUM_JOINTS,
        'keyjoints': MOTIVE_KEYJOINTS,
        'vr3joints': MOTIVE_VR3JOINTS,
        'alignment': MOTIVE_ALIGNMENT,
        'names': MOTIVE_JOINTS_NAMES,
        'toe': [
            MOTIVE_JOINTS_NAMES.index("LeftToeBase"),
            MOTIVE_JOINTS_NAMES.index("RightToeBase"),
        ],
        'foot': [
            MOTIVE_JOINTS_NAMES.index("LeftFoot"),
            MOTIVE_JOINTS_NAMES.index("LeftToeBase"),
            MOTIVE_JOINTS_NAMES.index("RightFoot"),
            MOTIVE_JOINTS_NAMES.index("RightToeBase"),
        ],
        'fcontact_y_threshold': 0.05,
        'fcontact_v_threshold': 1,
    },
    'smplx': {
        'parents': SMPL_JOINTS_PARENTS,
        'njoints': SMPL_NUM_JOINTS,
        'keyjoints': SMPL_KEYJOINTS,
        'alignment': SMPL_ALIGNMENT,
        'names': SMPL_JOINTS_NAMES,
        'toe': [
            SMPL_JOINTS_NAMES.index("left_foot"),
            SMPL_JOINTS_NAMES.index("right_foot")
        ],
        'foot': [
            SMPL_JOINTS_NAMES.index("left_ankle"),
            SMPL_JOINTS_NAMES.index("left_foot"), 
            SMPL_JOINTS_NAMES.index("right_ankle"),
            SMPL_JOINTS_NAMES.index("right_foot"),
        ],
        'vr3joints': [
            "head",
            "left_wrist",
            "right_wrist",  
        ],
        'fcontact_y_threshold': 0.05,
        'fcontact_v_threshold': 1,
    },
    'keyjoints': {
        'parents': KEYJOINTS11_PARENTS,
        'njoints': 11,
    }
}