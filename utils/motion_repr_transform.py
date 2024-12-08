import torch
from pytorch3d.transforms import quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d
from easyvolcap.engine import cfg
from easyvolcap.utils.data_utils import dotdict
import egomocap.utils.matrix as matrix
from egomocap.utils.skeleton import SG
from egomocap.utils.geo_transform import compute_root_quaternion_ay
from egomocap.utils.engine_utils import parse_args_list


### rarely changed
skeleton = cfg.get('skeleton', 'smplx')
data_fps = cfg.get('data_fps', 120)
train_fps = 30
downsample = data_fps // train_fps
dpose = 12

njoints = SG[skeleton]["njoints"]
parents = SG[skeleton]["parents"]
keyjoints = SG[skeleton]["keyjoints"]
D = njoints * dpose

nkeyjoints = SG["keyjoints"]["njoints"]
keyparents = SG["keyjoints"]["parents"]
KD = nkeyjoints * dpose
ALIGNMENT = SG[skeleton]["alignment"]
JOINTS_NAMES = SG[skeleton]["names"]
VR3JOINTS_NAMES = SG[skeleton]["vr3joints"]
VR3JOINTS_INDEX = [JOINTS_NAMES.index(j) for j in VR3JOINTS_NAMES]
### rarely changed

if cfg.get('motoken_cfg_file', None) is not None:
    motoken_cfg = parse_args_list(['-c', cfg.motoken_cfg_file])
    U = motoken_cfg.runner_cfg.get("unit_length", 4)
    cfg.norm_cfg.motoken_input_norm_file = motoken_cfg.norm_cfg.motoken_input_norm_file
else:
    motoken_cfg = None
    U = cfg.runner_cfg.get("unit_length", 4) # need to check when using other models


class MoReprTrans:

    @staticmethod
    def split_ctrl(ctrl):
        '''
            Split ctrl series to pose_pos, pose_rot, pose_vel, root_off, root_dir
        '''
        B, T, C = ctrl.shape
        pose = ctrl[..., :KD].reshape(B, T, nkeyjoints, dpose)

        return dotdict(
            pose_pos = pose[..., 0:3],
            pose_rot = rotation_6d_to_matrix(pose[..., 3:9]),
            pose_vel = pose[..., 9:12],
            root_off = ctrl[..., KD:KD+2],
            root_dir = ctrl[..., KD+2:KD+4],
        )


    @staticmethod
    def split_pose(ctrl):
        '''
            Split ctrl series to pose_pos, pose_rot, pose_vel, root_off, root_dir, root_ctrl
        '''
        B, T, C = ctrl.shape
        pose = ctrl[..., :D].reshape(B, T, njoints, dpose)

        return dotdict(
            pose_pos = pose[..., 0:3],
            pose_rot = rotation_6d_to_matrix(pose[..., 3:9]),
            pose_vel = pose[..., 9:12],
            root_off = ctrl[..., D:D+2],
            root_dir = ctrl[..., D+2:D+4],
            root_ctrl = ctrl[..., D:D+4],
        )
    

    @staticmethod
    def split_oppo_ctrl(oppo):
        '''
            Split oppo series to pose_pos, pose_rot, pose_vel
        '''
        B, T, C = oppo.shape
        pose = oppo.reshape(B, T, nkeyjoints, dpose)

        return dotdict(
            pose_pos = pose[..., 0:3],
            pose_rot = pose[..., 3:9],
            pose_vel = pose[..., 9:12],
        )


    @staticmethod
    def get_actor_motion(motion, frames, squeeze):
        actor_motion = dotdict()
        if squeeze:
            for key, item in motion.items():
                if isinstance(item, torch.Tensor):
                    actor_motion[key] = item[0, frames]
            return actor_motion
        
        for key, item in motion.items():
            if isinstance(item, torch.Tensor):
                actor_motion[key] = item[:, frames]
        return actor_motion
    
    @staticmethod
    def get_apd_motion(motion, K):
        for key, item in motion.items():
            if isinstance(item, torch.Tensor):
                motion[key] = torch.repeat_interleave(motion[key], K, dim=0)
        return motion

    @staticmethod
    def split_lowlevel_output(lowlevel_output: torch.Tensor):
        '''
            Split lowlevel_output to pose_pos, pose_rot, pose_vel
        '''
        B, T, C = lowlevel_output.shape
        pose = lowlevel_output[..., :D].reshape(B, T, njoints, dpose)
        
        return dotdict(
            pose_pos = pose[..., 0:3],
            pose_rot = rotation_6d_to_matrix(pose[..., 3:9]),
            pose_vel = pose[..., 9:12],
        )


    @staticmethod
    def get_root_transmat(pose_pos: torch.Tensor, root_pos: torch.Tensor): #返回local坐标系坐标
        '''
            Inputs:
                pose_pos: (B, J, 3) in world
                root_pos: (B, 1, 3) in world
            Outputs:
                root_transmat: rootmat
                root_rotmat: rootrot, root axis
        '''
        root_y_quat = compute_root_quaternion_ay(pose_pos, skeleton_type=skeleton)  # 计算root节点四元数
        root_rotmat = quaternion_to_matrix(root_y_quat) # 转换为rotation matrix
        root_transmat = matrix.get_TRS(root_rotmat, root_pos)   # 返回root的[R t]
        return root_transmat, root_transmat[..., :3, :3]
    
    @staticmethod
    def split_pose(ctrl):
        '''
            Split ctrl series to pose_pos, pose_rot, pose_vel, root_off, root_dir, root_ctrl
        '''
        B, T, C = ctrl.shape
        pose = ctrl[..., :D].reshape(B, T, njoints, dpose)

        return dotdict(
            pose_pos = pose[..., 0:3],
            pose_rot = rotation_6d_to_matrix(pose[..., 3:9]),
            pose_vel = pose[..., 9:12],
            root_off = ctrl[..., D:D+2],
            root_dir = ctrl[..., D+2:D+4],
            root_ctrl = ctrl[..., D:D+4],
        )