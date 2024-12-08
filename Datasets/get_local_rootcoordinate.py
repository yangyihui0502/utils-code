root_pos[..., 1] = 0 # y = 0,把root的投影点作为原点

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

def compute_root_quaternion_ay(joints, skeleton_type='smpl'):
    """
    Args:
        joints: (B, J, 3), in the start-frame, ay-coordinate
    Returns:
        root_quat: (B, 4) from z-axis to fz
    """
    joints_shape = joints.shape
    joints = joints.reshape((-1,) + joints_shape[-2:])
    t_ayfz2ay = joints[:, 0, :].detach().clone()
    t_ayfz2ay[:, 1] = 0  # do not modify y

    if skeleton_type == 'smplx':
        RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]  # (B, 2), hip point to left side
        RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]  # (B, 2), shoulder point to left side
    elif skeleton_type == 'motive':
        RL_xz_h = joints[:, 13, [0, 2]] - joints[:, 17, [0, 2]]
        RL_xz_s = joints[:, 5, [0, 2]] - joints[:, 9, [0, 2]]
    RL_xz = RL_xz_h + RL_xz_s
    I_mask = RL_xz.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction
    if I_mask.sum() > 0:
        log("{} samples can't decide the face direction".format(I_mask.sum()))
        for i in range(I_mask.shape[0]):
            if I_mask[i]:
                RL_xz[i] = RL_xz[i-1]

    x_dir = torch.zeros_like(t_ayfz2ay)  # (B, 3)
    x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 1] = 1  # (B, 3)
    z_dir = torch.cross(x_dir, y_dir, dim=-1)

    z_dir[..., 2] += 1e-9
    pos_z_vec = torch.tensor([0, 0, 1]).to(joints.device).float()  # (3,)
    root_quat = qbetween(pos_z_vec.repeat(joints.shape[0]).reshape(-1, 3), z_dir)  # (B, 4)
    root_quat = root_quat.reshape(joints_shape[:-2] + (4,))
    return root_quat