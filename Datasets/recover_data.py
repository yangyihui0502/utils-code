# recover_camera
camera_params[:, :, 6:] = get_camera_unnormalized(camera_params[:, :, 6:])
camera_matrix = torch.eye(4).tile(camera_params.shape[0],camera_params.shape[1],1,1)
camera_rot = transforms.rotation_6d_to_matrix(camera_params[:, :, :6])
camera_pos = camera_params[:, :, 6:]
camera_matrix[:, :, :3, :3] = camera_rot
camera_matrix[:, :, :3, 3] = camera_pos

def get_camera_unnormalized(camera_pos):
    norm_data = np.load('EgoMoGen_data/normalize/camera_trans_norm.npy')
    camera_pos = Renormalize(camera_pos.cpu(),norm_data)
    return camera_pos

# recover image
image = image*255
image = image.to(dtype=torch.uint8)

# recover ai4animation local motion
from utils.motion_repr_transform import MoReprTrans
from utils.geo_transform import *
loc_params = get_motion_unnormalized(pose_series.double())
loc_params = MoReprTrans.split_pose(loc_params)
init_root_mat = meta.init_root_mat.double().to('cpu')   # 序列的第一帧的根节点，用于后面的递推

glb = get_glb_root_by_loc(loc_params, init_root_mat)
glb = get_glb_pose_by_loc(loc_params, glb=glb)
glb.pose_pos = apply_T_on_points(glb.pose_pos, T_y2z)

def get_glb_root_by_loc(loc, init_root_mat, glb=None):
    # 将前一帧的root坐标系转化为global坐标系
    glb = dotdict() if glb is None else glb
    B, T, J, C = loc.pose_pos.shape
    
    glb.root_pos = torch.zeros((B, T, 3), dtype = torch.double)
    glb.root_rot = torch.zeros((B, T, 3, 3), dtype = torch.double)

    for t in range(T):
        if t == 0:
            glb.root_pos[:, [0]] = init_root_mat[:, :3, 3].unsqueeze(1) + \
                matrix.get_position_from_rotmat(matrix.xz2xyz(loc.root_off[:, t])[:, None],
                                                init_root_mat[:, :3, :3])
            glb.root_rot[:, 0] = matrix.get_mat_BfromA(init_root_mat[:, :3, :3],
                                                        matrix.xzdir2rotmat(loc.root_dir[:, t]).double())
        else:
            glb.root_pos[:, t] = (glb.root_pos[:, t-1].unsqueeze(1) + \
                matrix.get_position_from_rotmat(matrix.xz2xyz(loc.root_off[:, t])[:, None],
                                                glb.root_rot[:, t-1])).squeeze(1)
            glb.root_rot[:, t] = matrix.get_mat_BfromA(glb.root_rot[:, t-1],
                                                        matrix.xzdir2rotmat(loc.root_dir[:, t]).double())
    glb.root_mat = matrix.get_TRS(glb.root_rot, glb.root_pos).to(dtype=torch.float64)
    # TODO: move it out
    glb.root_ctrl = loc.root_ctrl
    return glb


def get_glb_pose_by_loc(loc, glb=None):
    glb = dotdict() if glb is None else glb
    B, T, J, C = loc.pose_pos.shape
    glb.pose_pos = torch.zeros((B, T, J, 3), dtype = torch.double)
    glb.pose_rot = torch.zeros((B, T, J, 3, 3), dtype = torch.double)
    glb.pose_vel = torch.zeros((B, T, J, 3), dtype = torch.double)
    interpolate = False

    for t in range(T):
        glb.pose_pos[:, t] = matrix.get_position_from(loc.pose_pos[:, t], glb.root_mat[:, t])
        glb.pose_rot[:, t] = matrix.get_mat_BfromA(glb.root_rot[:, t][:, None], loc.pose_rot[:, t])
        glb.pose_vel[:, t] = matrix.get_direction_from(loc.pose_vel[:, t], glb.root_mat[:, t])
        if interpolate:
        # 插值，使结果更平滑
            if t == 0:
                glb.pose_pos[:, t] = Lerp(glb.pose_pos[:, t], 
                                            self.pose_pos[:, -1] + glb.pose_vel[:, t] / train_fps, 0.5)
            else:
                glb.pose_pos[:, t] = Lerp(glb.pose_pos[:, t], 
                                            glb.pose_pos[:, t-1] + glb.pose_vel[:, t] / train_fps, 0.5)
    return glb