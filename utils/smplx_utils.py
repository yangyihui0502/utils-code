from egomogen.utils.body_model import BodyModelSMPLX, BodyModelSMPLH
from pytorch3d.transforms import *
import torch
import numpy as np


def make_smplx(type, **kwargs,):
    if type == 'trumans':
        model = BodyModelSMPLX(
            model_path='EgoMoGen_data/SMPL_Models',
            model_type='smplx',
            gender=kwargs.get('gender', 'male'),
            ext='npz',
            num_betas=10,
            use_pca=False,
        )
    elif type == 'egobody':
        model = BodyModelSMPLX(
            model_path='EgoMoGen_data/SMPL_Models',
            model_type='smplx',
            gender=kwargs.get('gender', 'male'),
            ext='npz',
            num_pca_comps=12,
        )
    elif type == 'boxing':
        model = BodyModelSMPLX(model_path='EgoMoGen_data/SMPL_Models',
                          model_type='smplx',
                          gender="male",
                          ext='npz',
                          use_pca=False,
                          ).double()
    else:
        raise NotImplementedError
    
    return model
def quat_inv(q):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    # res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    res = torch.tensor([1, -1, -1, -1], device=q.device).float() * q
    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    # t = 2.0 * np.cross(q[..., 1:], x)
    t = 2.0 * torch.cross(q[..., 1:], x)
    # res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)
    res = x + q[..., 0][..., None] * t + torch.cross(q[..., 1:], t)

    return res

def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    # res = np.concatenate(
    #     [
    #         y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
    #         y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
    #         y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
    #         y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
    #     ],
    #     axis=-1,
    # )
    res = torch.cat(
        [
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
        ],
        axis=-1,
    )

    return res

def inverse_kinematics_motion(
    global_pos,
    global_rot,
    parents,
):
    """
    Args:
        global_pos : (B, T, J, 3)
        global_rot (q) : (B, T, J, 4)
        parents : SMPLH_PARENTS
    Returns:
        local_pos : (B, T, J-1, 3)
        local_rot (q) : (B, T, J-1, 4)
    """
    J = 22
    local_pos = torch.zeros_like(global_pos)
    local_rot = torch.zeros_like(global_rot)
    local_pos[..., 0, :] = global_pos[..., 0, :]
    local_rot[..., 0, :] = global_rot[..., 0, :]

    local_pos[..., 1:, :] = quat_mul_vec(
        quat_inv(global_rot[..., parents[1:J], :]),
        global_pos[..., 1:, :] - global_pos[..., parents[1:J], :],
    )
    local_rot[..., 1:, :] = quat_mul(
        quat_inv(global_rot[..., parents[1:J], :]),
        global_rot[..., 1:, :])
    return local_pos[..., 1:, :], local_rot[..., 1:, :]

def adjust_mesh2ground(vertices):
    y_dis = vertices[..., :, 1]
    y_min = y_dis.min(axis=-1).reshape(1)
    vertices[..., :, 1]-=y_min
    return vertices, y_min

def skeleton2mesh(glb_pos, glb_rot, parents, zero_pose_pelvis=0):
    # trans = glb_pos[0] - zero_pose_pelvis
    orient = glb_rot[0].clone()

    # if isinstance(init_root_mat, torch.Tensor):
    #     orient = init_root_mat[:3, :3]
    # else:
    #     orient = torch.tensor(init_root_mat)[:3, :3]

    # glb_rot = glb_rot[1:]
    # glb_pos = glb_pos[1:]

    glb_rot = matrix_to_quaternion(glb_rot)
    loc_pos, loc_rot = inverse_kinematics_motion(glb_pos, glb_rot, parents)
    body_pose = quaternion_to_axis_angle(loc_rot)
    global_orient = matrix_to_axis_angle(orient)

    # global_orient = np.zeros_like(global_orient)

    # print(f'global_orient = {global_orient}, body_pose = {body_pose[:10]}')
    smplx_model = make_smplx(type='boxing').double()
    trans = np.zeros_like(global_orient)
    # body_pose = np.zeros_like(body_pose)
    # orient = np.zeros_like(orient)
    smplx_output = smplx_model(global_orient=np.double(global_orient.reshape(1,-1)),
                             transl=np.double(trans.reshape(1,-1)),
                             body_pose=np.double(body_pose.reshape(1,-1,3)),
                            )
    # 对齐pelvis和trans
    delta = smplx_output.joints[0,0] - glb_pos[0]
    delta = delta.detach().numpy()

    faces = smplx_model.bm.faces
    verts=smplx_output.vertices.detach().numpy().squeeze() - delta
    # verts = smplx_output.vertices.detach().numpy().squeeze()
    # verts, y_min = adjust_mesh2ground(verts)

    return verts, faces, smplx_output.joints-delta
