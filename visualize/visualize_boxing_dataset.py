from wis3d import Wis3D
import trimesh
import numpy as np
import os
from tqdm import tqdm
import os
import yaml
import smplx
import torch
import pytorch3d.transforms as transforms
from easyvolcap.utils.base_utils import dotdict
from egomogen.scripts.boxing_render import adjust_scene2ground
from egomogen.utils.smplx_utils import make_smplx
from egomogen.utils.geo_transform import apply_T_on_points
from egomocap.utils.wis3d_utils import make_vis3d, vis3d_add_coords, vis3d_add_skeleton
import egomogen.utils.matrix as matrix
from egomogen.utils.data_utils import *
from egomogen.utils.motion_repr_transform import *
from PIL import Image

T_y2z = torch.FloatTensor([[
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]]]).to(dtype=torch.double) # (1, 4, 4),(x,y,z)->(x,z,-y)

def vis_scene(scene_name,scene):
    wis3d.add_mesh(scene,name=f'{scene_name}')

def load_body_params(body_params):
    human_pose = body_params[6:69] 
    human_orient = body_params[3:6]  
    human_transl = body_params[0:3] 
    betas = body_params[85:95]
    gender = body_params[95]

    return {'human_pose':human_pose,'human_orient':human_orient,'human_transl':human_transl,'gender':gender,'betas':betas}

def vis_image(motion_id,wis3d,image):
    pil_image = Image.fromarray(image)
    wis3d.add_image(pil_image,name=f'image_{motion_id}')
                

def get_motion_unnormalized(motion):
    norm_data = np.load('EgoMoGen_data/normalize/pose_series_norm.npy')
    motion = Renormalize(motion.cpu(),norm_data)
    return motion


def get_camera_unnormalized(camera_pos):
    norm_data = np.load('EgoMoGen_data/normalize/camera_trans_norm.npy')
    camera_pos = Renormalize(camera_pos.cpu(),norm_data)
    return camera_pos


def visualize_motion(glb, vis3d, batch_size, frame_idx):
    glb_trans = glb.pose_pos[batch_size, frame_idx]
    
    # vis3d_add_skeleton(vis3d, 0, pose_pos[t], parents, 'pose-pos')
    # vis3d_add_coords(vis3d, 0, pose_rot[t], pose_pos[t], 'pose-rot')
    vis3d_add_skeleton(vis3d, 0, glb_trans, parents, f'frame_{frame_idx}')


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

def visualize(data):
    # frames * batch_size * ...

    scene=trimesh.load('blender/assets/boxring_color_mesh.obj')
    height=adjust_scene2ground(scene)
    scene.vertices[:,1]-=height

    folder='EgoMoGen_data/ego_render'
    
    image = data['image']
    camera_params = data['camera_params']
    pose_series = data['pose_series'] 
    meta = data['meta']

    image = image*255
    image = image.to(dtype=torch.uint8)

    camera_params[:, :, 6:] = get_camera_unnormalized(camera_params[:, :, 6:])
    camera_matrix = torch.eye(4).tile(camera_params.shape[0],camera_params.shape[1],1,1)
    camera_rot = transforms.rotation_6d_to_matrix(camera_params[:, :, :6])
    camera_pos = camera_params[:, :, 6:]
    camera_matrix[:, :, :3, :3] = camera_rot
    camera_matrix[:, :, :3, 3] = camera_pos

    loc_params = get_motion_unnormalized(pose_series.double())
    loc_params = MoReprTrans.split_pose(loc_params)
    init_root_mat = meta.init_root_mat.double().to('cpu')

    glb = get_glb_root_by_loc(loc_params, init_root_mat)
    glb = get_glb_pose_by_loc(loc_params, glb=glb)
    glb.pose_pos = apply_T_on_points(glb.pose_pos, T_y2z)

    for batch_idx in range(len(image)):
        # wis3d = Wis3D('output/visualize_dataset',f'batch_idx = {batch_idx}')
        # wis3d.add_mesh(scene,name='scene')
        for frame_idx in tqdm(range(len(image[batch_idx]))):
            vis3d = make_vis3d(None, f'{batch_idx}_{frame_idx}', 'output/visualize_dataset',)
            vis3d.add_mesh(scene,name='scene')

            visualize_motion(glb, vis3d, batch_idx, frame_idx)

            vis3d.add_camera_trajectory(glb.root_mat[batch_idx,frame_idx].unsqueeze(0),name='pelvis')
            vis3d.add_camera_trajectory(camera_matrix[batch_idx,frame_idx][None],name='camera')
            # a = torch.eye(4)
            # a[:3,3] = glb.pose_pos[batch_idx,frame_idx,0]
            # vis3d.add_camera_trajectory(a[None], name='pose_0')

            image_seq = image[batch_idx,frame_idx]
            vis3d.add_image(image = image_seq, name = 'image')
            # vis_image(frame_idx, wis3d, image)
            #vis_motion(body_params,motion_id,scene)
            #if motion_id==1:
            #    vis_motion_seq(body_params,motion_id,scene)