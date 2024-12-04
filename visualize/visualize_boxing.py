from wis3d import Wis3D
import trimesh
import numpy as np
import os
from tqdm import tqdm
import os
import yaml
import smplx
import torch
from egomogen.scripts.boxing_render import adjust_scene2ground
from egomogen.utils.smplx_utils import make_smplx
from PIL import Image

def vis_scene(scene_name,scene):
    wis3d.add_mesh(scene,name=f'{scene_name}')

def load_body_params(body_params):
    human_pose = body_params[6:69] 
    human_orient = body_params[3:6]  
    human_transl = body_params[0:3] 
    betas = body_params[85:95]
    gender = body_params[95]

    return {'human_pose':human_pose,'human_orient':human_orient,'human_transl':human_transl,'gender':gender,'betas':betas}

def load_camera(params):
    origin_matrix=params[69:85].reshape(4,4)
    camera_matrix=np.linalg.inv(origin_matrix)
    #camera_matrix=origin_matrix
    trans_matrix=np.eye(4)
    trans_matrix[:3,3]=camera_matrix[:3,3]
    return trans_matrix.reshape(1,4,4)
    #return camera_matrix.reshape(1,4,4)

def vis_image(motion_id,wis3d,image):
    pil_image = Image.fromarray(image)
    wis3d.add_image(pil_image,name=f'image_{motion_id}')

def vis_motion_only(body_params,motion_id):
    smplx_output=smplx_model_male(global_orient=torch.from_numpy(np.double(body_params['human_orient']).reshape(1,-1)),
                                transl=torch.from_numpy(np.double(body_params['human_transl']).reshape(1,-1)),
                                body_pose=torch.from_numpy(np.double(body_params['human_pose']).reshape(1,-1,3)),
                                betas = torch.from_numpy(np.double(body_params['betas']).reshape(1,-1))
                                )
    faces=smplx_model_male.bm.faces

    verts=smplx_output.vertices.detach().numpy().squeeze()
    
    mesh=trimesh.Trimesh(vertices=verts,faces=faces)

    wis3d.add_mesh(mesh,name=f'ego_motion_{motion_id}')

def vis_motion_seq(body_params,motion_id,camera_matrix):
    smplx_output=smplx_model_male(global_orient=torch.from_numpy(np.double(body_params['human_orient']).reshape(1,-1)),
                                transl=torch.from_numpy(np.double(body_params['human_transl']).reshape(1,-1)),
                                body_pose=torch.from_numpy(np.double(body_params['human_pose']).reshape(1,-1,3)),
                                betas = torch.from_numpy(np.double(body_params['betas']).reshape(1,-1))
                                )
    faces=smplx_model_male.bm.faces
    verts=smplx_output.vertices.detach().numpy().squeeze()
    
    mesh=trimesh.Trimesh(vertices=verts,faces=faces)

    wis3d.add_mesh(mesh,name=f'motion_{motion_id}')
    wis3d.add_camera_trajectory(camera_matrix,name=f'camera_{motion_id}')
                
smplx_model_male = make_smplx(type='boxing').double()

scene=trimesh.load('blender/assets/boxring_color_mesh.obj')
height=adjust_scene2ground(scene)
scene.vertices[:,1]-=height

motion_sum=9
folder='data/ego_render'

for file_name in os.listdir(folder):
    params = np.load(os.path.join(folder, file_name, f'smplx_params/smplx_params.npy'))
    images = np.load(os.path.join(folder, file_name, f'rgb/rgb.npy'),mmap_mode="r")

    # wis3d = Wis3D('output/visualize_boxing',f'{file_name}')
    # wis3d.add_mesh(scene,name='scene')

    # for motion_id in tqdm(range(2099,2300,1)):
    for motion_id in tqdm(range(0,len(params),15)):
        wis3d = Wis3D('output/visualize_boxing',f'{file_name}_{motion_id}')
        wis3d.add_mesh(scene,name='scene')

        body_params=load_body_params(params[motion_id,0])
        ego_params=load_body_params(params[motion_id,1])    
        camera_matrix=load_camera(params[motion_id,0])

        vis_motion_only(ego_params,motion_id)
        vis_motion_seq(body_params,motion_id,camera_matrix)

        image = images[motion_id-2099]
        vis_image(motion_id,wis3d,image)
        #vis_motion(body_params,motion_id,scene)
        #if motion_id==1:
        #    vis_motion_seq(body_params,motion_id,scene)