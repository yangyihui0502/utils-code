# motion: pos,rot为global
def cal_pose_series(motion, frames):
    # pose series: pose + root_ctrl
    pose = LowlevelMoRepr.cal_pose(motion, frames)
    root_ctrl = LowlevelMoRepr.cal_root_ctrl(motion, frames)
    pose_series = torch.cat([pose, root_ctrl], dim=-1)
    return pose_series

def cal_pose(motion, frames):
    B = motion.root_mat.size(0)
    root_mat, root_rot = motion.root_mat[:, frames], motion.root_rot[:, frames]
    pos = matrix.get_relative_position_to(motion.pose_pos[:, frames], root_mat)
    rot = matrix.get_mat_BtoA(root_rot[:, :, None], motion.pose_rot[:, frames])
    vel = matrix.get_relative_direction_to(motion.pose_vel[:, frames], root_mat)
    pose = torch.cat((pos, matrix_to_rotation_6d(rot), vel), dim=-1).view(B, len(frames), -1)
    return pose

def cal_root_ctrl(motion, frames):
    return motion.root_ctrl[:, frames]
    if ds: frames_next = [f + downsample for f in frames]
    else: frames_next = [f + 1 for f in frames]

    root_mat, root_rot = motion.root_mat[:, frames], motion.root_rot[:, frames]
    root_off = matrix.get_relative_position_to(motion.root_pos[:, frames_next][:, :, None], root_mat) # (T, None, 3), (T, 4, 4) -> (T, 1, 3)
    root_dir = matrix.get_mat_BtoA(root_rot, motion.root_rot[:, frames_next]) # (T, 3, 3), (T, 3, 3) -> (T, 3, 3)
    root_off_2d = root_off[..., 0, [0, 2]]
    root_dir_2d = root_dir[..., [0, 2], 2]

    root_ctrl = torch.cat((root_off_2d, root_dir_2d), dim=-1)
    return root_ctrl