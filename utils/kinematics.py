def inverse_kinematics_motion(
    global_pos,
    global_rot,
    parents=SMPLH_PARENTS,
):
    """
    Args:
        global_pos : (B, T, J-1, 3)
        global_rot (q) : (B, T, J-1, 4)
        parents : SMPLH_PARENTS
    Returns:
        local_pos : (B, T, J-1, 3)
        local_rot (q) : (B, T, J-1, 4)
    """
    J = 22
    local_pos = quat_mul_vec(
        quat_inv(global_rot[..., parents[1:J], :]),
        global_pos - global_pos[..., parents[1:J], :],
    )
    local_rot = quat_mul(
        quat_inv(global_rot[..., parents[1:J], :]),
        global_rot),
    return local_pos, local_rot


def forward_kinematics_motion(
    root_orient,
    pose_body,
    trans,
    joints_zero,
    smplh_parents=SMPLH_PARENTS,
    rot_type='pose_body',
):
    """
    Args:
        root_orient : (B, T, 3) for `pose_body`, (B, T, 3, 3) for `R`
        pose_body : (B, T, (J-1)*3) for `pose_body`, (B, T, J-1, 3, 3) for `R`
        trans : (B, T, 3)
        joints_zero : (B, J, 3)
        rot_type: pose_body, R
    Returns:
        posed_joints: (B, T, J, 3)
        R_global: (B, T, J, 3, 3)
        A: (B, T, J, 4, 4)
    """
    J = joints_zero.shape[1]  # 22 for smplh
    B, T = root_orient.shape[:2]
    if rot_type == 'pose_body':
        rot_aa = torch.cat([root_orient, pose_body], dim=-1).reshape(B, T, -1, 3)
        rot_mats = axis_angle_to_matrix(rot_aa)  # (B, T, J, 3, 3)
    elif rot_type == 'R':
        rot_mats = torch.cat([root_orient[:, :, None], pose_body], dim=2)

    joints_zero = torch.unsqueeze(joints_zero, dim=-1)  # (B, J, 3, 1)
    rel_joints = joints_zero.clone()
    rel_joints[:, 1:] -= joints_zero[:, smplh_parents[1:J]]
    rel_joints = rel_joints[:, None].expand(-1, T, -1, -1, -1)  # (B, T, J, 3, 1)

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(B*T, J, 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, J):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[smplh_parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # --- Returns
    # The last column of the transformations contains the posed joints, in the cam
    # NOTE: FK adds trans
    posed_joints = transforms[:, :, :3, 3].reshape(B, T, -1, 3) + trans.unsqueeze(2)

    # The rot of each joint in the cam
    R_global = transforms[:, :, :3, :3].reshape(B, T, J, 3, 3)

    # Relative transform, for LBS
    joints_homogen = F.pad(joints_zero, [0, 0, 0, 1])  # (B, J, 3->4, 1)
    joints_homogen = joints_homogen[:, None].expand(-1, T, -1, -1, -1)  # (B, T, J, 4, 1)
    transforms = transforms.reshape(B, T, J, 4, 4)
    A = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0])

    return posed_joints, R_global, A


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