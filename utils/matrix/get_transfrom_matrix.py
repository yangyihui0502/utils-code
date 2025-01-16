def get_TRS(rot_mat, pos):
    """_summary_

    Args:
        rot_mat (tensor): [N, 3, 3]
        pos (tensor): [N, 3]

    Returns:
        mat (tensor): [N, 4, 4]
    """
    if isinstance(rot_mat, torch.Tensor):
        mat = torch.eye(4, device=pos.device).repeat(pos.shape[:-1] + (1, 1))
    elif isinstance(rot_mat, np.ndarray):
        mat = np.eye(4, dtype=np.float32)
        for _ in range(len(pos.shape) - 1):
            mat = mat[None]
        mat = np.tile(mat, pos.shape[:-1] + (1, 1))
    else:
        raise ValueError
    mat[..., :3, :3] = rot_mat
    mat[..., :3, 3] = pos
    mat = normalized_matrix(mat)        # 对rotation matrix部分进行归一化
    return mat



def normalized_matrix(mat):
    if mat.shape[-1] == 4:
        rot_mat = mat[..., :-1, :-1]
    else:
        rot_mat = mat
    if isinstance(mat, torch.Tensor):
        rot_mat_norm = rot_mat / (rot_mat.norm(2, dim=-2, keepdim=True) + 1e-9)
        norm_mat = torch.zeros_like(mat)
    elif isinstance(mat, np.ndarray):
        rot_mat_norm = rot_mat / (np.linalg.norm(rot_mat, ord=2, axis=-2, keepdims=True) + 1e-9)
        norm_mat = np.zeros_like(mat)
    else:
        raise ValueError
    if mat.shape[-1] == 4:
        norm_mat[..., :-1, :-1] = rot_mat_norm
        norm_mat[..., :-1, -1] = mat[..., :-1, -1]
        norm_mat[..., -1, -1] = 1.0
    else:
        norm_mat = rot_mat_norm
    return norm_mat
