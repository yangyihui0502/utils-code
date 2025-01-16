# 将要预测的gt信息遮住

def prep_condition_mask(self, data, pose_len):
    # data: BS X T' X D  
    # Condition part is zeros, while missing part is ones. 
    mask = torch.ones_like(data).to(data.device)

    current_pose_sid = (pose_len + 9) * (self.input_frames-1)
    mask[:, :, current_pose_sid:current_pose_sid+pose_len] = torch.zeros(data.shape[0], data.shape[1], pose_len).to(data.device)

    return mask