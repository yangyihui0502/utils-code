from utils.data_utils import save_norm_data
# 传入所有train_data,按idx统一计算norm
def normalize(self, data, norm_file):
        shape = data[0].shape
        all_data = [data[i].view(-1, shape[-1]) for i in range(len(data))]
        all_data = torch.cat(all_data, dim=0)
        norm_data = all_data.mean(dim=0), all_data.std(dim=0)
        norm_data = torch.stack(norm_data, dim=0)
        save_norm_data(norm_file, norm_data.detach().cpu().numpy())
        
        return norm_data