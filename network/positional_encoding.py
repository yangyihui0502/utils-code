# 把timestep转为高维嵌入信息
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
sinu_pos_emb = SinusoidalPosEmb(dim)
fourier_dim = dim

self.time_mlp = nn.Sequential(
    sinu_pos_emb,
    nn.Linear(fourier_dim, time_dim),
    nn.GELU(),
    nn.Linear(time_dim, d_model)
)