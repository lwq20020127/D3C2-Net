import torch
from torch import nn
import torch.nn.functional as F
from utils import *


class RB(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(nf, nf, 3, padding=2, dilation=2),
        )
    
    def forward(self, x):
        return x + self.body(x)

class Stage(nn.Module):
    def __init__(self, nf, nb):
        super().__init__()
        self.hpn = nn.Sequential(
            nn.Conv2d(1, nf, 1), nn.Sigmoid(),
            nn.Conv2d(nf, 4, 1), nn.Softplus(),
        )
        self.pmn = nn.Sequential(
            nn.Conv2d(1, nf, 3, padding=1),
            *[RB(nf) for _ in range(nb)],
            nn.Conv2d(nf, 1, 3, padding=1),
        )
        self.ptsn = nn.Sequential(
            nn.Conv2d(1 + nf, nf, 3, padding=1),
            *[RB(nf) for _ in range(nb)],
        )
        self.scale_alpha = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        z, alpha, cs_ratio, Phi, PhiT, d_weight, d, D, y = x # z: (b, 1, h, w), alpha: (b, c, h, w)
        rho, mu, eta, beta = self.hpn(cs_ratio).chunk(4, dim=1)

        # 1. Image-Domain Block (IDB)
        z = z - rho * (PhiT(Phi(z) - y) + mu * (z - d(alpha)))
        alpha = alpha + self.pmn[1:-1](self.pmn[:1](z) + self.scale_alpha * alpha)
        z = z + self.pmn[-1:](alpha)

        # 2. Convolutional-Coding-Domain Block (CCDB)
        z_fft = torch.view_as_real(torch.fft.rfft2(z)).unsqueeze(2)
        alpha_fft = torch.view_as_real(torch.fft.rfft2(alpha)).unsqueeze(1)
        alpha = SolveFFT(alpha_fft, D, z_fft, eta, z.shape[-2:])
        alpha = alpha + self.ptsn(torch.cat([alpha, beta.expand_as(z)], dim=1))
        return z, alpha, cs_ratio, Phi, PhiT, d_weight, d, D, y

class D3C2Net(nn.Module):
    def __init__(self, T=25, nf=32, nb=2, k=5, B=32):
        super().__init__()
        global N
        N = B * B
        U, S, V = torch.linalg.svd(torch.randn(N, N))
        self.Phi_weight = nn.Parameter((U @ V).reshape(N, 1, B, B))
        self.Phi = lambda w: F.conv2d(w, self.Phi_weight.to(w.device), stride=B)
        self.PhiT = lambda w: F.conv_transpose2d(w, self.Phi_weight.to(w.device), stride=B)
        self.d_weight = nn.Parameter(torch.zeros(1, nf, k, k))
        self.d = lambda w: F.conv2d(w, self.d_weight.to(w.device), padding=k//2)
        self.init = nn.Sequential(
            nn.Conv2d(2, nf, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(nf, nf, 3, padding=1)
        )
        self.body = nn.Sequential(*[Stage(nf, nb) for _ in range(T)])

    def forward(self, x, cs_ratio):
        b = x.shape[0]
        q = (cs_ratio * N).ceil().reshape(b, 1)
        mask = (torch.arange(N,device=x.device).view(1,N).expand(x.shape[0],N) < q).view(b,N,1,1)
        # print(mask)
        Phi = lambda w: (self.Phi(w) * mask)
        PhiT, d_weight, d = self.PhiT, self.d_weight, self.d
        D = p2o(d_weight.unsqueeze(0), x.shape[-2:])
        cs_ratio = cs_ratio.reshape(b, 1, 1, 1)

        # 1. CS Sampling
        y = Phi(x) # (b, N, h//B, w//B)

        # 2. CS Reconstruction
        alpha = torch.cat([PhiT(y), cs_ratio.expand_as(x)], dim=1)
        alpha = self.init(alpha) # (b, nf, h, w)
        z, alpha = self.body([d(alpha), alpha, cs_ratio, Phi, PhiT, d_weight, d, D, y])[:2]
        return d(alpha)

if __name__ == '__main__':
    model = D3C2Net().cuda()
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#Param.', param_cnt/1e6, 'M')