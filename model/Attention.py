import torch
import torch.nn as nn
import numpy as np


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random': #打乱
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort() #保证升序输出
    return index

'''分频滤波实现'''
def split_frequencies(signal: torch.Tensor):
    B, H, E, L = signal.shape
    freq_len = L // 2 + 1
    fft_result = torch.fft.rfft(signal, dim=-1)  # 对 L 轴做傅里叶变换，结果形状: (B, H, E, L//2+1)
    power = fft_result.abs() ** 2  # 形状: (B, H, E, L//2+1)
    total_energy = power.sum(dim=-1, keepdim=True)  # 总能量，形状: (B, H, E, 1)
    cumulative_energy = power.cumsum(dim=-1)  # 按频率累积，形状: (B, H, E, L//2+1)
    energy_target = total_energy / 3 #计算能量划分阈值 (1/3 总能量)
    def find_split_indices(cum_energy, target_energy):
        """
        找到使得累积能量最接近 target_energy 的索引，保证每个部分至少包含一个频率
        """
        indices = (cum_energy >= target_energy).int().argmax(dim=-1)
        return indices.clamp(min=1)  # 最小值保证至少一个频率

    low_end = find_split_indices(cumulative_energy, energy_target)
    mid_end = find_split_indices(cumulative_energy, 2 * energy_target)
    # 生成掩码
    device = signal.device
    freq_indices = torch.arange(freq_len, device=device)[None, None, None, :]  # (1,1,1,L//2+1)

    low_mask = freq_indices < low_end.unsqueeze(-1) # 低频设置为True
    mid_mask = (freq_indices >= low_end.unsqueeze(-1)) & (freq_indices < mid_end.unsqueeze(-1)) # 中频设置为True
    high_mask = freq_indices >= mid_end.unsqueeze(-1) # 高频设置为True

    # 生成三种缺频信号
    low_mid = fft_result * (low_mask | mid_mask)+0.3*fft_result*high_mask
    low_high = fft_result * (low_mask | high_mask)+0.3*fft_result*mid_mask
    high_mid = fft_result * (high_mask | mid_mask)+0.3*fft_result*low_mask

    return low_mid, low_high, high_mid


class sparseKernelFT1d(nn.Module):
    def __init__(self,in_dim,out_dim,H,L):
        super(sparseKernelFT1d, self).__init__()
        self.scale = (1 / (in_dim * out_dim))
        self.weights1 = nn.Parameter(self.scale * torch.rand(H, in_dim//H, out_dim//H,L//2+1,dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(H, in_dim//H, out_dim//H,L//2+1,dtype=torch.float))
        self.weights1.requires_grad = True
        self.weights2.requires_grad = True

    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, x):#传入B H E L//2+1
        B,H,E,L = x.shape
        out_ft = self.compl_mul1d("bhil,hiol->bhol", x,torch.complex(self.weights1, self.weights2))
        x = torch.fft.irfft(out_ft, n=(x.size(-1)-1)*2) #B H E L
        return x


class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method) #从中随机抽取频率分量
        print('modes={}, index={}'.format(modes, self.index))

        self.n_heads = n_heads
        self.scale = (1 / (in_channels * out_channels)) #缩放分子

        #初始化复数的实部和虚部。这个人造复数实际上是为了更好的调整被转化成频域信号的X的幅度和相位，通过这两个可学习的参数实现自动优化。
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.n_heads, in_channels // self.n_heads, out_channels // self.n_heads,
                                    len(self.index), dtype=torch.float)) #H D D L
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.n_heads, in_channels // self.n_heads, out_channels // self.n_heads,
                                    len(self.index), dtype=torch.float))
        self.A = sparseKernelFT1d(in_channels,out_channels,n_heads,seq_len)
        self.B = sparseKernelFT1d(in_channels, out_channels, n_heads, seq_len)
        self.C = sparseKernelFT1d(in_channels, out_channels, n_heads, seq_len)

    # 复数乘法模块
    def compl_mul1d(self, order, x, weights):
        x_flag = True #用于标记是否是原生复数张量
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device)) ## 将实数转换为复数：实部=原值，虚部=全0
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag: # 任意一个输入原本是复数，复数乘法公式：(a+bi)(c+di) = (ac-bd) + (ad+bc)i
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else: #都是实数
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q, k, v):
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1) #B H E L
        x_ft = torch.fft.rfft(x, dim=-1) #B, H, E, L // 2 + 1
        lm,lh,hm = split_frequencies(x)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[3] or wi >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, wi] = self.compl_mul1d("bhi,hio->bho", x_ft[:, :, :, i],
                                                   torch.complex(self.weights1, self.weights2)[:, :, :, wi])
        x = torch.fft.irfft(out_ft, n=x.size(-1)) #B H E L
        lm = self.A(lm)
        lh = self.B(lh)
        hm = self.C(hm)
        return x,lm,lh,hm


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out,lm,lh,hm = self.inner_correlation(
            queries,
            keys,
            values,
        )
        out = out.view(B, L, -1)
        lm = lm.view(B, L, -1)
        lh = lh.view(B, L, -1)
        hm = hm.view(B, L, -1)

        return self.out_projection(out),self.out_projection(lm),self.out_projection(lh),self.out_projection(hm)


