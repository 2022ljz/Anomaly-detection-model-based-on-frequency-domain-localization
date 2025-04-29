import torch
import torch.nn as nn
import math


#X形状： B L D  d_model=256
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TCNBlock(nn.Module):
    def __init__(self, tab,in_dim, out_dim, dilation=1):
        super().__init__()
        # 计算因果卷积需要的padding量
        kernel_size = 3
        padding = (kernel_size - 1) * dilation//2 # 保证输入输出长度一致
        if tab == 0:
            self.conv = nn.Conv1d(in_dim, out_dim, kernel_size,
                                   padding=padding, dilation=dilation,
                                   padding_mode='circular',  # 循环填充保持序列完整性
                                   bias=False)
        if tab == 1:
            self.conv = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, 1),
                nn.GELU(),
                nn.Conv1d(in_dim, out_dim, 1)
            )

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.conv(x)


class TimeEmbedding(nn.Module):
    def __init__(self,c_in,d_model):
        super(TimeEmbedding, self).__init__()
        self.tele_dim = 1  # 遥测数据维度
        self.binary_dim = c_in - 1  # 二进制命令信号维度
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model // 2)
        self.token_conv1 = nn.Sequential(
            TCNBlock(0,self.tele_dim,d_model//8,1),
            nn.GELU(),
            TCNBlock(0,d_model//8, d_model//4,2),
            nn.GELU(),
            TCNBlock(0,d_model//4, d_model//2, 4),
        )
        self.token_conv2 = TCNBlock(0,self.binary_dim,d_model//2,1)
        self.token_conv3 = TCNBlock(1,d_model,d_model,1)


    def enhance_one_features(self,x):
        # 强化非0行的存在和所有1的存在。
        B, L, D = x.shape
        null_mask = (x.sum(dim=-1) == 0)  # (B, L)检测全0行
        enhanced = torch.ones_like(x, dtype=torch.float32)  # 创建全1基础矩阵
        col_indices = torch.arange(D, device=x.device).view(1, 1, D) + 1  # (1, 1, D) # 生成列索引矩阵（索引从1开始）
        mask = (x == 1) & (~null_mask.unsqueeze(-1))  # 非零行中的1的位置
        enhanced = torch.where(mask, col_indices.expand_as(x).float(), enhanced)  # 将原1的位置替换为列索引
        enhanced = torch.where(null_mask.unsqueeze(-1), torch.zeros_like(enhanced), enhanced)  # 恢复全零行的原始值
        return enhanced

    def forward(self,x):
        telemetry = x[:, :, :self.tele_dim]  # (B, L, 1)
        binary = x[:, :, self.tele_dim:]  # (B, L, binary_dim)
        telemetry = self.token_conv1(telemetry.permute(0, 2, 1)).transpose(1, 2)
        telemetry = self.norm(telemetry)
        binary = self.token_conv2(binary.permute(0, 2, 1)).transpose(1, 2)
        binary = self.norm(binary)
        x = torch.cat([telemetry,binary],dim = -1)
        x = self.token_conv3(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TimeEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x.contiguous())