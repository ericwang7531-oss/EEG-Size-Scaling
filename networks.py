"""定义网络结构"""
import torch
from torch import nn
from torch import Tensor
from einops import rearrange
from settings import Settings
import torch.nn.functional as F
from torch.backends import cudnn
from timm.layers import trunc_normal_
cudnn.benchmark = False
cudnn.deterministic = True


class EEGNet_LSTM(nn.Module):
    """
    EEGNet与LSTM的混合网络
    EEGNet论文DOI: 10.1088/1741-2552/aace8c
    """
    def __init__(self, n_input:int, n_output:int, custom_para:dict=None):
        """
        :param n_input: EEG通道数量
        :param n_output: 输出维度
        :param custom_para: 覆盖默认参数字典, 用于超参搜索
        """
        super(EEGNet_LSTM, self).__init__()

        net_para = Settings().EEGNet_LSTM.copy()  # 获取默认模型超参
        if custom_para is not None:  # 用自定义超参覆盖原始超参
            for key in custom_para:
                if key in net_para:
                    net_para[key] = custom_para[key]

        dr = net_para['dropout_rate']                  # 丢弃率

        n_b1k1 = net_para['block1_kernel1_num']        # Block 1 二维卷积核数量
        s_b1k1 = (1, net_para['block1_kernel1_size'])  # Block 1 二维卷积核尺寸
        d = net_para['block1_depth']                   # Block 1 深度卷积层深度
        n_b1k2 = int(n_b1k1 * d)                       # Block 1 深度卷积核数量
        s_b1p = (1, net_para['block1_pool_size'])      # Block 1 均值池化尺寸

        n_b2k = net_para['block2_kernel_num']          # Block 2 可分离卷积核数量
        s_b2k = (1, int(s_b1k1[-1] / s_b1p[-1]))       # Block 2 可分离卷积核尺寸
        s_b2p = (1, net_para['block2_pool_size'])      # Block 2 均值池化尺寸

        n_lstm_u = net_para['lstm_unit_num']           # LSTM单元数量
        n_lstm_l = net_para['lstm_layer_num']          # LSTM层数

        self.EEGNet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_b1k1, kernel_size=s_b1k1, padding='same', bias=False),  # 二维卷积层 (注意in/out_channels指的并不是脑电通道, 而是类似于图片的RBG通道)
            nn.BatchNorm2d(num_features=n_b1k1),
            nn.Conv2d(in_channels=n_b1k1, out_channels=n_b1k2, kernel_size=(n_input, 1), groups=n_b1k1, bias=False),  # 深度卷积层
            nn.BatchNorm2d(num_features=n_b1k2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=s_b1p),
            nn.Dropout(p=dr),

            nn.Conv2d(in_channels=n_b1k2, out_channels=n_b1k2, kernel_size=s_b2k, groups=n_b1k2, padding='same', bias=False),
            nn.Conv2d(in_channels=n_b1k2, out_channels=n_b2k, kernel_size=(1, 1), bias=False),  # PyTorch中需要两层Conv2d来实现可分离卷积
            nn.BatchNorm2d(num_features=n_b2k),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=s_b2p),
            nn.Dropout(p=dr)
        )

        self.LSTM = nn.LSTM(input_size=n_b2k, hidden_size=n_lstm_u, num_layers=n_lstm_l, batch_first=True, dropout=dr)

        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_lstm_u, out_features=n_output)  # 线性输出
        )

    def forward(self, x):
        """
        :param x: 输入的EEG批次, 形状为(batch_size, n_chan, win_len)
        """
        x = x.unsqueeze(1)          # 输出形状(batch_size, 1, n_chan, win_len)
        x = self.EEGNet(x)          # 输出形状(batch_size, n_b2k, 1, time), time=win_len//(s_b1p*s_b2p)
        x = x.squeeze(2)            # 输出形状(batch_size, n_b2k, time)
        x = x.permute(0, 2, 1)      # 匹配LSTM输入要求, 输出形状(batch_size, time, n_b2k)
        x, _ = self.LSTM(x)         # 丢弃隐藏状态, 输出形状(batch_size, time, n_lstm_u)
        x = x[:, -1, :]             # 取出最后一个时间步的输出, 形状(batch_size, n_lstm_u)
        out = self.output_layer(x)  # 输出形状(batch_size, n_output)
        return out


class DBConformer(nn.Module):
    """
    DBConformer
    原作者: Ziwei Wang
    原论文: DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding
    """
    def __init__(
            self, ch_num, time_sample_num, patch_size, n_classes,
            emb_size=40, spa_dim=16, kern_size=63, tem_depth=5, chn_depth=5,
            gate_flag=False, posemb_flag=True, chn_atten_flag=True, branch='all'
    ):
        super().__init__()

        self.embedding = _PatchEmbeddingTemporal(
            in_planes=ch_num,  # number of channels
            out_planes=emb_size,  # Default 40
            kernel_size=kern_size,
            radix=1,
            patch_size=patch_size,  # needs to be divisible by the number of time points
        )
        self.channel_embedding = _PatchEmbeddingSpatial(spa_dim=spa_dim, emb_size=emb_size)  # Default 16
        self.P = (time_sample_num-1) // patch_size  # Example: 1000 // 125 = 8
        # self.P = time_sample_num // patch_size
        self.C = ch_num  # number of channels
        self.D = emb_size
        self.gate_flag = gate_flag  # Default False, due to the reduced performance
        self.posemb_flag = posemb_flag  # Default True
        self.branch = branch  # Default 'all', options=[all, temporal]
        self.chn_atten_flag = chn_atten_flag  # Default True

        if self.posemb_flag:
            self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

        self.temporal_transformer = _TransformerEncoder(tem_depth, emb_size)
        self.spatial_transformer = _TransformerEncoder(chn_depth, emb_size)
        if self.gate_flag or self.branch == 'temporal' or self.branch == 'spatial':
            self.gate_fc = _Gate_FC(emb_size)  # dual-branch weighting aggregation
            self.classifier = _ClassificationHead(emb_size, n_classes)
        else:
            self.classifier = _ClassificationHead(emb_size * 2, n_classes)

            if self.chn_atten_flag:
                self.spatial_attn_pool = nn.Sequential(
                    nn.Linear(emb_size, emb_size),  # D → D
                    nn.Tanh(),
                    nn.Linear(emb_size, 1),  # D → 1 (score per channel)
                )

    def forward(self, x):  # x: (B, 1, C, T)
        x = x.squeeze(1)  # → (B, C, T)
        x_embed = self.embedding(x)  # → (B, P, D)
        x_embed_spatial = self.channel_embedding(x)  # (B, C, D)
        if self.posemb_flag:
            x_embed = x_embed + self.pos_embedding_temporal  # temporal positional encoding
            x_embed_spatial = x_embed_spatial + self.pos_embedding_spatial  # spatial positional encoding

        # Temporal Transformer (attention over time dimension)
        x_temporal = self.temporal_transformer(x_embed)  # (B, P, D)
        # Spatial Transformer (attention over channels interpreted as tokens)
        x_spatial = self.spatial_transformer(x_embed_spatial)  # (B, C, D)

        x_fused, out = None, None
        if self.branch == 'temporal':
            x_fused = x_temporal.mean(dim=1)
            _, out = self.classifier(x_fused)  # out: (B, n_classes)
        elif self.branch == 'spatial':  # Using S-Conformer-only doesn't work
            x_fused = x_spatial.mean(dim=1)
            _, out = self.classifier(x_fused)  # out: (B, n_classes)
        elif self.branch == 'all':
            if self.gate_flag:
                # gated-fusion
                gate = torch.sigmoid(
                    self.gate_fc(torch.cat([x_temporal.mean(dim=1), x_spatial.mean(dim=1)], dim=-1)))  # shape: (B, D)
                x_fused = gate * x_spatial.mean(dim=1) + (1 - gate) * x_temporal.mean(dim=1)
            else:
                if self.chn_atten_flag:
                    # Attention Scores
                    x_t = x_temporal.mean(dim=1)
                    attn_scores = self.spatial_attn_pool(x_spatial)  # (B, C, 1)
                    attn_weights = torch.softmax(attn_scores, dim=1)  # (B, C, 1)
                    x_s = torch.sum(attn_weights * x_spatial, dim=1)  # (B, D)
                    x_fused = torch.cat([x_t, x_s], dim=-1)  # → (B, 2*D)
                else:
                    # Mean pooling
                    x_fused = torch.cat([
                        x_temporal.mean(dim=1),
                        x_spatial.mean(dim=1)
                    ], dim=-1)  # → (B, 2*D)
            _, out = self.classifier(x_fused)  # out: (B, n_classes)
        return x_fused, out


class _Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class _InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out


class _Stem(nn.Module):
    def __init__(self, in_planes, out_planes = 64, kernel_size=63, patch_size=125, radix=2):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix

        self.sconv = _Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups = radix),
                           bn=nn.BatchNorm1d(self.mid_planes), activation=None)

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(
                _Conv(
                    conv=nn.Conv1d(
                        self.out_planes, self.out_planes, kernel_size,
                        stride=1, groups=self.out_planes, padding=kernel_size // 2, bias=False,
                    ),
                    bn=nn.BatchNorm1d(self.out_planes), activation=None
                )
            )
            kernel_size //= 2

        self.interFre = _InterFre()

        self.downSampling = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(0.5)  #

    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]
        out = self.interFre(out)
        out = out[:, :, :-1]  # Example: 14001 has 1001 time points, we exclude the final point
        out = self.downSampling(out)
        out = self.dp(out)
        return out


class _PatchEmbeddingTemporal(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, radix, patch_size):
        """
        Outputs patch embeddings of shape (B, P, D)
        """
        super().__init__()
        self.stem = _Stem(
            in_planes=in_planes * radix,
            out_planes=out_planes,
            kernel_size=kernel_size,
            patch_size=patch_size,
            radix=radix
        )
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: (B, C, T)
        out = self.stem(x)         # (B, D, P)
        out = out.permute(0, 2, 1) # → (B, P, D)
        return out


class _PatchEmbeddingSpatial(nn.Module):
    def __init__(self, spa_dim, emb_size=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, spa_dim, kernel_size=25, stride=5, padding=12),  # Output: (B*C, spa_dim, T')
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),  # Output: (B*C, spa_dim, 1)
            nn.Flatten(),             # → (B*C, 16)
            nn.Linear(spa_dim, emb_size)   # → (B*C, emb_size)
        )

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        x = x.unsqueeze(2)         # (B, C, 1, T)
        x = x.reshape(B * C, 1, T)  # → (B*C, 1, T)
        x = self.encoder(x)        # → (B*C, emb_size)
        x = x.view(B, C, -1)       # → (B, C, emb_size)
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class _TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[_TransformerEncoderBlock(emb_size) for _ in range(depth)])


class _ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class _Gate_FC(nn.Sequential):
    def __init__(self, emb_size):
        super().__init__()
        self.fc = nn.Linear(emb_size * 2, emb_size)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out
