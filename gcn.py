import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.nn.pytorch.conv import GraphConv


class AttentionLayer(nn.Module):
    """注意力层，用于增强边的重要性"""

    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, in_dim]
        # 计算注意力权重
        attn_weights = torch.sigmoid(self.fc(x))
        return attn_weights * x


class ImprovedRGCN(nn.Module):
    """改进的关系图卷积网络

    改进点:
    1. 添加了注意力机制，关注重要的边
    2. 添加了多层GCN
    3. 添加了残差连接
    4. 添加了Layer Normalization
    """

    def __init__(self, num_nodes, embed_size, num_rels, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(num_nodes, embed_size)
        self.num_layers = num_layers

        # 多层GCN
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(embed_size, embed_size)
                for rel in range(num_rels)
            }))

        # 注意力层
        self.attention = AttentionLayer(embed_size)

        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_size) for _ in range(num_layers)
        ])

        # 初始化
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        for layer in self.layers:
            for conv in layer.mods.values():
                conv.reset_parameters()

        # 使用Xavier初始化嵌入层
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, g, features=None):
        """前向传播

        Args:
            g: DGL图
            features: 节点特征，如果为None则使用嵌入层

        Returns:
            h: 节点嵌入
        """
        if features is None:
            h = self.embed(torch.arange(g.num_nodes(), device=g.device))
        else:
            h = features

        h_initial = h  # 用于残差连接

        # 多层GCN传播
        for i in range(self.num_layers):
            # 关系图卷积
            h_conv = self.layers[i](g, {'node': h})['node']

            # 应用注意力
            h_attn = self.attention(h_conv)

            # 残差连接
            h = h + h_attn

            # Layer Normalization
            h = self.layer_norms[i](h)

            # 非线性激活
            h = F.relu(h)

        # 最终加上初始嵌入（全局残差）
        h = h + h_initial

        return h

class RGCN(nn.Module):
    def __init__(self, num_nodes, embed_size, num_rels):
        super().__init__()
        self.embed = nn.Embedding(num_nodes, embed_size)
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(embed_size, embed_size) for rel in range(num_rels)
        })

    def forward(self, g, features):
        h = self.conv1(g, {'node': features})
        return h