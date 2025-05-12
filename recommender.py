import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """自注意力机制"""

    def __init__(self, embed_dim):
        super().__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # 用于计算注意力权重的额外线性层
        self.attention_fc = nn.Linear(embed_dim, embed_dim)  # 输出与 embed_dim 相同维度的权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [batch_size, embed_dim]

        # Q, K, V 都是对 x 的线性变换
        q = self.query_layer(x)  # [batch_size, embed_dim]
        k = self.key_layer(x)  # [batch_size, embed_dim]
        v = self.value_layer(x)  # [batch_size, embed_dim]

        #  ----------!!! 主要改动!!!!-----------
        # 简化：通过 q 和 k 的交互
        attention_scores = torch.tanh(q * k)
        attention_weights = torch.sigmoid(self.attention_fc(attention_scores))  # [batch_size, embed_dim]

        # 用注意力权重加权 Value 向量
        weighted_v = attention_weights * v  # 元素级乘法

        return weighted_v  # 返回加权后的向量 [batch_size, embed_dim]


class ImprovedGroupRecommender(nn.Module):
    """改进的群组推荐模型

    改进点:
    1. 添加自注意力机制
    2. 多层感知器结构优化
    3. 添加批归一化
    4. 增加模型深度
    """

    def __init__(self, embed_size, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3): # 增加参数
        super().__init__()
        self.embed_size = embed_size

        # 自注意力层
        self.user_attention = SelfAttention(embed_size)
        self.item_attention = SelfAttention(embed_size)
        self.group_attention = SelfAttention(embed_size)

        # 特征交互层 (MLP)
        # 输入维度是 3 * embed_size 因为拼接了三个经过注意力处理的嵌入
        self.interaction = nn.Sequential(
            nn.Linear(3 * embed_size, hidden_dim1), # 举例：3*64 -> 128
            nn.BatchNorm1d(hidden_dim1), # 添加批归一化
            nn.LeakyReLU(0.1), # LeakyReLU激活
            nn.Dropout(dropout_rate), # Dropout
            nn.Linear(hidden_dim1, hidden_dim2),    # 举例： 128 -> 64
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.5) # 可以使用不同的 dropout 率
        )

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2), # 进一步降维，例如 64 -> 32
            nn.BatchNorm1d(hidden_dim2 // 2),
            nn.LeakyReLU(0.1),
            # nn.Dropout(dropout_rate * 0.5), # 可选：在最后一层前也加 dropout
            nn.Linear(hidden_dim2 // 2, 1), # 输出层
            nn.Sigmoid()       # ------- 修改：Sigmoid激活函数进行输出约束，约束到[0,1]
        )
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # Xavier初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # 偏置初始化为0
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, user_emb, item_emb, group_emb):
        """前向传播

        Args:
            user_emb: 用户嵌入, shape: [batch_size, embed_size]
            item_emb: 物品嵌入, shape: [batch_size, embed_size]
            group_emb: 群组嵌入, shape: [batch_size, embed_size]

        Returns:
            pred: 预测评分, shape: [batch_size, 1]
        """
        # 应用自注意力
        user_attn = self.user_attention(user_emb)
        item_attn = self.item_attention(item_emb)
        group_attn = self.group_attention(group_emb)

        # 特征交互
        concat_emb = torch.cat([user_attn, item_attn, group_attn], dim=1)
        interaction_emb = self.interaction(concat_emb)

        # 预测
        pred = self.predictor(interaction_emb)

        return pred

class GroupRecommender(nn.Module):
    """
    简单的群组推荐模型（基础）。
    接收用户、物品和群组的嵌入，将它们拼接起来，
    然后通过一个多层感知机 (MLP) 来预测评分。
    """
    def __init__(self, embed_size, hidden_dim=256, dropout=0.3):
        """
        初始化 GroupRecommender 模型.
        Args:
            embed_size (int): 输入嵌入的维度 (用户、物品、群组嵌入维度相同).
            hidden_dim (int): MLP 隐藏层的维度.
            dropout_rate (float): Dropout 比率.
        """

          # 保持一致，初始化添加Sigmoid激活函数以进行约束
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * embed_size, hidden_dim), # 输入层
            nn.ReLU(), # 激活函数
            nn.Dropout(dropout), # Dropout层
            nn.Linear(hidden_dim, 1), # 输出层
            nn.Sigmoid()   #-------- 输出约束

        )
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, user_emb, item_emb, group_emb):
        """前向传播"""
        # 拼接嵌入
        x = torch.cat([user_emb, item_emb, group_emb], dim=1)
        # 通过全连接层
        score = self.fc(x)
        return score