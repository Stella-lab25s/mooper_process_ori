import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvKB(nn.Module):
    """
    ConvKB 模型用于知识图谱链接预测/评分。
    将头实体、关系、尾实体的嵌入堆叠起来，
    然后应用一维卷积和全连接层来计算三元组的分数。
    """
    def __init__(self, embed_size, num_filters=100, dropout_rate=0.2):
        """
        ConvKB 模型初始化 .
        Args:
            embed_size (int): 实体和关系嵌入的维度.
            num_filters (int): 卷积层输出通道数 (滤波器数量).
            dropout_rate (float): Dropout 比率.
        """
        super().__init__()
        self.embed_size = embed_size
        self.num_filters = num_filters

        # 以下是说明，仅供参考
        # 一维卷积层：
        # 输入通道为 3 (h, r, t 嵌入堆叠)
        # 输出通道为 num_filters
        # 卷积核大小为 1 (实际上是跨嵌入维度进行卷积，但这里kernel_size=1表示不考虑相邻"词")

        # ConvE 使用二维卷积，ConvKB 使用一维卷积 + 全连接
        # 这里实现的是 ConvKB 的结构
        # 输入形状: (batch_size, 3, embed_size)
        # 输出形状: (batch_size, num_filters, embed_size)
        self.conv1d = nn.Conv1d(in_channels=3, out_channels=num_filters, kernel_size=1, stride=1)

        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        # 输入维度是卷积输出展平后的大小 (num_filters * embed_size)
        # 输出维度为 1 (三元组的得分)
        self.fc = nn.Linear(num_filters * embed_size, 1)

        # 可以添加 Batch Normalization，取消注释就可以
        # self.bn0 = nn.BatchNorm1d(3) # 输入 BN
        # self.bn1 = nn.BatchNorm1d(num_filters) # 卷积后 BN

    def forward(self, h, r, t):
        """
        前向传播
        Args:
            h (torch.Tensor): 头实体嵌入 (batch_size, embed_size).
            r (torch.Tensor): 关系嵌入 (batch_size, embed_size).
            t (torch.Tensor): 尾实体嵌入 (batch_size, embed_size).
        Returns:
            torch.Tensor: 三元组的得分 (batch_size, 1).
        """
        # 将 h, r, t 嵌入在维度 1 上堆叠
        # 形状变为: (batch_size, 3, embed_size)
        x = torch.stack([h, r, t], dim=1)

        # 可选：应用输入 Batch Normalization
        # x = self.bn0(x)

        # 应用一维卷积和 ReLU激活函数
        # 输出形状: (batch_size, num_filters, embed_size)
        x = F.relu(self.conv1d(x))

        # 可选：应用卷积后 Batch Normalization
        # x = self.bn1(x)

        # 将卷积输出展平
        # 形状变为: (batch_size, num_filters * embed_size)
        x = x.view(x.size(0), -1) # x.view(batch_size, -1)

        # 应用 Dropout
        x = self.dropout(x)

        # 应用全连接层得到最终分数
        # 输出形状: (batch_size, 1)
        score = self.fc(x)

        return score