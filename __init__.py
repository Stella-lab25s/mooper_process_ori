# 模型包初始化文件
from .convkb import ConvKB
from .gcn import RGCN , ImprovedRGCN # 可以取消注释 ImprovedRGCN 如果需要
from .recommender import GroupRecommender , ImprovedGroupRecommender

__all__ = [
    'ConvKB',
    'RGCN',
    'ImprovedRGCN',
    'GroupRecommender',
    'ImprovedGroupRecommender'
]
