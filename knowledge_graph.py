import warnings

import torch
import dgl
import pandas as pd # 导入 pandas
import traceback # 导入 traceback

class KnowledgeGraphBuilder:
    def __init__(self, processed_data, user_col='user_id_mapped', item_col='item_id_mapped', group_col='group_id', rating_col='rating'): # ** 修改点: 添加列名参数 **
        """
        初始化 KnowledgeGraphBuilder。

        Args:
            processed_data (pd.DataFrame): 包含映射后 ID 和原始评分的 DataFrame。
                                           需要包含 user_col, item_col, group_col, rating_col 指定的列。
            user_col (str): 用户映射后 ID 列名.
            item_col (str): 物品映射后 ID 列名.
            group_col (str): 群组 ID 列名.
            rating_col (str): 原始评分列名.
        """
        print("DEBUG: Initializing KnowledgeGraphBuilder...")
        if not isinstance(processed_data, pd.DataFrame):
             raise ValueError("输入数据必须是 Pandas DataFrame")
        self.data = processed_data
        # ** 修改点: 存储列名 **
        self.user_col = user_col
        self.item_col = item_col
        self.group_col = group_col
        self.rating_col = rating_col

        # 检查必需列
        required_cols = [user_col, item_col, group_col, rating_col]
        if not all(col in self.data.columns for col in required_cols):
             raise ValueError(f"输入 DataFrame 缺少必需列: {required_cols}. Available: {self.data.columns.tolist()}")

        self.entity2id = {}
        # ** 修改: 更新了关系类型定义  **
        self.relation2id = {
            'user_rated_item': 0,       # 用户评分物品
            'item_rated_by_user': 1,    # 反向：物品被用户评分
            'user_interacts_item': 2,   # 用户交互物品 (高分)
            'item_interacted_by_user': 3,# 反向：物品被用户交互
            'user_belongs_to_group': 4, # 用户属于群组
            'group_has_member_user': 5, # 反向：群组有成员
            'group_prefers_item': 6,    # 群组偏好物品
            'item_preferred_by_group': 7 # 反向：物品被群组偏好
        }
        self.num_entities = 0
        self.num_relations = len(self.relation2id)
        self.graph = None
        # 存储映射后的 ID 范围 (从数据中动态获取)
        self.num_users = self.data[self.user_col].nunique()
        self.num_items = self.data[self.item_col].nunique()
        self.num_groups = self.data[self.group_col].nunique()
        # 计算实体偏移量
        self.entity_offset = {'user': 0, 'item': self.num_users, 'group': self.num_users + self.num_items}


    def build_mappings(self):
        """
        创建实体到全局 ID 的映射。
        """
        print("DEBUG: 开始构建实体全局 ID 映射...")
        self.num_entities = self.num_users + self.num_items + self.num_groups
        self.entity2id = {} # 清空旧映射

        # 用户映射 (0 to num_users-1) -> (0 to num_users-1)
        user_ids_mapped = sorted(self.data[self.user_col].unique())
        for idx, user_mapped_id in enumerate(user_ids_mapped):
             # 假设 user_mapped_id 已经是 0-based 连续的
             global_id = self.entity_offset['user'] + idx # 或者直接用 user_mapped_id 如果保证连续
             self.entity2id[('user', user_mapped_id)] = global_id

        # 物品映射 (0 to num_items-1) -> (num_users to num_users+num_items-1)
        item_ids_mapped = sorted(self.data[self.item_col].unique())
        for idx, item_mapped_id in enumerate(item_ids_mapped):
             global_id = self.entity_offset['item'] + idx
             self.entity2id[('item', item_mapped_id)] = global_id

        # 群组映射 (0 to num_groups-1) -> (num_users+num_items to num_entities-1)
        group_ids = sorted(self.data[self.group_col].unique())
        for idx, group_id in enumerate(group_ids):
             global_id = self.entity_offset['group'] + idx
             self.entity2id[('group', group_id)] = global_id

        # 验证 self.num_entities 是否与最大 global_id + 1 匹配
        max_calculated_gid = self.entity_offset['group'] + len(group_ids) -1
        if self.num_entities != max_calculated_gid + 1:
             warnings.warn(f"Calculated num_entities ({self.num_entities}) does not match max global id ({max_calculated_gid}). Check mapping logic.")
             self.num_entities = max_calculated_gid + 1 # 更新为实际值

        print(f"实体映射完成: 总共 {self.num_entities} 个实体。")
        print(f"- 用户: {self.num_users} (全局 ID 范围: {self.entity_offset['user']} to {self.entity_offset['item']-1})")
        print(f"- 物品: {self.num_items} (全局 ID 范围: {self.entity_offset['item']} to {self.entity_offset['group']-1})")
        print(f"- 群组: {self.num_groups} (全局 ID 范围: {self.entity_offset['group']} to {self.num_entities-1})")

        return self

    def _get_entity_id(self, entity_type, type_specific_id):
        """辅助函数：获取实体的全局 ID (使用映射字典)"""
        key = (entity_type, type_specific_id)
        global_id = self.entity2id.get(key)
        # if global_id is None:
        #      print(f"DEBUG: Warning - Cannot find global ID for key: {key}")
        return global_id


    def build_graph(self, rating_threshold=3.5):
        """
        构建 DGL 知识图谱。
        """
        if not self.entity2id:
            raise ValueError("实体映射尚未创建，请先调用 build_mappings()")

        print("DEBUG: 开始构建 DGL 图...")
        triples = [] # (head_global_id, relation_id, tail_global_id)

        # --- 添加关系 ---
        print("DEBUG: 添加关系边...")
        # 1 & 2: 用户-物品关系
        for _, row in self.data.iterrows():
            user_mapped_id = row[self.user_col]
            item_mapped_id = row[self.item_col]
            rating = row[self.rating_col]
            user_gid = self._get_entity_id('user', user_mapped_id)
            item_gid = self._get_entity_id('item', item_mapped_id)

            if user_gid is None or item_gid is None: continue

            # rated 关系
            triples.append((user_gid, self.relation2id['user_rated_item'], item_gid))
            triples.append((item_gid, self.relation2id['item_rated_by_user'], user_gid))

            # interacts 关系
            if rating > rating_threshold:
                triples.append((user_gid, self.relation2id['user_interacts_item'], item_gid))
                triples.append((item_gid, self.relation2id['item_interacted_by_user'], user_gid))

        # 3: 用户-群组关系
        user_group_data = self.data[[self.user_col, self.group_col]].drop_duplicates()
        for _, row in user_group_data.iterrows():
            user_mapped_id = row[self.user_col]
            group_id = row[self.group_col]
            user_gid = self._get_entity_id('user', user_mapped_id)
            group_gid = self._get_entity_id('group', group_id)

            if user_gid is None or group_gid is None: continue

            triples.append((user_gid, self.relation2id['user_belongs_to_group'], group_gid))
            triples.append((group_gid, self.relation2id['group_has_member_user'], user_gid))

        # 4: 群组-物品偏好关系
        group_item_avg_ratings = self.data.groupby([self.group_col, self.item_col])[self.rating_col].mean().reset_index()
        preferred_items = group_item_avg_ratings[group_item_avg_ratings[self.rating_col] > rating_threshold]
        for _, row in preferred_items.iterrows():
            group_id = row[self.group_col]
            item_mapped_id = row[self.item_col]
            group_gid = self._get_entity_id('group', group_id)
            item_gid = self._get_entity_id('item', item_mapped_id)

            if group_gid is None or item_gid is None: continue

            triples.append((group_gid, self.relation2id['group_prefers_item'], item_gid))
            triples.append((item_gid, self.relation2id['item_preferred_by_group'], group_gid))

        # --- 构建 DGL 图 ---
        if not triples:
             warnings.warn("没有生成任何三元组，图将为空。")
             src, dst, rel = [], [], []
        else:
             unique_triples = sorted(list(set(triples)))
             print(f"总共生成 {len(triples)} 条三元组，去重后剩余 {len(unique_triples)} 条。")
             src = [t[0] for t in unique_triples]
             rel = [t[1] for t in unique_triples]
             dst = [t[2] for t in unique_triples]

        print(f"构建 DGL 图，包含 {len(src)} 条边...")
        try:
             self.graph = dgl.graph((src, dst), num_nodes=self.num_entities)
             self.graph.edata['rel_type'] = torch.LongTensor(rel)
        except dgl.DGLError as e:
             print(f"DGL 构建图时出错: {e}")
             print("可能原因：节点 ID 超出范围或数据类型错误。")
             print(f"num_entities: {self.num_entities}")
             if src: print(f"Max src ID: {max(src)}, Max dst ID: {max(dst)}")
             traceback.print_exc()
             raise
        except Exception as e:
             print(f"构建图时发生未知错误: {e}")
             traceback.print_exc()
             raise


        print("DGL 图构建完成。")
        if rel:
             rel_counts = pd.Series(rel).value_counts().sort_index()
             print("关系类型统计 (去重后):")
             for rel_id, count in rel_counts.items():
                  rel_name = [name for name, idx in self.relation2id.items() if idx == rel_id]
                  if rel_name: print(f"- {rel_name[0]} (ID {rel_id}): {count} 条边")
                  else: print(f"- 未知关系 (ID {rel_id}): {count} 条边")

        return self.graph

    def get_triplets(self):
        """获取知识图谱三元组列表 (h_gid, r_id, t_gid)"""
        if self.graph is None: raise ValueError("图尚未构建。")
        if self.graph.num_edges() == 0: return []
        edges = self.graph.edges(); rel_types = self.graph.edata['rel_type']
        return [(edges[0][i].item(), rel_types[i].item(), edges[1][i].item()) for i in range(self.graph.num_edges())]

    def get_num_entities(self): return self.num_entities
    def get_num_relations(self): return self.num_relations
