import warnings
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import dgl
import time
import traceback

class Trainer:
    def __init__(self, model, kg_model, recom_model, num_rels, embed_size, device):
        """
        初始化 Trainer。

        Args:
            model: 包含节点嵌入层的模型 (例如 RGCN, ImprovedRGCN)。主要使用其 self.model.embed。
            kg_model: 知识图谱评分模型 (例如 ConvKB)。
            recom_model: 推荐评分模型。
            num_rels: 图谱中关系的数量。
            embed_size: 嵌入维度。
            device: 计算设备 (cpu or cuda)。
        """
        self.device = device
        self.model = model.to(device)  # GCN/RGCN model for embeddings
        self.kg_model = kg_model.to(device)  # ConvKB
        self.recom_model = recom_model.to(device) # Recommender model
        self.num_rels = num_rels
        self.embed_size = embed_size

        # 关系嵌入层 (KG 训练需要)
        self.rel_embedding = torch.nn.Embedding(num_rels, embed_size).to(device)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)

        # 优化器
        # KG 优化器: 优化节点嵌入、关系嵌入和 ConvKB 参数
        self.kg_optim = optim.Adam(
            list(self.model.parameters()) + # 优化 GCN/RGCN 的所有参数 (包括嵌入和其他层)
            list(self.rel_embedding.parameters()) + # 优化关系嵌入
            list(self.kg_model.parameters()),      # 优化 ConvKB 参数
            lr=0.001, # 可调学习率
            weight_decay=1e-5 # L2 正则化
        )

        # 推荐模型优化器: 仅优化推荐模型参数
        self.recom_optim = optim.Adam(
             self.recom_model.parameters(), # 只优化推荐模型参数
             lr=0.005, # 可调学习率
             weight_decay=1e-5
        )

        # 用于存储训练好的节点嵌入 (在 train_kg 后更新)
        self.node_embeddings = None


    def _negative_sampling_kg(self, pos_triples_tensor, num_entities, num_neg=1):
        """为 KG 训练生成负样本三元组。"""
        num_pos = pos_triples_tensor.shape[0]
        neg_triples_list = []
        pos_triples_set = set(tuple(map(int, t)) for t in pos_triples_tensor.tolist())

        for i in range(num_pos):
            h, r, t = pos_triples_tensor[i].tolist()
            count = 0
            attempts = 0
            while count < num_neg and attempts < num_neg * 10: # 增加尝试次数
                attempts += 1
                if np.random.random() < 0.5: # 替换头
                    neg_h = np.random.randint(0, num_entities)
                    if neg_h != h and (neg_h, r, t) not in pos_triples_set:
                        neg_triples_list.append([neg_h, r, t])
                        count += 1
                else: # 替换尾
                    neg_t = np.random.randint(0, num_entities)
                    if neg_t != t and (h, r, neg_t) not in pos_triples_set:
                        neg_triples_list.append([h, r, neg_t])
                        count += 1
            while count < num_neg and len(neg_triples_list) > i * num_neg:
                 neg_triples_list.append(neg_triples_list[-1]) # 重复最后一个
                 count += 1

        if not neg_triples_list:
             print("DEBUG (Trainer): 警告: 未能生成任何负样本！")
             return torch.empty((0, 3), dtype=torch.long, device=self.device)

        return torch.tensor(neg_triples_list, dtype=torch.long, device=self.device)


    def _calc_kg_loss(self, pos_triples, neg_triples):
        """计算知识图谱的 BPR 损失。"""
        try:
            # 优化嵌入层
            node_embeds = self.model.embed.weight

            # 正样本嵌入
            pos_h_idx, pos_r_idx, pos_t_idx = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
            pos_h_embeds = node_embeds[pos_h_idx]
            pos_t_embeds = node_embeds[pos_t_idx]
            pos_r_embeds = self.rel_embedding(pos_r_idx)
            pos_score = self.kg_model(pos_h_embeds, pos_r_embeds, pos_t_embeds)

            # 负样本嵌入
            neg_h_idx, neg_r_idx, neg_t_idx = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
            neg_h_embeds = node_embeds[neg_h_idx]
            neg_t_embeds = node_embeds[neg_t_idx]
            neg_r_embeds = self.rel_embedding(neg_r_idx)
            neg_score = self.kg_model(neg_h_embeds, neg_r_embeds, neg_t_embeds)

            # BPR 损失
            num_pos = pos_score.shape[0]
            num_neg_total = neg_score.shape[0]
            if num_neg_total == 0: return torch.tensor(0.0, device=self.device, requires_grad=True)

            if num_neg_total % num_pos == 0:
                 num_neg_per_pos = num_neg_total // num_pos
                 pos_score = pos_score.repeat_interleave(num_neg_per_pos, dim=0)
            else:
                 print(f"DEBUG (Trainer): 警告: 负样本数量 ({num_neg_total}) 与正样本数量 ({num_pos}) 不匹配。")
                 min_len = min(pos_score.shape[0], neg_score.shape[0])
                 pos_score = pos_score[:min_len]; neg_score = neg_score[:min_len]

            loss = F.softplus(neg_score - pos_score).mean()
            return loss
        except IndexError as e:
             print(f"\nDEBUG (Trainer): KG Loss 计算时发生索引错误: {e}")
             print("可能原因：负采样生成的实体 ID 超出了嵌入层范围。")
             print(f"嵌入层大小: {node_embeds.shape[0]}")
             print(f"最大正样本索引: H={pos_h_idx.max()}, T={pos_t_idx.max()}")
             print(f"最大负样本索引: H={neg_h_idx.max()}, T={neg_t_idx.max()}")
             raise # 重新抛出错误
        except Exception as e:
             print(f"\nDEBUG (Trainer): KG Loss 计算时发生未知错误: {e}")
             traceback.print_exc()
             raise

    def train_kg(self, graph, num_entities, epochs=100, batch_size=1024): # 添加 batch_size
        """知识图谱训练"""
        print(f"Starting KG training with batch_size={batch_size}...")
        self.model.train()
        self.kg_model.train()
        self.rel_embedding.train()

        if graph.num_edges() == 0:
            print("DEBUG (Trainer): 警告: 图为空，跳过 KG 训练。")
            with torch.no_grad(): self.node_embeddings = self.model.embed.weight.data.clone().detach().to(self.device)
            return

        # 获取所有边 ID
        all_edge_ids = torch.arange(graph.num_edges(), device=self.device) # 确保边 ID 在 GPU 上
        num_batches = (graph.num_edges() + batch_size - 1) // batch_size

        print(f"KG 训练: {epochs} 轮, 总边数: {graph.num_edges()}, 批次数: {num_batches}")
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            # 打乱边 ID 进行批处理
            shuffled_edge_ids = all_edge_ids[torch.randperm(graph.num_edges())]

            for i in range(num_batches):
                batch_start_time = time.time()
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, graph.num_edges())
                batch_edge_ids = shuffled_edge_ids[start_idx:end_idx]

                if len(batch_edge_ids) == 0: continue

                # 获取当前批次的正样本三元组
                # ** 修改点: 确保 graph 在正确的设备上 ** (已在 main.py 中处理)
                batch_src, batch_dst = graph.find_edges(batch_edge_ids) # find_edges 需要 CPU 上的 eid? 不，应该和 graph 同设备
                batch_rel = graph.edata['rel_type'][batch_edge_ids]
                pos_triples_batch = torch.stack([batch_src, batch_rel, batch_dst], dim=1)

                # 负采样
                neg_triples_batch = self._negative_sampling_kg(pos_triples_batch, num_entities, num_neg=1)

                if neg_triples_batch.shape[0] == 0: continue

                # 计算损失
                loss = self._calc_kg_loss(pos_triples_batch, neg_triples_batch)

                # 反向传播和优化
                self.kg_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.kg_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.rel_embedding.parameters(), max_norm=1.0)
                self.kg_optim.step()

                total_loss += loss.item()

                # 打印批次损失 (可选，可能减慢训练)
                # if (i + 1) % 500 == 0: # 每 500 个批次打印一次
                #     batch_time = time.time() - batch_start_time
                #     print(f"KG Epoch {epoch}/{epochs-1}, Batch {i+1}/{num_batches}, Batch Loss: {loss.item():.4f}, Time/Batch: {batch_time:.3f}s")


            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            epoch_time = time.time() - epoch_start_time
            print(f"KG Epoch {epoch}/{epochs-1}, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

        # 训练完成后保存最终嵌入
        with torch.no_grad():
            self.node_embeddings = self.model.embed.weight.data.clone().detach().to(self.device)
            print("KG training finished.")


    def train_recom(self, train_loader, epochs=50):
        """推荐模型训练"""
        print("Starting Recommender training...")
        if self.node_embeddings is None:
             print("DEBUG (Trainer): 错误: KG 嵌入尚未训练。请先运行 train_kg。")
             # 使用初始嵌入作为备选
             warnings.warn("Using initial embeddings for recommendation training.")
             node_embeds_fixed = self.model.embed.weight.clone().detach().to(self.device)
        else:
             node_embeds_fixed = self.node_embeddings.clone().detach().to(self.device)

        self.recom_model.train() # 设置推荐模型为训练模式

        print(f"Recommender 训练: {epochs} 轮, 训练数据批次数: {len(train_loader)}")
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            batch_count = 0
            for batch in train_loader:
                # 假设 DataLoader 输出的是内部/全局 ID
                users_idx, items_idx, groups_idx, ratings = [b.to(self.device) for b in batch]

                try:
                    # 使用固定的 KG 嵌入获取向量
                    user_emb = node_embeds_fixed[users_idx]
                    item_emb = node_embeds_fixed[items_idx]
                    group_emb = node_embeds_fixed[groups_idx]
                except IndexError as e:
                     print(f"\nDEBUG (Trainer): Recom training IndexError: {e}")
                     print(f"Embeddings shape: {node_embeds_fixed.shape}")
                     print(f"Max User Idx: {users_idx.max().item() if len(users_idx)>0 else 'N/A'}")
                     print(f"Max Item Idx: {items_idx.max().item() if len(items_idx)>0 else 'N/A'}")
                     print(f"Max Group Idx: {groups_idx.max().item() if len(groups_idx)>0 else 'N/A'}")
                     continue # 跳过这个 batch

                # 预测
                preds = self.recom_model(user_emb, item_emb, group_emb)
                # 计算损失 (例如 MSE)
                loss = F.mse_loss(preds.squeeze(), ratings.float()) # 确保 ratings 是 float

                # 反向传播和优化 (只更新推荐模型参数)
                self.recom_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.recom_model.parameters(), max_norm=1.0) # 可选
                self.recom_optim.step()

                total_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                 avg_loss = total_loss / batch_count
                 epoch_time = time.time() - epoch_start_time
                 print(f"Recom Epoch {epoch}/{epochs-1}, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
            else:
                 print(f"Recom Epoch {epoch}: No batches processed.")

        print("Recommender training finished.")

    # --- *修改点*: 添加 predict 方法 ---
    def predict(self, test_loader):
        """使用训练好的模型进行预测"""
        print("DEBUG (Trainer): Starting prediction...")
        if self.node_embeddings is None:
             print("DEBUG (Trainer): 错误: 节点嵌入未训练或不可用。")
             return torch.tensor([]), torch.tensor([]) # 返回空张量
        if self.recom_model is None:
             print("DEBUG (Trainer): 错误: 推荐模型未初始化。")
             return torch.tensor([]), torch.tensor([])

        self.recom_model.eval() # 设置推荐模型为评估模式
        # 使用最终的 KG 嵌入进行预测
        node_embeds_fixed = self.node_embeddings.clone().detach().to(self.device)

        all_preds = []
        all_labels = []

        print(f"DEBUG (Trainer): Prediction - Test data batches: {len(test_loader)}")
        with torch.no_grad(): # 预测时不需要计算梯度
            for batch_idx, batch in enumerate(test_loader):
                # 假设 DataLoader 输出的是内部/全局 ID
                users_idx, items_idx, groups_idx, labels = [b.to(self.device) for b in batch]

                try:
                    # 获取嵌入
                    user_emb = node_embeds_fixed[users_idx]
                    item_emb = node_embeds_fixed[items_idx]
                    group_emb = node_embeds_fixed[groups_idx]
                except IndexError as e:
                     print(f"\nDEBUG (Trainer): Prediction IndexError at batch {batch_idx}: {e}")
                     print(f"Embeddings shape: {node_embeds_fixed.shape}")
                     print(f"Max User Idx: {users_idx.max().item() if len(users_idx)>0 else 'N/A'}")
                     print(f"Max Item Idx: {items_idx.max().item() if len(items_idx)>0 else 'N/A'}")
                     print(f"Max Group Idx: {groups_idx.max().item() if len(groups_idx)>0 else 'N/A'}")
                     continue # 跳过这个 batch

                # 预测
                preds = self.recom_model(user_emb, item_emb, group_emb)

                all_preds.append(preds.squeeze()) # 移除维度为1的维度
                all_labels.append(labels) # 收集标签

        print("DEBUG (Trainer): Prediction loop finished.")
        if not all_preds: # 如果列表为空
             print("DEBUG (Trainer): No predictions were generated.")
             return torch.tensor([]), torch.tensor([])

        # 将列表中的所有批次张量连接成一个大张量
        try:
            # .cpu() 将张量移回 CPU 以便后续处理（例如转换为 NumPy）
            final_preds = torch.cat(all_preds).cpu()
            final_labels = torch.cat(all_labels).cpu()
        except RuntimeError as e:
            print(f"DEBUG (Trainer): Error concatenating predictions/labels: {e}")
            # 尝试过滤掉空张量
            all_preds = [p for p in all_preds if p.numel() > 0]
            all_labels = [l for l in all_labels if l.numel() > 0]
            if not all_preds: return torch.tensor([]), torch.tensor([])
            final_preds = torch.cat(all_preds).cpu()
            final_labels = torch.cat(all_labels).cpu()

        print(f"DEBUG (Trainer): Final predictions shape: {final_preds.shape}, Labels shape: {final_labels.shape}")
        return final_preds, final_labels
