import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import warnings
import numpy as np
import traceback
import os
import sys
import time

# --- 从项目中导入 ---
try:
    print("DEBUG: 正在导入模块...")
    from data_preprocessing import DataProcessor
    from knowledge_graph import KnowledgeGraphBuilder
    from models.gcn import RGCN, ImprovedRGCN
    from models.convkb import ConvKB
    from models.recommender import GroupRecommender, ImprovedGroupRecommender
    from train import Trainer
    from evaluate import Evaluator

    try:
        from baselines import PopularityRecommender, SVDRecommender

        baselines_available = True
        print("DEBUG: baselines.py 导入成功。")
    except ImportError:
        warnings.warn("baselines.py not found or contains errors. Skipping baseline comparison.")
        baselines_available = False
    from experiments import Experiment  

    print("DEBUG: 模块导入成功。")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


# --- Helper function to prepare DataLoaders ---
def prepare_loaders(train_data_with_target, test_data_with_target, target_column, batch_size=256,
                    user_col='user_internal_id', item_col='item_internal_id', group_col='group_internal_id'):
    print("DEBUG: 进入 prepare_loaders 函数...")
    try:
        required_cols = [user_col, item_col, group_col, target_column]
        if not all(col in train_data_with_target.columns for col in required_cols):
            raise KeyError(f"训练数据缺少必需列: {required_cols}, 实际包含: {train_data_with_target.columns.tolist()}")
        if not all(col in test_data_with_target.columns for col in required_cols):
            raise KeyError(f"测试数据缺少必需列: {required_cols}, 实际包含: {test_data_with_target.columns.tolist()}")

        train_data_with_target = train_data_with_target.astype(
            {user_col: int, item_col: int, group_col: int, target_column: float})
        test_data_with_target = test_data_with_target.astype(
            {user_col: int, item_col: int, group_col: int, target_column: float})

        if train_data_with_target[target_column].isnull().any():
            warnings.warn(f"训练数据的目标列 '{target_column}' 包含 NaN 值，将填充为 0。")
            train_data_with_target[target_column].fillna(0.0, inplace=True)
        if test_data_with_target[target_column].isnull().any():
            warnings.warn(f"测试数据的目标列 '{target_column}' 包含 NaN 值，将填充为 0。")
            test_data_with_target[target_column].fillna(0.0, inplace=True)

        train_users_indices = torch.LongTensor(train_data_with_target[user_col].values)
        train_items_indices = torch.LongTensor(train_data_with_target[item_col].values)
        train_groups_indices = torch.LongTensor(train_data_with_target[group_col].values)
        train_targets = torch.FloatTensor(train_data_with_target[target_column].values)

        train_dataset = TensorDataset(train_users_indices, train_items_indices, train_groups_indices, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"DEBUG: 训练 DataLoader 创建，大小: {len(train_loader.dataset)}")

        test_users_indices = torch.LongTensor(test_data_with_target[user_col].values)
        test_items_indices = torch.LongTensor(test_data_with_target[item_col].values)
        test_groups_indices = torch.LongTensor(test_data_with_target[group_col].values)
        test_targets = torch.FloatTensor(test_data_with_target[target_column].values)

        test_dataset = TensorDataset(test_users_indices, test_items_indices, test_groups_indices, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"DEBUG: 测试 DataLoader 创建，大小: {len(test_loader.dataset)}")

        print("DEBUG: 数据加载器准备完成。")
        return train_loader, test_loader
    except KeyError as e:
        print(f"数据转换 DataLoader 时出错: 缺少列 {e}。"); sys.exit(1)
    except Exception as e:
        print(f"准备数据加载器时发生意外错误: {e}"); traceback.print_exc(); sys.exit(1)


class BaselineModel:  # 备用基线
    def __init__(self, name="基线模型"):
        self.name = name; self.user_avg = {}; self.item_avg = {}; self.global_avg = 0; self.user_encoder = None; self.item_encoder = None

    def train(self, train_data_mapped, user_encoder=None, item_encoder=None):
        print(f"DEBUG: 训练 {self.name}...");
        self.user_encoder = user_encoder;
        self.item_encoder = item_encoder
        try:
            # 基线模型使用 'rating', 'user_id_mapped', 'item_id_mapped'
            self.global_avg = train_data_mapped['rating'].mean()
            user_ratings = train_data_mapped.groupby('user_id_mapped')['rating'].mean();
            self.user_avg = user_ratings.to_dict()
            item_ratings = train_data_mapped.groupby('item_id_mapped')['rating'].mean();
            self.item_avg = item_ratings.to_dict()
            print(
                f"DEBUG: {self.name} 训练完成。全局平均分: {self.global_avg:.4f if not pd.isna(self.global_avg) else 'N/A'}")
        except KeyError as e:
            print(f"DEBUG: 训练 {self.name} 时出错: {e}"); self.global_avg = 3.0; print(
                f"DEBUG: 警告: {self.name} 训练数据不完整...")

    def predict(self, test_data_orig_ids):  # test_data_orig_ids 应该有 'user_id', 'item_id'
        print(f"DEBUG: 评估 {self.name}...");
        preds = []
        if self.user_encoder is None or self.item_encoder is None: return np.full(len(test_data_orig_ids),
                                                                                  self.global_avg if not pd.isna(
                                                                                      self.global_avg) else 3.0)
        for _, row in test_data_orig_ids.iterrows():
            user_orig_id = row.get('user_id');
            item_orig_id = row.get('item_id');
            user_mapped_id = None;
            item_mapped_id = None  # 基线假设列名为'user_id', 'item_id'
            try:
                if user_orig_id is not None and user_orig_id in self.user_encoder.classes_: user_mapped_id = \
                self.user_encoder.transform([user_orig_id])[0]
                if item_orig_id is not None and item_orig_id in self.item_encoder.classes_: item_mapped_id = \
                self.item_encoder.transform([item_orig_id])[0]
            except Exception:
                pass
            user_avg_rating = self.user_avg.get(user_mapped_id);
            item_avg_rating = self.item_avg.get(item_mapped_id)
            if user_avg_rating is not None and item_avg_rating is not None:
                pred = (user_avg_rating + item_avg_rating) / 2
            elif user_avg_rating is not None:
                pred = user_avg_rating
            elif item_avg_rating is not None:
                pred = item_avg_rating
            else:
                pred = self.global_avg if not pd.isna(self.global_avg) else 3.0
            preds.append(pred)
        print(f"DEBUG: {self.name} 评估完成.");
        return np.array(preds)


class GroupBasedModel:  # 备用基线
    def __init__(self, name="群组模型"):
        self.name = name; self.group_item_ratings = {}; self.group_avg = {}; self.item_avg = {}; self.global_avg = 0; self.item_encoder = None

    def train(self, train_data_mapped, user_encoder=None, item_encoder=None):
        print(f"DEBUG: 训练 {self.name}...");
        self.item_encoder = item_encoder
        try:
            # 基线模型使用 'rating', 'group_id', 'item_id_mapped'
            self.global_avg = train_data_mapped['rating'].mean()
            group_ratings = train_data_mapped.groupby('group_id')['rating'].mean();
            self.group_avg = group_ratings.to_dict()
            item_ratings = train_data_mapped.groupby('item_id_mapped')['rating'].mean();
            self.item_avg = item_ratings.to_dict()
            group_item = train_data_mapped.groupby(['group_id', 'item_id_mapped'])['rating'].mean();
            self.group_item_ratings = group_item.to_dict()
            print(f"DEBUG: {self.name} 训练完成。")
        except KeyError as e:
            print(f"DEBUG: 训练 {self.name} 时出错: {e}"); self.global_avg = 3.0; print(
                f"DEBUG: 警告: {self.name} 训练数据不完整...")

    def predict(self, test_data_orig_ids):  # test_data_orig_ids 应该有 'group_id', 'item_id'
        print(f"DEBUG: 评估 {self.name}...");
        preds = []
        if self.item_encoder is None: return np.full(len(test_data_orig_ids),
                                                     self.global_avg if not pd.isna(self.global_avg) else 3.0)
        for _, row in test_data_orig_ids.iterrows():
            group_id = row.get('group_id');
            item_orig_id = row.get('item_id');
            item_mapped_id = None  # 基线假设列名为 'item_id'
            try:
                if item_orig_id is not None and item_orig_id in self.item_encoder.classes_: item_mapped_id = \
                self.item_encoder.transform([item_orig_id])[0]
            except Exception:
                pass
            group_item_key = (group_id, item_mapped_id)
            if group_item_key in self.group_item_ratings:
                pred = self.group_item_ratings[group_item_key]
            elif group_id in self.group_avg:
                pred = self.group_avg[group_id]
            elif item_mapped_id in self.item_avg:
                pred = self.item_avg[item_mapped_id]
            else:
                pred = self.global_avg if not pd.isna(self.global_avg) else 3.0
            preds.append(pred)
        print(f"DEBUG: {self.name} 评估完成.");
        return np.array(preds)


def main():
    print("DEBUG: ================== 程序开始 ==================")
    start_time_main = time.time()
    DATA_DIR = 'data'
    DATA_FILE = os.path.join(DATA_DIR, 'MOOPer/interaction/challenge_interaction.csv')
    USER_COLUMN_ORIG = 'user_id'
    ITEM_COLUMN_ORIG = 'challenge_id'
    SCORE_COLUMN_ORIG = 'final_score'
    EXPECTED_COLUMNS = [USER_COLUMN_ORIG, ITEM_COLUMN_ORIG, SCORE_COLUMN_ORIG]
    N_GROUPS = 10;
    EMBED_SIZE = 64;
    KG_EPOCHS = 5;
    RECOM_EPOCHS = 5
    BATCH_SIZE = 256;
    TEST_SIZE = 0.2;
    RANDOM_STATE = 42
    RESULTS_DIR = 'results';
    RESULTS_PLOT_FILE = os.path.join(RESULTS_DIR, 'experiment_results_final.png')

    NORMALIZE_SCORE = True
    MIN_SCORE_FOR_NORM = 0.0
    MAX_SCORE_FOR_NORM = 100.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    os.makedirs(DATA_DIR, exist_ok=True);
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nDEBUG: === 1. 数据加载与检查 ===")
    if not os.path.exists(DATA_FILE):
        print(f"错误: 未找到数据文件 '{DATA_FILE}'。"); sys.exit(1)
    else:
        print(f"从 '{DATA_FILE}' 加载数据...")
        try:
            input_df = pd.read_csv(DATA_FILE)
            if not all(col in input_df.columns for col in EXPECTED_COLUMNS):
                print(
                    f"\n错误: '{DATA_FILE}' 文件缺少必需的原始列。期望列: {EXPECTED_COLUMNS}, 找到列: {input_df.columns.tolist()}");
                sys.exit(1)
            if SCORE_COLUMN_ORIG not in input_df.columns: print(
                f"\n错误: '{DATA_FILE}' 文件缺少指定的评分列 '{SCORE_COLUMN_ORIG}'"); sys.exit(1)

            input_df[SCORE_COLUMN_ORIG] = pd.to_numeric(input_df[SCORE_COLUMN_ORIG], errors='coerce')
            initial_len = len(input_df);
            input_df.dropna(subset=[SCORE_COLUMN_ORIG], inplace=True)
            if len(input_df) < initial_len: print(f"DEBUG: 删除了 {initial_len - len(input_df)} 行无效评分记录。")
            print("数据文件格式检查通过。")

            if NORMALIZE_SCORE:
                actual_min_score = input_df[SCORE_COLUMN_ORIG].min()
                actual_max_score = input_df[SCORE_COLUMN_ORIG].max()
                print(f"DEBUG: 原始评分 '{SCORE_COLUMN_ORIG}' 范围: [{actual_min_score}, {actual_max_score}]")
                if pd.isna(actual_min_score) or pd.isna(actual_max_score) or actual_min_score == actual_max_score:
                    warnings.warn(f"无法从数据中获取有效的最小/最大评分值或范围为零。将禁用归一化。")
                    NORMALIZE_SCORE = False
                else:
                    MIN_SCORE_FOR_NORM = actual_min_score
                    MAX_SCORE_FOR_NORM = actual_max_score
                print(f"DEBUG: 用于归一化的 MIN_SCORE: {MIN_SCORE_FOR_NORM}, MAX_SCORE: {MAX_SCORE_FOR_NORM}")

        except Exception as e:
            print(f"加载或检查 '{DATA_FILE}' 时出错: {e}"); sys.exit(1)
    print(f"原始数据加载完成，共 {len(input_df)} 条有效评分记录。")

    print("\nDEBUG: === 2. 数据预处理 ===")
    processor = DataProcessor(input_df, user_col=USER_COLUMN_ORIG, item_col=ITEM_COLUMN_ORIG,
                              rating_col=SCORE_COLUMN_ORIG)
    try:
        processor.preprocess().cluster_users(n_groups=N_GROUPS, random_state=RANDOM_STATE)
        train_df_proc, test_df_proc = processor.split_data(test_size=TEST_SIZE, random_state=RANDOM_STATE)
        user_encoder, item_encoder = processor.get_mappings()
        processed_data_full = processor.get_processed_data()
        print(f"DEBUG: 数据预处理完成。")
        if train_df_proc.empty or test_df_proc.empty: sys.exit("错误：预处理或划分后训练/测试集为空！")
    except Exception as e:
        print(f"数据预处理失败: {e}"); traceback.print_exc(); sys.exit(1)

    print("\nDEBUG: === 3. 构建知识图谱 ===")
    kg_builder = KnowledgeGraphBuilder(processed_data_full, user_col='user_id_mapped', item_col='item_id_mapped',
                                       group_col='group_id', rating_col='rating')
    try:
        kg_builder.build_mappings()
        num_entities = kg_builder.get_num_entities();
        num_rels = kg_builder.get_num_relations()
        print(f"DEBUG: 实体数量 (全局): {num_entities}, 关系数量: {num_rels}")
        if num_entities <= 0: sys.exit("错误: 实体数量为 0。")
        graph = kg_builder.build_graph();
        graph = graph.to(device);
        print(f"DEBUG: 图已移动到 {graph.device}")
        print(f"DEBUG: 知识图谱构建完成。")
    except Exception as e:
        print(f"构建知识图谱时发生意外错误: {e}"); traceback.print_exc(); sys.exit(1)

    print("\nDEBUG: === 4. 准备 DataLoader 数据 ===")
    target_col_for_loader = 'rating'

    train_df_for_loader = train_df_proc.copy()
    test_df_for_loader = test_df_proc.copy()

    if NORMALIZE_SCORE:
        print(f"DEBUG: 对评分列 'rating' 进行 Min-Max 归一化到 [0, 1]...")
        target_col_for_loader = 'rating_normalized'
        if (MAX_SCORE_FOR_NORM - MIN_SCORE_FOR_NORM) == 0:
            warnings.warn("MAX_SCORE_FOR_NORM 和 MIN_SCORE_FOR_NORM 相等，归一化结果将为0或NaN。跳过归一化。")
            target_col_for_loader = 'rating'
            NORMALIZE_SCORE = False
        else:
            train_df_for_loader[target_col_for_loader] = (train_df_for_loader['rating'] - MIN_SCORE_FOR_NORM) / (
                        MAX_SCORE_FOR_NORM - MIN_SCORE_FOR_NORM)
            test_df_for_loader[target_col_for_loader] = (test_df_for_loader['rating'] - MIN_SCORE_FOR_NORM) / (
                        MAX_SCORE_FOR_NORM - MIN_SCORE_FOR_NORM)
            train_df_for_loader[target_col_for_loader] = train_df_for_loader[target_col_for_loader].clip(0, 1)
            test_df_for_loader[target_col_for_loader] = test_df_for_loader[target_col_for_loader].clip(0, 1)
            print(
                f"DEBUG: 归一化后的训练集评分范围: [{train_df_for_loader[target_col_for_loader].min():.4f}, {train_df_for_loader[target_col_for_loader].max():.4f}]")
            print(
                f"DEBUG: 归一化后的测试集评分范围: [{test_df_for_loader[target_col_for_loader].min():.4f}, {test_df_for_loader[target_col_for_loader].max():.4f}]")

    train_df_for_loader['user_internal_id'] = train_df_for_loader['user_id_mapped'].apply(
        lambda x: kg_builder._get_entity_id('user', x))
    train_df_for_loader['item_internal_id'] = train_df_for_loader['item_id_mapped'].apply(
        lambda x: kg_builder._get_entity_id('item', x))
    train_df_for_loader['group_internal_id'] = train_df_for_loader['group_id'].apply(
        lambda x: kg_builder._get_entity_id('group', x))
    train_df_for_loader.dropna(subset=['user_internal_id', 'item_internal_id', 'group_internal_id'], inplace=True)
    train_df_for_loader = train_df_for_loader.astype(
        {'user_internal_id': int, 'item_internal_id': int, 'group_internal_id': int})

    test_df_for_loader['user_internal_id'] = test_df_for_loader['user_id_mapped'].apply(
        lambda x: kg_builder._get_entity_id('user', x))
    test_df_for_loader['item_internal_id'] = test_df_for_loader['item_id_mapped'].apply(
        lambda x: kg_builder._get_entity_id('item', x))
    test_df_for_loader['group_internal_id'] = test_df_for_loader['group_id'].apply(
        lambda x: kg_builder._get_entity_id('group', x))
    test_df_for_loader.dropna(subset=['user_internal_id', 'item_internal_id', 'group_internal_id'], inplace=True)
    test_df_for_loader = test_df_for_loader.astype(
        {'user_internal_id': int, 'item_internal_id': int, 'group_internal_id': int})

    if train_df_for_loader.empty or test_df_for_loader.empty: sys.exit("错误: 准备 DataLoader 数据后，训练/测试集为空！")

    train_loader, test_loader = prepare_loaders(
        train_df_for_loader, test_df_for_loader, target_column=target_col_for_loader, batch_size=BATCH_SIZE,
        user_col='user_internal_id', item_col='item_internal_id', group_col='group_internal_id'
    )

    print("\nDEBUG: === 5. 初始化模型 ===")
    kg_embedding_model = ImprovedRGCN(num_entities, EMBED_SIZE, num_rels, num_layers=2).to(device)
    convkb = ConvKB(EMBED_SIZE).to(device)
    recommender = ImprovedGroupRecommender(EMBED_SIZE).to(device)
    print(f"DEBUG: 使用推荐模型: {recommender.__class__.__name__}")
    print("DEBUG: 模型初始化完成。")

    print("\nDEBUG: === 6. 训练模型 ===")
    trainer = Trainer(model=kg_embedding_model, kg_model=convkb, recom_model=recommender, num_rels=num_rels,
                      embed_size=EMBED_SIZE, device=device)
    trainer.train_kg(graph, num_entities, epochs=KG_EPOCHS)
    trainer.train_recom(train_loader, epochs=RECOM_EPOCHS)
    print("DEBUG: 模型训练完成。")

    print("\nDEBUG: === 7. 评估 GCN+KG 模型 ===")
    preds_normalized, labels_normalized_from_loader = trainer.predict(test_loader)
    if preds_normalized.nelement() == 0:
        print("错误：GCN+KG 模型预测结果为空！");
        our_model_results_dict = {'MAE': np.nan, 'RMSE': np.nan, 'NDCG@10': np.nan, 'Satisfaction': np.nan}
    else:
        if NORMALIZE_SCORE:
            print("DEBUG: 将预测反归一化回原始评分尺度...")
            # --- *修改点*: 使用 Min-Max 反归一化 ---
            preds_original_scale = preds_normalized.cpu().numpy() * (
                        MAX_SCORE_FOR_NORM - MIN_SCORE_FOR_NORM) + MIN_SCORE_FOR_NORM
            labels_original_scale = test_df_for_loader['rating'].values[:len(preds_original_scale)]  # 使用原始 rating 列
            print(f"DEBUG: 反归一化后 Preds 范围: [{preds_original_scale.min():.2f}, {preds_original_scale.max():.2f}]")
            print(f"DEBUG: 原始标签范围: [{labels_original_scale.min():.2f}, {labels_original_scale.max():.2f}]")
            # ------------------------------------
        else:
            preds_original_scale = preds_normalized.cpu().numpy();
            labels_original_scale = labels_normalized_from_loader.cpu().numpy()

        groups_internal = test_df_for_loader['group_internal_id'].values[:len(preds_original_scale)]
        print("DEBUG: 使用原始评分尺度计算评估指标...")
        our_model_results_dict = Evaluator.evaluate_all(preds_original_scale, labels_original_scale, groups_internal)
        print("GCN+KG 模型评估结果 (基于原始评分尺度):")
        for metric, value in our_model_results_dict.items(): print(f"- {metric}: {value:.4f}")

    print("\nDEBUG: === 8. 对比实验 ===")
    if baselines_available:
        if hasattr(user_encoder, 'classes_') and hasattr(item_encoder, 'classes_'):  # 确保编码器可用
            # 1. 为基线模型准备训练数据:
            train_data_for_baselines = train_df_proc.rename(columns={
                processor.internal_user_col: 'user_id',  # 确保是 'user_id'
                processor.internal_item_col: 'item_id',  # 确保是 'item_id'
                processor.internal_rating_col: 'rating'  # 确保是 'rating'
            })
            # 确保 train_data_for_baselines 也有 'user_id_mapped', 'item_id_mapped' (如果基线模型需要)
            if 'user_id_mapped' not in train_data_for_baselines.columns and 'user_id' in train_data_for_baselines.columns:
                train_data_for_baselines['user_id_mapped'] = user_encoder.transform(train_data_for_baselines['user_id'])
            if 'item_id_mapped' not in train_data_for_baselines.columns and 'item_id' in train_data_for_baselines.columns:
                train_data_for_baselines['item_id_mapped'] = item_encoder.transform(train_data_for_baselines['item_id'])

            # 2. 为基线模型准备测试数据: 包含原始ID、原始评分、group_id，并重命名为标准列名
            temp_group_info = processed_data_full[[USER_COLUMN_ORIG, 'group_id']].drop_duplicates(
                subset=[USER_COLUMN_ORIG])
            input_df_with_groups = pd.merge(input_df, temp_group_info, on=USER_COLUMN_ORIG, how='left')
            # ---  修复 Pandas FutureWarning ---
            input_df_with_groups['group_id'] = input_df_with_groups['group_id'].fillna(-1)
            input_df_with_groups['group_id'] = input_df_with_groups['group_id'].astype(int)

            # 从 input_df_with_groups (包含原始列名和group_id) 中划分测试集
            _, test_df_orig_for_exp_raw_cols = train_test_split(
                input_df_with_groups,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )
            # --- *修改点*: 将测试集的列名也重命名为基线模型期望的 'user_id', 'item_id', 'rating' ---
            test_df_orig_ids_for_exp = test_df_orig_for_exp_raw_cols.rename(columns={
                USER_COLUMN_ORIG: 'user_id',
                ITEM_COLUMN_ORIG: 'item_id',
                SCORE_COLUMN_ORIG: 'rating'
            })
            # ------------------------------------------------------------------------------------

            exp = Experiment(train_data_for_baselines, test_df_orig_ids_for_exp)

            class OurModelWrapper:
                def __init__(self, trained_trainer, test_loader_for_predict, normalize_output=False, min_score=0.0,
                             max_score=1.0):
                    self.name = "Our GCN+KG";
                    self.trainer = trained_trainer;
                    self.test_loader_internal_ids = test_loader_for_predict
                    self.normalize_output = normalize_output;
                    self.min_score = min_score;
                    self.max_score = max_score;
                    self._trained = True

                def train(self, train_data, user_encoder=None, item_encoder=None):
                    print(f"DEBUG: 模型 '{self.name}' 已训练，跳过。")

                def predict(self, test_data_orig_ids):  # test_data_orig_ids 有原始ID
                    print(f"DEBUG: 评估 {self.name}...");
                    preds_tensor, _ = self.trainer.predict(self.test_loader_internal_ids)
                    preds_np = preds_tensor.cpu().numpy()
                    if self.normalize_output:
                        print(f"DEBUG: {self.name} 反归一化预测 (Min-Max)...")
                        preds_np = preds_np * (self.max_score - self.min_score) + self.min_score  # Min-Max 反归一化
                    print(f"DEBUG: {self.name} 评估完成。")
                    if len(preds_np) != len(test_data_orig_ids):
                        print(
                            f"DEBUG: 警告! {self.name} 预测数量 ({len(preds_np)}) 与原始测试数据 ({len(test_data_orig_ids)}) 不匹配!")
                        min_len = min(len(preds_np), len(test_data_orig_ids));
                        return preds_np[:min_len]
                    return preds_np

            exp.add_model('Our GCN+KG', OurModelWrapper(trainer, test_loader, NORMALIZE_SCORE, MIN_SCORE_FOR_NORM,
                                                        MAX_SCORE_FOR_NORM))
            # --- *修改点*: 基线模型初始化时使用它们期望的标准列名 'user_id', 'item_id', 'rating' ---
            exp.add_model('Popularity', PopularityRecommender(item_col='item_id', rating_col='rating'))
            exp.add_model('SVD',
                          SVDRecommender(user_col='user_id', item_col='item_id', rating_col='rating', n_factors=50))

            exp.run(user_encoder=user_encoder, item_encoder=item_encoder)

            print("\nDEBUG: === 9. 生成对比报告 ===")
            results_df, fig = exp.report(plot=True)
            if not results_df.empty:
                print("\n对比实验结果:");
                print(results_df)
                if fig:
                    try:
                        import matplotlib; exp.save_plot(RESULTS_PLOT_FILE)
                    except ImportError:
                        print("\nMatplotlib not installed. Cannot save plot.")
                    except Exception as e:
                        print(f"保存图表时出错: {e}")
                else:
                    print("未能生成图表对象，无法保存。")
            else:
                print("未能生成有效的对比实验结果。")
        else:
            print("DEBUG: 错误 - User encoder 或 Item encoder 未正确初始化或基线模型不可用。跳过对比实验。")  # 更明确的错误
    else:
        print("\nSkipping baseline comparison as baselines.py was not found or had errors.")
    end_time_main = time.time()
    print(f"\nDEBUG: ================== 程序结束 (总耗时: {end_time_main - start_time_main:.2f} 秒) ==================")


if __name__ == "__main__":
    main()
