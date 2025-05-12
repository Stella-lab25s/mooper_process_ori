import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score # 引入 RMSE 和 Accuracy
from collections import defaultdict
import warnings # 处理警告
import pandas as pd
import traceback

class Evaluator:
    """
    包含各种推荐系统评估指标的静态方法的类。
    """
    @staticmethod
    def mae(preds, labels):
        """计算平均绝对误差 (MAE)"""
        if len(preds) == 0 or len(labels) == 0 or len(preds) != len(labels):
            print("DEBUG (MAE): 输入数组为空或长度不匹配。")
            return np.nan
        try:
            preds = np.asarray(preds)
            labels = np.asarray(labels)
            valid_idx = ~np.isnan(preds) & ~np.isnan(labels)
            if np.sum(valid_idx) == 0: return np.nan
            return mean_absolute_error(labels[valid_idx], preds[valid_idx])
        except Exception as e:
            print(f"DEBUG (MAE): 计算时发生错误: {e}")
            return np.nan

    # --- *修改点*: 添加 RMSE 函数 ---
    @staticmethod
    def rmse(preds, labels):
        """计算均方根误差 (RMSE)"""
        if len(preds) == 0 or len(labels) == 0 or len(preds) != len(labels):
            print("DEBUG (RMSE): 输入数组为空或长度不匹配。")
            return np.nan
        try:
            preds = np.asarray(preds)
            labels = np.asarray(labels)
            valid_idx = ~np.isnan(preds) & ~np.isnan(labels)
            if np.sum(valid_idx) == 0: return np.nan
            # 检查是否有无效值
            mse = mean_squared_error(labels[valid_idx], preds[valid_idx])
            if mse < 0:
                print(f"DEBUG (RMSE): Warning - MSE is negative ({mse}). Returning NaN.")
                return np.nan
            return np.sqrt(mse)
        except Exception as e:
            print(f"DEBUG (RMSE): 计算时发生错误: {e}")
            return np.nan

    @staticmethod
    def precision(preds, labels, threshold=70.0): # 假设原始评分是0-100，阈值设为70
        """计算准确率 (Precision/Accuracy)."""
        if len(preds) == 0 or len(labels) == 0 or len(preds) != len(labels):
            print("DEBUG (Precision): 输入数组为空或长度不匹配。")
            return np.nan
        try:
            preds = np.asarray(preds)
            labels = np.asarray(labels)
            valid_idx = ~np.isnan(preds) & ~np.isnan(labels)
            if np.sum(valid_idx) == 0: return np.nan
            pred_classes = (preds[valid_idx] >= threshold).astype(int)
            label_classes = (labels[valid_idx] >= threshold).astype(int)
            return accuracy_score(label_classes, pred_classes)
        except Exception as e:
            print(f"DEBUG (Precision): 计算时发生错误: {e}")
            return np.nan

    # --- *修改点*: 修改 NDCG 函数以处理 relevance ---
    @staticmethod
    def ndcg(preds, labels, k=10, relevance_bins=None):
        """
        计算归一化折损累计收益 (NDCG@k)。
        使用 relevance_bins 将连续标签转换为离散等级。
        """
        warnings.warn("This NDCG implementation calculates a global score...") # 保留警告

        if len(preds) == 0 or len(labels) == 0 or len(preds) != len(labels):
            print(f"DEBUG (NDCG@{k}): 输入数组为空或长度不匹配。")
            return np.nan

        try:
            preds = np.asarray(preds)
            labels = np.asarray(labels) # 假设这里的 labels 是原始评分尺度
            valid_idx = ~np.isnan(preds) & ~np.isnan(labels) # 过滤 NaN
            if np.sum(valid_idx) == 0: return 0.0

            preds = preds[valid_idx]
            labels = labels[valid_idx]

            # --- 将连续标签转换为离散相关性等级 ---
            if relevance_bins is not None and len(relevance_bins) > 1:
                # 使用 pandas.cut 进行分箱，生成从 0 开始的等级
                relevance_grades = pd.cut(labels, bins=relevance_bins, right=False, labels=False, include_lowest=True)
                relevance_grades = np.nan_to_num(relevance_grades, nan=0.0).astype(int) # 将 NaN 视为最低等级 0
                print(f"DEBUG (NDCG@{k}): Labels converted to grades using bins {relevance_bins}. Min grade: {np.min(relevance_grades)}, Max grade: {np.max(relevance_grades)}")
            else:
                # 默认进行二元化处理
                warnings.warn(f"NDCG relevance_bins not provided or invalid. Using binary relevance (label >= threshold is 1, else 0).")
                threshold = 70.0 # 与 precision 阈值一致或单独设置
                relevance_grades = (labels >= threshold).astype(int)
                print(f"DEBUG (NDCG@{k}): Using binary relevance. Threshold={threshold}")
            # ----------------------------------------------------

            actual_k = min(k, len(preds))
            if actual_k == 0: return 0.0

            # 使用转换后的 relevance_grades 进行后续计算
            paired_scores = list(zip(preds, relevance_grades))
            ranked_items = sorted(paired_scores, key=lambda x: x[0], reverse=True)

            # 计算 DCG@k
            dcg = 0.0
            for i in range(actual_k):
                relevance = ranked_items[i][1] # 使用离散等级
                gain = np.power(2.0, relevance) - 1.0 # 标准 DCG 增益
                dcg += gain / np.log2(i + 2.0)

            # 计算 IDCG@k
            ideal_ranked_items = sorted(paired_scores, key=lambda x: x[1], reverse=True) # 按等级排序
            idcg = 0.0
            for i in range(actual_k):
                relevance = ideal_ranked_items[i][1]
                gain = np.power(2.0, relevance) - 1.0
                idcg += gain / np.log2(i + 2.0)

            print(f"DEBUG (NDCG@{k}): Calculated DCG={dcg:.4f}, IDCG={idcg:.4f}")

            if idcg == 0: return 0.0
            else:
                ndcg_score = dcg / idcg
                if np.isnan(ndcg_score):
                     print(f"DEBUG (NDCG@{k}): Calculated NDCG@{k} is NaN.")
                     return 0.0
                return ndcg_score

        except Exception as e:
            print(f"DEBUG (NDCG@{k}): Calculation error: {e}")
            traceback.print_exc()
            return np.nan


    @staticmethod
    def group_satisfaction(preds, labels, groups): # labels 参数保留
        """计算群组满意度 (群组平均预测分的平均值)"""
        if len(groups) == 0 or len(preds) != len(groups):
            print("DEBUG (Satisfaction): 输入数组为空或长度不匹配。")
            return np.nan
        try:
            preds = np.asarray(preds)
            groups = np.asarray(groups)
            valid_idx = ~np.isnan(preds) & ~np.isnan(groups)
            if np.sum(valid_idx) == 0: return np.nan

            preds = preds[valid_idx]
            groups = groups[valid_idx]

            group_preds_map = defaultdict(list)
            for g, p in zip(groups, preds): group_preds_map[g].append(p)

            satisfaction_scores = [np.mean(group_preds_map[g_id]) for g_id in group_preds_map if group_preds_map[g_id]]

            if not satisfaction_scores: print("DEBUG (Satisfaction): 未能计算任何群组的平均分。"); return np.nan

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                satisfaction = np.nanmean(satisfaction_scores)

            if np.isnan(satisfaction): print("DEBUG (Satisfaction): 计算得到的群组满意度为 NaN。"); return np.nan
            return satisfaction
        except Exception as e:
            print(f"DEBUG (Satisfaction): 计算时发生错误: {e}")
            return np.nan

    # --- *修改点*: 修改 evaluate_all ---
    @classmethod
    def evaluate_all(cls, preds, labels, groups, k=10, precision_threshold=70.0, ndcg_bins=None):
        """
        计算所有定义的评估指标.
        Args:
            preds (np.ndarray): 预测评分数组 (原始尺度).
            labels (np.ndarray): 真实评分数组 (原始尺度).
            groups (np.ndarray): 群组 ID 数组.
            k (int): NDCG 的 k 值.
            precision_threshold (float): 准确率计算的阈值 (基于原始尺度).
            ndcg_bins (list, optional): 用于 NDCG 计算的分箱边界 (基于原始尺度).
        Returns:
            dict: 包含所有指标名称和值的字典.
        """
        if len(preds) != len(labels) or len(preds) != len(groups):
            raise ValueError("Length of predictions, labels, and groups must match.")

        if len(preds) == 0:
             warnings.warn("Evaluation data is empty. Returning NaN for all metrics.")
             return {'MAE': np.nan, 'RMSE': np.nan, 'Precision': np.nan, f'NDCG@{k}': np.nan, 'Satisfaction': np.nan}

        # 定义 NDCG 默认分箱规则 (基于原始 0-100 评分)
        if ndcg_bins is None:
             default_ndcg_bins = [0, 60, 70, 80, 90, 101] # [0,60)->0, [60,70)->1, ..., [90,101)->4
             print(f"DEBUG (evaluate_all): Using default NDCG bins: {default_ndcg_bins}")
             ndcg_bins = default_ndcg_bins

        print("DEBUG: 开始计算所有评估指标...")
        mae_score = cls.mae(preds, labels)
        rmse_score = cls.rmse(preds, labels) # 新增
        precision_score = cls.precision(preds, labels, threshold=precision_threshold)
        ndcg_score = cls.ndcg(preds, labels, k=k, relevance_bins=ndcg_bins) # 传递 bins
        satisfaction_score = cls.group_satisfaction(preds, labels, groups) # 保持不变
        print("DEBUG: 评估指标计算完成。")

        return {
            'MAE': mae_score,
            'RMSE': rmse_score, # 新增
            'Precision': precision_score, # 使用调整后的阈值
            f'NDCG@{k}': ndcg_score, # 使用修正后的计算
            'Satisfaction': satisfaction_score
        }
