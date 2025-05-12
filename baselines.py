import warnings
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd # 导入 randomized_svd
from collections import defaultdict

class PopularityRecommender:
    """
    基于流行度（平均评分）的推荐器。
    """
    def __init__(self, item_col='item_id', rating_col='rating'):
        self.item_col = item_col
        self.rating_col = rating_col
        self.item_avg_ratings = None
        self.global_avg_rating = 0.0
        print(f"DEBUG (Popularity): Initialized with item_col='{item_col}', rating_col='{rating_col}'")



    def train(self, train_data: pd.DataFrame, user_encoder=None, item_encoder=None): # 添加 user_encoder, item_encoder 以匹配接口
        """
        计算训练数据中每个物品的平均评分和全局平均评分。
        Args:
            train_data (pd.DataFrame): 包含原始物品 ID 和评分的训练数据。
                                       注意：此模型直接使用 train_data 中的原始列名。
            user_encoder: 未使用。
            item_encoder: 未使用。
        """
        print(f"DEBUG (Popularity): Training Popularity Recommender...")
        if not train_data.empty:
            # 确保使用的是正确的列名
            if self.item_col not in train_data.columns:
                raise KeyError(f"PopularityRecommender: item_col '{self.item_col}' not found in train_data columns: {train_data.columns.tolist()}")
            if self.rating_col not in train_data.columns:
                raise KeyError(f"PopularityRecommender: rating_col '{self.rating_col}' not found in train_data columns: {train_data.columns.tolist()}")

            self.item_avg_ratings = train_data.groupby(self.item_col)[self.rating_col].mean()
            self.global_avg_rating = train_data[self.rating_col].mean()
            print(f"DEBUG (Popularity): Found average ratings for {len(self.item_avg_ratings)} items.")
        else:
            warnings.warn("Training data is empty for PopularityRecommender. Predictions will use global average (0.0).")
            self.item_avg_ratings = pd.Series(dtype=float) # 创建一个空的 Series
            self.global_avg_rating = 0.0
        print("DEBUG (Popularity): Popularity Recommender training complete.")


    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        为测试数据中的每个交互预测评分。
        Args:
            test_data (pd.DataFrame): 包含原始物品 ID 的测试数据。
        Returns:
            np.ndarray: 预测评分数组。
        """
        print("DEBUG (Popularity): Predicting with Popularity Recommender...")
        if self.item_avg_ratings is None:
            # 如果模型未训练，则发出警告并预测全局平均值（如果已计算）
            warnings.warn("PopularityRecommender not fitted yet or training data was empty. Predicting global average.")
            return np.full(len(test_data), self.global_avg_rating if not pd.isna(self.global_avg_rating) else 0.0)

        if self.item_col not in test_data.columns:
            raise KeyError(f"PopularityRecommender: item_col '{self.item_col}' not found in test_data columns: {test_data.columns.tolist()}")

        predictions = test_data[self.item_col].map(self.item_avg_ratings).fillna(self.global_avg_rating)
        print("DEBUG (Popularity): Popularity predictions generated.")
        return predictions.values

class SVDRecommender:
    """
    基于 SVD (奇异值分解) 的矩阵分解推荐器。
    """
    def __init__(self, user_col='user_id', item_col='item_id', rating_col='rating', n_factors=50, random_state=42):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.n_factors = n_factors
        self.random_state = random_state

        self.user_map = None
        self.item_map = None
        self.user_factors = None
        self.item_factors = None # V^T
        self.global_avg = 0.0
        print(f"DEBUG (SVD): Initialized with user_col='{user_col}', item_col='{item_col}', rating_col='{rating_col}', n_factors={n_factors}")



    def train(self, train_data: pd.DataFrame, user_encoder=None, item_encoder=None): # 添加 user_encoder, item_encoder 以匹配接口
        """
        训练 SVD 模型。
        Args:
            train_data (pd.DataFrame): 包含原始用户/物品 ID 和评分的训练数据。
                                       注意：此模型直接使用 train_data 中的原始列名。
            user_encoder: 未使用。
            item_encoder: 未使用。
        """
        print(f"DEBUG (SVD): Training SVD Recommender (n_factors={self.n_factors})...")
        if train_data.empty:
             warnings.warn("Training data is empty for SVDRecommender. Model cannot be trained.")
             return

        # 确保使用的是正确的列名
        required_cols_svd = [self.user_col, self.item_col, self.rating_col]
        if not all(col in train_data.columns for col in required_cols_svd):
            raise KeyError(f"SVDRecommender: Missing one of {required_cols_svd} in train_data columns: {train_data.columns.tolist()}")


        self.global_avg = train_data[self.rating_col].mean()

        try:
            print("DEBUG (SVD): Creating user-item matrix for SVD...")
            user_item_matrix = pd.pivot_table(
                train_data,
                index=self.user_col,
                columns=self.item_col,
                values=self.rating_col
            )
            print(f"DEBUG (SVD): User-item matrix shape: {user_item_matrix.shape}")

            self.user_map = {uid: i for i, uid in enumerate(user_item_matrix.index)}
            self.item_map = {iid: j for j, iid in enumerate(user_item_matrix.columns)}

            matrix_filled = user_item_matrix.fillna(self.global_avg).values

            print("DEBUG (SVD): Performing SVD...")
            # randomized_svd 通常更快且内存效率更高
            U, Sigma, VT = randomized_svd(matrix_filled,
                                          n_components=self.n_factors,
                                          n_iter=5, # 可以调整迭代次数
                                          random_state=self.random_state)
            # self.user_factors = U * Sigma
            self.user_factors = U
            self.item_factors = np.diag(Sigma) @ VT
            # self.user_factors = U @ np.diag(np.sqrt(Sigma))
            # self.item_factors = np.diag(np.sqrt(Sigma)) @ VT

            print("DEBUG (SVD): SVD training complete.")

        except MemoryError:
            print("\nError: MemoryError during SVD matrix creation. SVD baseline might not be feasible.")
            self.user_factors = None; self.item_factors = None
        except Exception as e:
            print(f"Error during SVD training: {e}")
            self.user_factors = None; self.item_factors = None


    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的 SVD 模型预测评分。
        """
        print("DEBUG (SVD): Predicting with SVD Recommender...")
        if self.user_factors is None or self.item_factors is None or self.user_map is None or self.item_map is None:
            warnings.warn("SVD model not trained successfully or missing mappings. Predicting global average.")
            return np.full(len(test_data), self.global_avg if not pd.isna(self.global_avg) else 0.0)

        if not all(col in test_data.columns for col in [self.user_col, self.item_col]):
            raise KeyError(f"SVDRecommender: test_data missing one of '{self.user_col}', '{self.item_col}'. Columns: {test_data.columns.tolist()}")


        preds = []
        for _, row in test_data.iterrows():
            user_orig_id = row[self.user_col]
            item_orig_id = row[self.item_col]
            pred = self.global_avg # 默认预测

            if user_orig_id in self.user_map and item_orig_id in self.item_map:
                user_idx = self.user_map[user_orig_id]
                item_idx = self.item_map[item_orig_id]

                if user_idx < self.user_factors.shape[0] and item_idx < self.item_factors.shape[1]:
                     user_vector = self.user_factors[user_idx, :]
                     item_vector = self.item_factors[:, item_idx] # item_factors 是 V^T
                     pred = np.dot(user_vector, item_vector)
                     # 如果在训练时没有减去均值，这里一般不用加回来
                     # 如果因子分解的是中心化评分，则需要加上均值
                     # pred += self.global_avg
                # else:
                    # print(f"DEBUG (SVD): User or Item index out of bounds during prediction. User: {user_orig_id}({user_idx}), Item: {item_orig_id}({item_idx})")

            preds.append(pred)

        print("DEBUG (SVD): SVD predictions generated.")
        return np.array(preds)
