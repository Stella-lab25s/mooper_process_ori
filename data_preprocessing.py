import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # 用于将原始 ID 映射到连续整数
import warnings # 导入警告

class DataProcessor:
    def __init__(self, data_source, user_col='user_id', item_col='item_id', rating_col='rating'): # 添加列名参数
        """
        初始化 DataProcessor。

        Args:
            data_source: 可以是文件路径 (str) 或 Pandas DataFrame。
            user_col (str): 原始用户 ID 列名.
            item_col (str): 原始物品 ID 列名.
            rating_col (str): 原始评分列名.
        """
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        # --- *修改点*: 内部统一使用 'user_id', 'item_id', 'rating' ---
        self.internal_user_col = 'user_id'
        self.internal_item_col = 'item_id'
        self.internal_rating_col = 'rating'
        self.required_columns_orig = [user_col, item_col, rating_col] # 检查原始列名

        self.file_path = None # 初始化 file_path
        self.raw_data = None  # 初始化 raw_data

        if isinstance(data_source, str):
            print(f"DEBUG: DataProcessor initialized with file path: {data_source}")
            self.file_path = data_source # 存储文件路径
        elif isinstance(data_source, pd.DataFrame):
            print("DEBUG: DataProcessor initialized with DataFrame.")
            self.raw_data = data_source.copy() # 使用提供的 DataFrame
        else:
            raise ValueError("data_source 必须是文件路径 (str) 或 Pandas DataFrame")

        self.data = None # 处理后的数据 (包含映射后 ID 和 group_id)
        self.user_item_matrix = None
        self.user_groups = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def _check_columns(self, df, columns_to_check):
        """检查 DataFrame 是否包含必需的列"""
        missing_cols = [col for col in columns_to_check if col not in df.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺少必需的列: {', '.join(missing_cols)}. Available columns: {df.columns.tolist()}")

    def preprocess(self):
        """数据预处理: 加载(如果需要), 清洗, 重命名列, ID映射"""
        print("DEBUG: 进入 DataProcessor.preprocess...")
        # --- 加载数据 (如果初始化时提供的是路径) ---
        if self.file_path is not None and self.raw_data is None:
             try:
                  print(f"DEBUG: Reading CSV from self.file_path: {self.file_path}")
                  self.raw_data = pd.read_csv(self.file_path) # 在这里加载数据
             except Exception as e:
                  # 重新抛出更具体的错误
                  raise Exception(f"Error loading data from file '{self.file_path}': {e}")
        elif self.raw_data is None:
             # 如果既没有文件路径也没有提供 DataFrame
             raise ValueError("No data source (file path or DataFrame) provided to DataProcessor.")

        print("DEBUG: 检查原始数据列...")
        self._check_columns(self.raw_data, self.required_columns_orig)

        # --- 选择并重命名列 ---
        print("DEBUG: 选择并重命名列...")
        self.data = self.raw_data[self.required_columns_orig].copy()
        rename_map = {
            self.user_col: self.internal_user_col,
            self.item_col: self.internal_item_col,
            self.rating_col: self.internal_rating_col
        }
        self.data.rename(columns=rename_map, inplace=True)
        print(f"DEBUG: 列已重命名为: {self.data.columns.tolist()}")


        # --- 清洗数据 ---
        print("DEBUG: 清洗数据 (处理缺失值)...")
        initial_rows = len(self.data)
        # 确保评分列是数值类型，无法转换的变成 NaN
        self.data[self.internal_rating_col] = pd.to_numeric(self.data[self.internal_rating_col], errors='coerce')
        self.data = self.data.dropna(subset=[self.internal_user_col, self.internal_item_col, self.internal_rating_col])
        if len(self.data) < initial_rows:
            print(f"DEBUG: 删除了 {initial_rows - len(self.data)} 行包含缺失值或无效评分的记录。")
        if len(self.data) == 0:
            raise ValueError("错误: 清洗后数据为空。")

        # --- ID 映射 ---
        print("DEBUG: 进行 ID 映射...")
        self.data['user_id_mapped'] = self.user_encoder.fit_transform(self.data[self.internal_user_col])
        self.data['item_id_mapped'] = self.item_encoder.fit_transform(self.data[self.internal_item_col])
        print(f"用户 ID 已映射到 0-{len(self.user_encoder.classes_)-1}")
        print(f"物品 ID 已映射到 0-{len(self.item_encoder.classes_)-1}")

        # --- 创建 user-item 交互矩阵 (用于聚类) ---
        print("DEBUG: 创建 User-Item 交互矩阵...")
        try:
            self.user_item_matrix = pd.pivot_table(
                self.data,
                index='user_id_mapped',
                columns='item_id_mapped',
                values=self.internal_rating_col, # 使用内部评分列名
                aggfunc='mean'
            ).fillna(0)
            print("User-Item 交互矩阵创建完成。")
        except Exception as e:
             print(f"创建 User-Item 矩阵时出错: {e}")
             raise

        print("DEBUG: DataProcessor.preprocess 完成。")
        return self

    def cluster_users(self, n_groups=10, random_state=42):
        """用户聚类分群 (在 preprocess 之后调用)"""
        if self.data is None or self.user_item_matrix is None:
            raise ValueError("数据尚未预处理，请先调用 preprocess()")

        print(f"DEBUG: 开始用户聚类 (K={n_groups})...")
        if self.user_item_matrix.empty:
             warnings.warn("用户物品矩阵为空，无法进行聚类。将为所有用户分配群组 0。")
             num_users = len(self.user_encoder.classes_)
             self.user_groups = np.zeros(num_users, dtype=int)
             user_id_map = {user_idx: 0 for user_idx in range(num_users)}
             self.data['group_id'] = self.data['user_id_mapped'].map(user_id_map)
        else:
             kmeans = KMeans(n_clusters=n_groups, random_state=random_state, n_init=10)
             self.user_groups = kmeans.fit_predict(self.user_item_matrix.values)
             user_id_to_group_map = {user_id: group_id for user_id, group_id in zip(self.user_item_matrix.index, self.user_groups)}
             self.data['group_id'] = self.data['user_id_mapped'].map(user_id_to_group_map)
             if self.data['group_id'].isnull().any():
                  warnings.warn("部分用户的 group_id 未能成功映射！将填充为 -1。")
                  self.data['group_id'] = self.data['group_id'].fillna(-1).astype(int)

        print(f"用户聚类完成，已分配到 {self.data['group_id'].nunique()} 个群组。")
        # 打印群组分布
        print("DEBUG: 群组分布:")
        print(self.data['group_id'].value_counts())
        return self

    def split_data(self, test_size=0.2, random_state=42):
        """
        数据划分 (在 cluster_users 之后调用)
        返回包含映射后 ID, group_id, 和原始评分 ('rating') 的训练集和测试集。
        """
        if self.data is None or 'group_id' not in self.data.columns:
            raise ValueError("数据尚未预处理或聚类，请先调用 preprocess() 和 cluster_users()")

        print(f"DEBUG: 开始划分数据 (测试集比例: {test_size})...")
        # self.data 包含 user_id_mapped, item_id_mapped, rating, group_id, 以及原始 ID 列
        train_df, test_df = train_test_split(
            self.data, # 使用包含所有处理后信息的 DataFrame
            test_size=test_size,
            random_state=random_state,
        )
        print("DEBUG: 数据划分完成。")
        print(f"DEBUG: 训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
        return train_df, test_df

    def get_mappings(self):
        """获取用户和物品的 ID 映射器"""
        return self.user_encoder, self.item_encoder

    def get_processed_data(self):
        """获取包含所有处理信息的 DataFrame"""
        return self.data
