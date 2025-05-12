import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
import inspect
import warnings
import time # 确保导入 time
import matplotlib
import matplotlib.pyplot as plt

try:

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("DEBUG (experiments.py): Matplotlib font set to SimHei.")
except Exception as e_simhei:
    print(f"DEBUG (experiments.py): Failed to set SimHei font ({e_simhei}), trying Microsoft YaHei...")
    try:
        # 'Microsoft YaHei' (微软雅黑)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("DEBUG (experiments.py): Matplotlib font set to Microsoft YaHei.")
    except Exception as e_yahei:
        print(f"DEBUG (experiments.py): Failed to set Microsoft YaHei font ({e_yahei}).")
        warnings.warn(
            "中文字体设置失败，图表中的中文可能仍然是乱码。"
            "请确保您的系统已安装 SimHei 或 Microsoft YaHei 字体，"
            "或者在代码中指定您系统中已安装的其他中文字体名称。"
        )
# ------------------------------------
try:
    from evaluate import Evaluator
except ImportError:
    print("错误: 无法导入 Evaluator。请确保 evaluate.py 文件存在且在 Python 路径中。")
    class Evaluator: # 定义虚拟类以防崩溃
        @staticmethod
        def evaluate_all(preds, labels, groups, k=10, precision_threshold=70.0, ndcg_bins=None):
            warnings.warn("Using dummy Evaluator. Metrics will be NaN.")
            return {'MAE': np.nan, 'RMSE': np.nan, 'Precision': np.nan, f'NDCG@{k}': np.nan, 'Satisfaction': np.nan}

class Experiment:
    def __init__(self, train_data_mapped, test_data_orig_ids):
        """
        初始化实验设置。

        Args:
            train_data_mapped (pd.DataFrame): 包含映射后 ID 和原始评分 ('rating') 的训练数据集。
                                                这个数据会传递给所有模型的 train 方法。
            test_data_orig_ids (pd.DataFrame): 包含原始 ID、原始评分 ('rating') 和 'group_id' 的测试数据集。
                                               这个数据会传递给所有模型的 predict 方法。
        """
        self.train_data_mapped = train_data_mapped
        self.test_data_orig_ids = test_data_orig_ids
        self.models = {} # 用来存储模型对象
        self.results = {} # 用来存储每个模型的最终评估指标

        required_test_cols = ['rating'] # 评估至少需要原始评分
        # group_id 对 Satisfaction 指标是必需的
        if 'group_id' not in self.test_data_orig_ids.columns:
            warnings.warn(f"测试数据 test_data_orig_ids 缺少 'group_id' 列。Satisfaction 指标可能无法准确计算。")
        if not all(col in self.test_data_orig_ids.columns for col in required_test_cols):
             warnings.warn(f"测试数据 test_data_orig_ids 缺少评估所需的列: {required_test_cols}. 评估可能失败或不准确。")


    def add_model(self, name, model_object):
        """向实验中添加一个模型。"""
        self.models[name] = model_object
        self.results[name] = {}
        print(f"DEBUG: 模型 '{name}' 已添加到实验中。")

    def run(self, user_encoder=None, item_encoder=None, precision_threshold=70.0, ndcg_bins=None):
        """运行所有已添加模型的训练和评估"""
        if not self.models:
            print("错误: 没有模型被添加到实验中。")
            return

        print("\nDEBUG (Experiment): --- 开始运行对比实验 ---")
        start_time_exp_run = time.time()
        try:
            # 确保 'rating' 列存在于测试数据中
            if 'rating' not in self.test_data_orig_ids.columns:
                raise KeyError(f"'rating' column is missing from test_data_orig_ids. Columns: {self.test_data_orig_ids.columns.tolist()}")
            true_labels = self.test_data_orig_ids['rating'].values # 原始评分

            if 'group_id' in self.test_data_orig_ids.columns:
                 groups = self.test_data_orig_ids['group_id'].values
            else:
                 warnings.warn("测试数据缺少 'group_id'，满意度指标将使用默认值 -1。")
                 groups = np.full(len(self.test_data_orig_ids), -1) # 创建一个默认的 group 数组
        except KeyError as e:
            print(f"错误: 测试数据缺少必需的列 {e}。无法进行评估。")
            return

        for name, model in self.models.items(): # 直接迭代模型对象
            print(f"\nDEBUG (Experiment): --- 处理模型: {name} ---")
            start_time_model = time.time()
            try:
                # 检查模型是否有 train 方法
                if hasattr(model, 'train') and callable(model.train):
                    print(f"DEBUG (Experiment): 训练 {name}...")
                    sig = inspect.signature(model.train)
                    train_args = {'train_data': self.train_data_mapped} # 所有模型都用映射数据训练
                    if 'user_encoder' in sig.parameters: train_args['user_encoder'] = user_encoder
                    if 'item_encoder' in sig.parameters: train_args['item_encoder'] = item_encoder
                    model.train(**train_args)
                    print(f"DEBUG (Experiment): {name} 训练完成。")
                else:
                    warnings.warn(f"模型 '{name}' 缺少 'train' 方法或 'train' 不是可调用对象。跳过训练步骤。")

                # 检查模型是否有 predict 方法
                if not hasattr(model, 'predict') or not callable(model.predict):
                    warnings.warn(f"模型 '{name}' 缺少 'predict' 方法或 'predict' 不是可调用对象。跳过评估。")
                    self.results[name] = {'Error': 'Missing predict method'} # 直接将错误信息字典赋值
                    continue # 跳到下一个模型

                print(f"DEBUG (Experiment): 评估 {name}...")
                predictions = model.predict(self.test_data_orig_ids) # 预测时使用原始ID的测试集
                print(f"DEBUG (Experiment): {name} 评估完成。")

                # 确保预测结果长度与测试数据一致
                if len(predictions) != len(self.test_data_orig_ids):
                     warnings.warn(f"模型 '{name}' 预测数量 ({len(predictions)}) 与测试数据量 ({len(self.test_data_orig_ids)}) 不匹配！将截断。")
                     predictions = predictions[:len(self.test_data_orig_ids)]
                     true_labels_matched = true_labels[:len(predictions)]
                     groups_matched = groups[:len(predictions)]
                else:
                     true_labels_matched = true_labels
                     groups_matched = groups

                if len(predictions) > 0: # 确保有预测结果才评估
                     metrics = Evaluator.evaluate_all(
                         predictions, true_labels_matched, groups_matched,
                         precision_threshold=precision_threshold, ndcg_bins=ndcg_bins
                     )
                     self.results[name] = metrics # 直接将 metrics 字典赋值
                     print(f"DEBUG (Experiment): 模型 '{name}' 评估结果: {metrics}")
                else:
                     print(f"DEBUG (Experiment): 模型 '{name}' 没有生成有效预测，跳过评估。")
                     self.results[name] = {'MAE': np.nan, 'RMSE': np.nan, 'Precision': np.nan, 'NDCG@10': np.nan, 'Satisfaction': np.nan}

            except Exception as e:
                print(f"处理模型 '{name}' 时发生严重错误: {e}")
                traceback.print_exc()
                self.results[name] = {'Error': str(e)} # 直接将错误信息字典赋值
            finally:
                end_time_model = time.time()
                print(f"处理模型 '{name}' 耗时: {end_time_model - start_time_model:.2f} 秒")

        end_time_exp_run = time.time()
        print(f"\nDEBUG (Experiment): --- 对比实验运行结束 (总耗时: {end_time_exp_run - start_time_exp_run:.2f} 秒) ---")

    def report(self, plot=True):
        if not self.results:
            print("没有可报告的结果。请先运行 run() 方法。")
            return pd.DataFrame(), None

        valid_metrics_list = []
        model_names_for_df = []
        error_models_info = {}

        for name, metrics_or_error in self.results.items():
            if isinstance(metrics_or_error, dict):
                if 'Error' in metrics_or_error:
                    error_models_info[name] = metrics_or_error['Error']
                elif 'MAE' in metrics_or_error: # 假设 MAE 是一个有效的指标
                    valid_metrics_list.append(metrics_or_error)
                    model_names_for_df.append(name)
            else:
                warnings.warn(f"模型 '{name}' 的结果格式不正确或 metrics 为空。")


        if not valid_metrics_list:
             print("没有有效的评估结果可供报告。")
             if error_models_info:
                  print("以下模型在处理时遇到错误:")
                  for name, error_msg in error_models_info.items(): print(f"- {name}: {error_msg}")
             return pd.DataFrame(), None

        results_df = pd.DataFrame(valid_metrics_list, index=model_names_for_df)
        for col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        results_df = results_df.round(4)

        fig = None
        if plot:
            print("\nDEBUG (Experiment): 生成结果图表...")
            try:
                # 确保 matplotlib 已导入 (在文件顶部已导入)
                metrics_to_plot = [col for col in ['MAE', 'RMSE', 'Precision', 'NDCG@10', 'Satisfaction'] if col in results_df.columns]
                plot_df = results_df[metrics_to_plot].dropna(axis=1, how='all') # 只绘制非全 NaN 的列

                if not plot_df.empty:
                    num_metrics = len(plot_df.columns)
                    ncols = min(num_metrics, 3); nrows = (num_metrics + ncols - 1) // ncols
                    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
                    axes = axes.flatten()
                    for i, metric in enumerate(plot_df.columns):
                        ax = axes[i]; plot_data_metric = plot_df[metric].dropna()
                        if not plot_data_metric.empty:
                             plot_data_metric.plot(kind='bar', ax=ax, rot=30)
                             ax.set_title(f'Comparison of {metric}'); ax.set_ylabel(metric); ax.set_xlabel('Model')
                             ax.grid(axis='y', linestyle='--')
                             for container in ax.containers: ax.bar_label(container, fmt='%.4f', padding=3)
                        else:
                             ax.set_title(f'Comparison of {metric} (No data)'); ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                    for j in range(i + 1, len(axes)): fig.delaxes(axes[j]) # 隐藏多余的子图
                    plt.tight_layout(pad=3.0); print("DEBUG (Experiment): 图表对象已生成。")
                else: print("没有有效的指标可以绘制图表。")
            except ImportError: print("Matplotlib 未安装，无法生成图表。"); fig = None
            except Exception as e: print(f"绘制图表时出错: {e}"); traceback.print_exc(); fig = None
        return results_df, fig

    def save_plot(self, filepath):
         results_df, fig = self.report(plot=True) # 调用 report 获取 fig
         if fig is not None and not results_df.empty:
              print(f"\nDEBUG (Experiment): 尝试保存结果图表到: {filepath}")
              try:
                   dir_name = os.path.dirname(filepath)
                   if dir_name and not os.path.exists(dir_name): # 检查目录是否为空字符串
                        os.makedirs(dir_name, exist_ok=True)
                   fig.savefig(filepath, bbox_inches='tight'); plt.close(fig) # 保存后关闭图形
                   print(f"图表已保存到 {filepath}")
              except Exception as e: print(f"保存图表到 '{filepath}' 时出错: {e}"); traceback.print_exc(); plt.close('all')
         elif results_df.empty: print("没有有效的评估结果，无法保存图表。")
         else: print("未能成功生成图表对象，无法保存。")
