import pandas as pd
import torch
from torch_geometric.data import TemporalData
from datetime import datetime
import networkx as nx
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pywt  # 小波变换库
from scipy.spatial.distance import cdist
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.stats import boxcox
import os
import glob
import logging


def read_and_preprocess_data(account_address):
    # 读取 CSV 文件
    # df_data = pd.read_csv(f'/data/hqn/EthereumHeist/Heist_data/KuCoinHacker/{account_address}.csv')
    df_data = pd.read_csv(f'/data/hqn/BlockchainSpider/data/{account_address}.csv')

    # 将 'from' 和 'to' 转换为整数索引
    unique_nodes = pd.concat([df_data['from'], df_data['to']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    df_data['from_idx'] = df_data['from'].map(node_to_idx)
    df_data['to_idx'] = df_data['to'].map(node_to_idx)

    # 提取边索引和时间戳
    src = df_data['from_idx'].values
    dst = df_data['to_idx'].values
    data_timestamps = df_data['timeStamp'].values

    # 转换为 PyTorch 张量
    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)
    data_timestamps = torch.tensor(data_timestamps, dtype=torch.float)

    # 确保数据按时间戳排序
    sorted_indices = torch.argsort(data_timestamps)
    src = src[sorted_indices]
    dst = dst[sorted_indices]
    data_timestamps = data_timestamps[sorted_indices]

    return src, dst, data_timestamps

def create_temporal_data(src, dst, data_timestamps):
    # 创建 PyG 的 TemporalData 对象
    temporal_data = TemporalData(
        src=src,  # 源节点
        dst=dst,  # 目标节点
        t=data_timestamps  # 时间戳
    )
    return temporal_data

def split_into_time_windows(temporal_data, num_windows=1000):
    # 计算总时间跨度
    min_time = temporal_data.t.min().item()
    max_time = temporal_data.t.max().item()
    total_time_span = max_time - min_time

    # 划分时间窗口
    window_size = total_time_span / num_windows

    # 存储每个窗口的子图
    subgraphs = []

    # 划分窗口并处理每个窗口
    for i in range(num_windows):
        # 计算当前窗口的起始和结束时间
        start_time = min_time + i * window_size
        end_time = start_time + window_size if i < num_windows - 1 else max_time  # 最后一个窗口包含剩余时间

        # 提取当前窗口内的交易
        mask = (temporal_data.t >= start_time) & (temporal_data.t < end_time)
        subgraph = temporal_data[mask]

        # 将子图添加到列表中
        subgraphs.append(subgraph)

    return subgraphs

def analyze_subgraphs(subgraphs):
    time_data = []

    for i, subgraph in enumerate(subgraphs):
        # 如果子图为空，不记录空子图的信息
        if subgraph.num_events == 0:
            print(f"Window {i + 1}: No events")
            continue

        # 获取当前窗口的最小和最大时间戳
        min_time = subgraph.t.min().item()
        max_time = subgraph.t.max().item()

        # 将 Unix 时间戳转换为 yy-mm-dd 格式
        min_time_str = datetime.utcfromtimestamp(min_time).strftime('%y-%m-%d')
        max_time_str = datetime.utcfromtimestamp(max_time).strftime('%y-%m-%d')

        print(f"Window {i + 1}: Events={subgraph.num_events}, Time range [{min_time_str}, {max_time_str}]")
        # 将子图转换为 networkx 的图对象
        edge_list = list(zip(subgraph.src.cpu().numpy(), subgraph.dst.cpu().numpy()))
        G = nx.DiGraph()  # 使用有向图
        G.add_edges_from(edge_list)

        # 计算 3-node motifs 数量
        motifs_3 = nx.triadic_census(G)  # 计算所有 3-node motifs
        # 排除空 motif '003'，并存储 motifs 数量
        motifs_filtered = {k: v for k, v in motifs_3.items() if k != '003'}
        # 将子图编号、时间范围和 motifs 数量添加到数据列表
        time_data.append({
            'Subgraph ID': i + 1,  # 子图编号从 1 开始
            'Begin Time': min_time_str,
            'End Time': max_time_str,
            **motifs_filtered  # 展开 motifs 数量
        })
        print({
            'Subgraph ID': i + 1,  # 子图编号从 1 开始
            'Begin Time': min_time_str,
            'End Time': max_time_str,
            **motifs_filtered  # 展开 motifs 数量
        })

    return time_data

def save_to_csv(time_data, account_address):
    # 将数据存储到 csv_y.csv
    with open(f'data/{account_address}/{account_address}_time.csv', 'w', newline='') as csvfile:
        # 获取表头（Subgraph ID, Begin Time, End Time, 所有 motifs 类型）
        header = ['Subgraph ID', 'Begin Time', 'End Time'] + list(time_data  [0].keys())[3:]
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()  # 写入表头
        writer.writerows(time_data)  # 写入数据

    print(f'CSV file saved: data/{account_address}/{account_address}_time.csv')



# wavelet analysis

def next_power_of_2(n):
    """计算下一个 2 的幂次方"""
    return 2 ** int(np.ceil(np.log2(n)))

def wavelet_analysis(time_series, wavelet='cmor1.5-1.0', scales=np.arange(1, 128)):
    """
    使用连续小波变换（CWT）分析时间序列
    :param time_series: 一维时间序列
    :param wavelet: 小波基函数，默认为 'cmor1.5-1.0'（Complex Morlet 小波）
    :param scales: 尺度参数，控制频率范围
    :return: 小波系数矩阵，尺度数组
    """
    # 计算信号长度的下一个 2 的幂次方的两倍
    original_length = len(time_series)
    next_pow2 = next_power_of_2(original_length)
    desired_length = 2 * next_pow2

    # 扩展信号长度
    padded_time_series = np.pad(time_series, (0, desired_length - original_length), mode='constant')

    # 计算连续小波变换
    coefficients, frequencies = pywt.cwt(padded_time_series, scales, wavelet)

    # 扣除延长后的数据，只保留与原始信号对齐的部分
    coefficients = coefficients[:, :original_length]
    return coefficients, scales



def log_transform(time_series):
    """
    对时间序列进行对数变换
    :param time_series: 一维时间序列
    :return: 对数变换后的时间序列
    """
    return np.log(time_series + 1)  # 加 1 避免对 0 取对数



def z_score_normalization(time_series):
    """
    对时间序列进行 Z-Score 标准化
    :param time_series: 一维时间序列
    :return: 标准化后的时间序列
    """
    mean = np.mean(time_series)  # 计算均值
    std = np.std(time_series)    # 计算标准差
    return (time_series - mean) / std  # 标准化


def robust_scaling(time_series):
    """
    对时间序列进行 Robust Scaling（鲁棒归一化）
    :param time_series: 一维时间序列
    :return: 归一化后的时间序列
    """
    median = np.median(time_series)  # 计算中位数
    iqr = np.percentile(time_series, 75) - np.percentile(time_series, 25)  # 计算四分位距
    return (time_series - median) / iqr  # 归一化


def boxcox_transformation(time_series, offset=1):
    """
    对时间序列进行 Box-Cox 变换（处理 0 值）
    :param time_series: 一维时间序列
    :param offset: 偏移量，用于处理 0 值，默认为 1
    :return: 变换后的时间序列
    """
    # 检查是否有 0 或负值
    if np.any(time_series <= 0):
        # print("警告：时间序列中包含 0 或负值，将应用偏移法处理。")
        time_series = time_series + offset  # 偏移法处理
    
    # 进行 Box-Cox 变换
    transformed, _ = boxcox(time_series)
    return transformed




def plot_transformed_time_series(account_address, df, multivariate_columns, time_series_log_dict):
    """
    绘制对数变换后的时间序列图并保存
    :param account_address: 账户地址，用于保存图片
    :param df: 包含时间序列的 DataFrame
    :param multivariate_columns: 多元序列变量列表
    :param time_series_log_dict: 对数变换后的时间序列数据（字典形式，key 为列名，value 为变换后的数据）
    """
    # 创建画布
    plt.figure(figsize=(80, 6))
    
    # 使用 colormap 为每条线分配唯一颜色
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    colors = colors[:len(multivariate_columns)]  # 只取需要的颜色
    
    # 绘制每条 motifs 的折线图
    for idx, column in enumerate(multivariate_columns):
        plt.plot(df['End Time'], time_series_log_dict[column], label=column, color=colors[idx])
    

    # 设置图表标题和标签
    plt.title('3-node Motifs counts by evolving subgraphs', fontsize=16)
    plt.xlabel('Timeline (Year-Month-Day)', fontsize=12)
    plt.ylabel('Motifs Count (transformed)', fontsize=12)
    
    # 添加图例
    plt.legend(title='Motifs', bbox_to_anchor=(0, 1.15), ncol=15, loc='upper left')
    
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.xticks(fontsize=10)
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'data/{account_address}/{account_address}_transformed_time_series.png', dpi=300, bbox_inches='tight')
    
    print(f'Saved data/{account_address}/{account_address}_transformed_time_series.png')
    # # 显示图像
    # plt.show()



def analyze_multivariate_time_series(account_address, csv_file, transform_method='log', wavelet='cmor1.5-1.0', scales=np.arange(1, 128)):
    """
    分析多元时间序列的非连续周期
    :param account_address: 账户地址，用于保存图片
    :param csv_file: CSV文件路径
    :param transform_method: 标准化方法，可选 'log', 'zscore', 'robust', 'boxcox'
    :param wavelet: 小波基函数，默认为 'cmor1.5-1.0'
    :param scales: 尺度参数，控制频率范围
    :return: 平均小波系数矩阵，尺度数组，时间戳数组
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 将 'End Time' 转换为时间戳
    df['End Time'] = pd.to_datetime(df['End Time'], format='%y-%m-%d')
    # 按时间排序
    df = df.sort_values(by='End Time')
    # 提取多元序列变量
    multivariate_columns = ['012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300']
    # 存储每个变量的小波系数
    all_coefficients = []
    # 存储每个变量的变换结果
    time_series_transformed_dict = {}
    
    # 对每个变量进行分析
    for column in multivariate_columns:
        time_series = df[column].values
        # 根据 transform_method 选择标准化方法
        if transform_method == 'log':
            time_series_transformed = log_transform(time_series)
        elif transform_method == 'zscore':
            time_series_transformed = z_score_normalization(time_series)
        elif transform_method == 'robust':
            time_series_transformed = robust_scaling(time_series)
        elif transform_method == 'boxcox':
            time_series_transformed = boxcox_transformation(time_series)
        else:
            raise ValueError(f"未知的标准化方法: {transform_method}")
        
        # 存储变换结果
        time_series_transformed_dict[column] = time_series_transformed
        # 进行小波变换
        coefficients, scales = wavelet_analysis(time_series_transformed, wavelet, scales)
        all_coefficients.append(coefficients)
    
    # 计算平均小波系数
    avg_coefficients = np.mean(all_coefficients, axis=0)
    
    # 绘制变换后的时间序列图并保存
    plot_transformed_time_series(account_address, df, multivariate_columns, time_series_transformed_dict)
    
    return avg_coefficients, scales, df['End Time']





def plot_wavelet_coefficients(account_address,coefficients, scales, timestamps):
    """
    绘制小波系数图
    :param coefficients: 小波系数矩阵
    :param scales: 尺度参数
    :param timestamps: 时间戳数组
    """
    if len(timestamps) == 0 or len(scales) == 0:
        raise ValueError("timestamps 或 scales 为空，无法绘图")
    
    plt.figure(figsize=(10, 6))
    # 绘制小波系数幅值
    plt.imshow(np.abs(coefficients), extent=[timestamps.iloc  [0], timestamps.iloc[-1], scales[-1], scales  [0]],
               cmap='jet', aspect='auto', vmax=abs(coefficients).max(), vmin=0)
    plt.colorbar(label='Magnitude')
    plt.title('Wavelet Coefficients (Magnitude)')
    plt.xlabel('Time')
    plt.ylabel('Scales')
    plt.savefig(f'data/{account_address}/{account_address}_wavelet_magnitude.png', dpi=300, bbox_inches='tight')
    # plt.show()
    print(f'Saved data/{account_address}/{account_address}_wavelet_magnitude.png')
    
    


# def find_significant_maxima(wavelet_var, scales, prominence=0.1, top_n=3):
#     """
#     找出小波方差图中所有显著的最大值，并返回最显著的 top_n 个
#     :param wavelet_var: 小波方差值，一维数组
#     :param scales: 尺度参数，一维数组
#     :param prominence: 显著性的阈值，默认为 0.1
#     :param top_n: 返回最显著的最大值数量，默认为 3
#     :return: 显著最大值的尺度和小波方差值
#     """
#     # 寻找局部最大值
#     print(f"prominence= {prominence}")
#     peaks, properties = find_peaks(wavelet_var, prominence=prominence)
    
#     # 如果没有找到峰值，返回空数组
#     if len(peaks) == 0:
#         return np.array([]), np.array([])
    
#     # 根据显著性（prominence）排序，选择最显著的 top_n 个
#     prominences = properties['prominences']
#     sorted_indices = np.argsort(prominences)[::-1]  # 从大到小排序
#     top_indices = sorted_indices[:top_n]  # 取前 top_n 个
    
#     # 提取最显著的 top_n 个最大值的尺度和小波方差值
#     significant_scales = scales[peaks[top_indices]]
#     significant_variances = wavelet_var[peaks[top_indices]]
    
#     return significant_scales, significant_variances


def find_significant_maxima(wavelet_var, scales, prominence=0.1, top_n=3):
    """
    找出小波方差图中所有显著的最大值，并返回最显著的 top_n 个
    :param wavelet_var: 小波方差值，一维数组
    :param scales: 尺度参数，一维数组
    :param prominence: 显著性的阈值，默认为 0.1
    :param top_n: 返回最显著的最大值数量，默认为 3
    :return: 显著最大值的尺度、小波方差值和显著性值
    """
    # 寻找局部最大值
    print(f"prominence= {prominence}")
    peaks, properties = find_peaks(wavelet_var, prominence=prominence)
    
    # 如果没有找到峰值，返回空数组
    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # 根据显著性（prominence）排序，选择最显著的 top_n 个
    prominences = properties['prominences']
    sorted_indices = np.argsort(prominences)[::-1]  # 从大到小排序
    top_indices = sorted_indices[:top_n]  # 取前 top_n 个
    
    # 提取最显著的 top_n 个最大值的尺度、小波方差值和显著性值
    significant_scales = scales[peaks[top_indices]]
    significant_variances = wavelet_var[peaks[top_indices]]
    significant_prominences = prominences[top_indices]
    
    return significant_scales, significant_variances, significant_prominences

# def plot_wavelet_variance_with_maxima(account_address, coefficients, scales, timestamps, prominence=0.2):
#     """
#     绘制小波方差图，并输出显著的最大值坐标
#     :param account_address: 账户地址，用于保存图片
#     :param coefficients: 小波系数矩阵
#     :param scales: 尺度参数
#     :param prominence: 显著性的阈值，默认为 0.1
#     """
#     # 计算小波方差
#     wavelet_var = np.mean(np.abs(coefficients)**2, axis=1)
    
#     # 找出显著的最大值
#     significant_scales, significant_variances = find_significant_maxima(wavelet_var, scales, prominence)
    
#     # 输出显著的最大值坐标(主周期)
    
#     print("Significant Maxima Coordinates:")
#     for scale, variance in zip(significant_scales, significant_variances):
#         print(f"Scale: {scale:.2f}, Variance: {variance:.2f}")
    
#     # 绘制小波方差图
#     plt.figure(figsize=(10, 6))
#     plt.plot(scales, wavelet_var, marker='o', linestyle='-', color='b', label='Wavelet Variance')
#     plt.title('Wavelet Variance')
#     plt.xlabel('Scales')
#     plt.ylabel('Variance')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(f'data/{account_address}/{account_address}_wavelet_variance.png', dpi=300, bbox_inches='tight')
#     # plt.show()
#     print(f'Saved data/{account_address}/{account_address}_wavelet_variance.png')
    
#     #绘制所有显著最大值对应的主周期趋势图
#     plot_main_period_trends(account_address, coefficients, scales, timestamps, significant_scales)


def plot_wavelet_variance_with_maxima(account_address, coefficients, scales, timestamps, prominence=0.2):
    """
    绘制小波方差图，并输出显著的最大值坐标
    :param account_address: 账户地址，用于保存图片
    :param coefficients: 小波系数矩阵
    :param scales: 尺度参数
    :param prominence: 显著性的阈值，默认为 0.1
    """
    # 计算小波方差
    wavelet_var = np.mean(np.abs(coefficients)**2, axis=1)
    
    # 找出显著的最大值
    significant_scales, significant_variances, significant_prominences = find_significant_maxima(wavelet_var, scales, prominence)
    
    # 输出显著的最大值坐标(主周期)
    print("Significant Maxima Coordinates:")
    for scale, variance, prominence in zip(significant_scales, significant_variances, significant_prominences):
        print(f"Scale: {scale:.2f}, Variance: {variance:.2f}, Prominence: {prominence:.2f}")
    
    # 绘制小波方差图
    plt.figure(figsize=(10, 6))
    plt.plot(scales, wavelet_var, marker='o', linestyle='-', color='b', label='Wavelet Variance')
    plt.title('Wavelet Variance')
    plt.xlabel('Scales')
    plt.ylabel('Variance')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'data/{account_address}/{account_address}_wavelet_variance.png', dpi=300, bbox_inches='tight')
    # plt.show()
    print(f'Saved data/{account_address}/{account_address}_wavelet_variance.png')
    
    # 绘制所有显著最大值对应的主周期趋势图
    plot_main_period_trends(account_address, coefficients, scales, timestamps, significant_scales, significant_prominences)
    
    
    
# def plot_main_period_trends(account_address, coefficients, scales, timestamps, significant_scales):
#     """
#     绘制所有显著最大值对应的主周期趋势图，并为每个 Scale 分配唯一颜色
#     :param account_address: 账户地址，用于保存图片
#     :param coefficients: 小波系数矩阵
#     :param scales: 尺度参数
#     :param timestamps: 时间戳数组
#     :param significant_scales: 显著最大值的尺度
#     """
#     # 预定义颜色列表
#     colors = [
#         '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
#         '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
#         '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
#     ]
    
#     # 如果显著尺度的数量超过颜色列表的长度，循环使用颜色
#     if len(significant_scales) > len(colors):
#         colors = colors * (len(significant_scales) // len(colors) + 1)
    
#     plt.figure(figsize=(10, 6))
    
#     # 遍历所有显著尺度
#     for idx, scale in enumerate(significant_scales):
#         # 找到尺度对应的索引
#         scale_idx = np.where(scales == scale)  [0]  [0]
#         # 提取该尺度对应的小波系数
#         scale_coefficients = coefficients[scale_idx, :]
#         # 绘制该尺度的小波系数随时间的变化趋势，并分配唯一颜色
#         plt.plot(timestamps, scale_coefficients, label=f'Scale {scale:.2f}', color=colors[idx])
    
#     # 设置图表标题和标签
#     plt.title('Main Period Trends')
#     plt.xlabel('Time')
#     plt.ylabel('Wavelet Coefficients')
    
#     # 添加图例
#     plt.legend(title='Scales', bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     # 设置网格
#     plt.grid(True, linestyle='--', alpha=0.6)
    
#     # 调整布局
#     plt.tight_layout()
    
#     # 保存图像
#     plt.savefig(f'data/{account_address}/{account_address}_main_period_trends.png', dpi=300, bbox_inches='tight')
    
#     # # 显示图像
#     # plt.show()
#     print(f'Saved data/{account_address}/{account_address}_main_period_trends.png')


def plot_main_period_trends(account_address, coefficients, scales, timestamps, significant_scales, prominences):
    """
    绘制所有显著最大值对应的主周期趋势图，并为每个 Scale 分配唯一颜色
    :param account_address: 账户地址，用于保存图片
    :param coefficients: 小波系数矩阵
    :param scales: 尺度参数
    :param timestamps: 时间戳数组
    :param significant_scales: 显著最大值的尺度
    :param prominences: 显著最大值的显著性值
    """
    # 定义颜色列表，根据显著性大小分配颜色 仅绘制top3 所以只分配了top3的颜色
    colors = ['red', 'orange', 'yellow']  # 最显著为红色，其次为橙色，最后为黄色
    default_color = 'black'  # 默认颜色为黑色
    
    # 根据显著性值（prominences）从大到小排序
    sorted_indices = np.argsort(prominences)[::-1]  # 从大到小排序
    significant_scales = significant_scales[sorted_indices]
    prominences = prominences[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    
    # 遍历所有显著尺度
    for idx, (scale, prominence) in enumerate(zip(significant_scales, prominences)):
        # 找到尺度对应的索引
        scale_idx = np.where(scales == scale)  [0]  [0]
        # 提取该尺度对应的小波系数
        scale_coefficients = coefficients[scale_idx, :]
        # 分配颜色：如果颜色列表用完了，使用默认颜色（黑色）
        color = colors[idx] if idx < len(colors) else default_color
        # 绘制该尺度的小波系数随时间的变化趋势，并分配颜色
        plt.plot(timestamps, scale_coefficients, label=f'Scale {scale:.2f} (Prominence: {prominence:.2f})', color=color)
    
    # 设置图表标题和标签
    plt.title('Main Period Trends')
    plt.xlabel('Time')
    plt.ylabel('Wavelet Coefficients')
    
    # 添加图例
    plt.legend(title='Scales', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'data/{account_address}/{account_address}_main_period_trends.png', dpi=300, bbox_inches='tight')
    
    # # 显示图像
    # plt.show()
    print(f'Saved data/{account_address}/{account_address}_main_period_trends.png')

def main_periodicity_analysis(account_address):
    # 创建以 account_address 命名的目录
    output_dir = f'data/{account_address}'
    os.makedirs(output_dir, exist_ok=True)  # 如果目录已存在，不会报错
    # 读取并预处理数据
    src, dst, data_timestamps = read_and_preprocess_data(account_address)

    # 创建 TemporalData 对象
    temporal_data = create_temporal_data(src, dst, data_timestamps)

    # 划分时间窗口
    subgraphs = split_into_time_windows(temporal_data)

    # 分析子图并计算 motifs
    time_data = analyze_subgraphs(subgraphs)

    # 保存结果到 CSV 文件
    save_to_csv(time_data, account_address)

    # 打印总子图数量
    print(f"Total subgraphs: {len(subgraphs)}")

    csv_file = f"data/{account_address}/{account_address}_time.csv"
    # 分析多元时间序列
    # scales = np.arange(1, 128)  # 尺度参数 704
    scales = np.arange(1, 704)  # 尺度参数
    avg_coefficients, scales, timestamps = analyze_multivariate_time_series(account_address,csv_file,wavelet='cmor1.5-1.0', scales=scales)
    # 绘制小波系数图
    plot_wavelet_coefficients(account_address,avg_coefficients, scales, timestamps)
    # 绘制小波方差图并找出显著的最大值，并画出所有主周期趋势图
    plot_wavelet_variance_with_maxima(account_address, avg_coefficients, scales,timestamps, prominence=0.1)  


def get_csv_files(directory):
    # 获取目录下所有的CSV文件
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # 过滤掉目录，只保留文件
    csv_files = [f for f in csv_files if os.path.isfile(f)]
    
    # 只返回文件名（去掉路径和扩展名）
    csv_files = [os.path.splitext(os.path.basename(f))  [0] for f in csv_files]

    print("Number of CSV file crawled:")
    print(len(csv_files))
    return csv_files

def get_exist_address():
    target_directory = "/data/hqn/EthereumHeist/Periodicity_analysis/data"
    subdirectories = [d for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))]
    print("Number of processed CSV files:")
    print(len(subdirectories))
    return subdirectories




if __name__ == "__main__":
    directory = "/data/hqn/BlockchainSpider/data"
    csv_files = get_csv_files(directory)
    existing_addresses =  get_exist_address()

    # 配置错误日志
    logging.basicConfig(filename='/data/hqn/EthereumHeist/Periodicity_analysis/error.log', level=logging.ERROR, format='%(asctime)s - %(message)s')

    for address in csv_files:
        try:
            if address in existing_addresses:
                print(f"{address} 已存在于处理后的data目录中")
                continue
            else:
                main_periodicity_analysis(address)
                print(f"成功处理地址: {address}")

        except Exception as e:
            logging.error(f"处理地址 {address} 时出错: {e}")
            print(f"处理地址 {address} 时出错: {e}")
    


    


