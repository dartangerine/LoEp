import numpy as np
import pandas as pd
from multiprocessing import shared_memory


def read_bedgraph(filepath):
    """读取bedgraph文件并返回DataFrame，NaN填充为0"""
    df = pd.read_csv(filepath, sep='\t', header=None, 
                     names=['chrom', 'start', 'end', 'value'])
    df['value'] = df['value'].fillna(0)
    return df


def parse_window_sizes(window_size_str):
    """
    解析窗口大小参数
    
    参数:
        window_size_str: 窗口大小字符串，可以是单个数字或逗号分隔的列表
    
    返回:
        窗口大小列表（整数）
    """
    if ',' in window_size_str:
        return [int(x.strip()) for x in window_size_str.split(',')]
    else:
        return [int(window_size_str)]


def calculate_weight(values1, values2, bin_idx, n_bins, weight_method):
    """
    计算权重：使用当前bin及其相邻bin（共3个bin）的平均值
    
    参数:1   
        values1, values2: 两个数据数组
        bin_idx: 当前bin的索引
        n_bins: 总bin数
        weight_method: 权重计算方法
    
    返回:
        权重值
    """
    start_idx = max(0, bin_idx - 1)
    end_idx = min(n_bins, bin_idx + 2)
    
    local_values1 = values1[start_idx:end_idx]
    local_values2 = values2[start_idx:end_idx]
    
    m1 = np.mean(local_values1)
    m2 = np.mean(local_values2)
    
    if weight_method == 'arithmetic':
        weight = (m1 + m2) / 2
    elif weight_method == 'geometric':
        if m1 >= 0 and m2 >= 0:
            weight = np.sqrt(m1 * m2)
        else:
            weight = 0
    elif weight_method == 'harmonic':
        if m1 > 0 and m2 > 0:
            weight = 2 * m1 * m2 / (m1 + m2)
        else:
            weight = 0
    elif weight_method == 'quadratic':
        weight = np.sqrt((m1**2 + m2**2) / 2)
    elif weight_method == 'minimum':
        weight = min(m1, m2)
    elif weight_method == 'maximum':
        weight = max(m1, m2)
    else:
        raise ValueError(f"未知的权重计算方法: {weight_method}")
    
    return weight


def aggregate_values(values, aggregation):
    """
    根据聚合方法计算聚合值
    
    参数:
        values: 值列表
        aggregation: 聚合方法 ('mean', 'max', 'min', 'median')
    
    返回:
        聚合后的值
    """
    if aggregation == 'mean':
        return np.mean(values)
    elif aggregation == 'max':
        return np.max(values)
    elif aggregation == 'min':
        return np.min(values)
    elif aggregation == 'median':
        return np.median(values)
    else:
        raise ValueError(f"未知的聚合方法: {aggregation}")


def aggregate_diff_values(values, aggregation):
    """
    根据聚合方法计算差异值的聚合（考虑正负号）
    
    参数:
        values: 值列表
        aggregation: 聚合方法 ('mean', 'max', 'min', 'median')
    
    返回:
        聚合后的值
    """
    if aggregation == 'mean':
        return np.mean(values)
    elif aggregation == 'max':
        # 最远离0：绝对值最大
        abs_values = np.abs(values)
        max_idx = np.argmax(abs_values)
        return values[max_idx]
    elif aggregation == 'min':
        # 最靠近0：绝对值最小
        abs_values = np.abs(values)
        min_idx = np.argmin(abs_values)
        return values[min_idx]
    elif aggregation == 'median':
        return np.median(values)
    else:
        raise ValueError(f"未知的聚合方法: {aggregation}")


def create_shared_memory(values1, values2):
    """
    创建共享内存并复制数据
    
    参数:
        values1, values2: numpy数组
    
    返回:
        (shm1, shm2, shared_array1, shared_array2)
    """
    shm1 = shared_memory.SharedMemory(create=True, size=values1.nbytes)
    shm2 = shared_memory.SharedMemory(create=True, size=values2.nbytes)
    
    shared_array1 = np.ndarray(values1.shape, dtype=np.float64, buffer=shm1.buf)
    shared_array2 = np.ndarray(values2.shape, dtype=np.float64, buffer=shm2.buf)
    shared_array1[:] = values1[:]
    shared_array2[:] = values2[:]
    
    return shm1, shm2, shared_array1, shared_array2


def cleanup_shared_memory(shm1, shm2):
    """
    清理共享内存
    
    参数:
        shm1, shm2: 共享内存对象
    """
    shm1.close()
    shm2.close()
    shm1.unlink()
    shm2.unlink()


def write_output(bg1, output_path, result_values, column_name):
    """
    写入输出文件
    
    参数:
        bg1: 原始bedgraph DataFrame
        output_path: 输出文件路径
        result_values: 结果值数组
        column_name: 结果列名
    """
    output_df = bg1[['chrom', 'start', 'end']].copy()
    output_df[column_name] = result_values
    output_df.to_csv(output_path, sep='\t', header=False, index=False)


def print_statistics(result_values, result_name):
    """
    打印统计信息
    
    参数:
        result_values: 结果值数组
        result_name: 结果名称
    """
    print(f"{result_name}统计:")
    print(f"  最小值: {np.min(result_values):.4f}")
    print(f"  最大值: {np.max(result_values):.4f}")
    print(f"  平均值: {np.mean(result_values):.4f}")
    print(f"  中位数: {np.median(result_values):.4f}")
