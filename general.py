import numpy as np
import pandas as pd
from multiprocessing import shared_memory


def read_bedgraph(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None, 
                     names=['chrom', 'start', 'end', 'value'])
    df['value'] = df['value'].fillna(0)
    return df


def parse_window_sizes(window_size_str):
    if ',' in window_size_str:
        return [int(x.strip()) for x in window_size_str.split(',')]
    else:
        return [int(window_size_str)]


def calculate_weight(values1, values2, bin_idx, n_bins, weight_method):
    """
    计算权重：使用当前bin及其相邻bin（共3个bin）的平均值
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
        raise ValueError(f"Unknown correlation weight calculation method: {weight_method}")
    
    return weight


def aggregate_values(values, aggregation):
    if aggregation == 'mean':
        return np.mean(values)
    elif aggregation == 'max':
        return np.max(values)
    elif aggregation == 'min':
        return np.min(values)
    elif aggregation == 'median':
        return np.median(values)
    else:
        raise ValueError(f"Unknow aggregation method: {aggregation}")


def aggregate_diff_values(values, aggregation):
    """
    根据聚合方法计算差异值的聚合（考虑正负号）
    values: 值列表
    """
    if aggregation == 'mean':
        return np.mean(values)
    elif aggregation == 'max':
        abs_values = np.abs(values)
        max_idx = np.argmax(abs_values)
        return values[max_idx]
    elif aggregation == 'min':
        abs_values = np.abs(values)
        min_idx = np.argmin(abs_values)
        return values[min_idx]
    elif aggregation == 'median':
        return np.median(values)
    else:
        raise ValueError(f"Unknow aggregation method: {aggregation}")


def create_shared_memory(values1, values2):
    """
    创建共享内存并复制数据
    values1, values2: numpy数组
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
    shm1, shm2: 共享内存对象
    """
    shm1.close()
    shm2.close()
    shm1.unlink()
    shm2.unlink()


def write_output(bg1, output_path, result_values, column_name):

    output_df = bg1[['chrom', 'start', 'end']].copy()
    output_df[column_name] = result_values
    output_df.to_csv(output_path, sep='\t', header=False, index=False)


def print_statistics(result_values, result_name):
    print(f"{result_name} have been PROCEED! --")
    print(f"  min value: {np.min(result_values):.4f}")
    print(f"  max value: {np.max(result_values):.4f}")
    print(f"  mean: {np.mean(result_values):.4f}")
    print(f"  median: {np.median(result_values):.4f}")
