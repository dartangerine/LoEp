"""
ELC (Epigenomic Local Correlation) 模块
整合所有相关性计算方法
"""

import numpy as np
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import mutual_info_score
from multiprocessing import Pool, cpu_count, shared_memory
from tqdm import tqdm

from general import (
    read_bedgraph, create_shared_memory, cleanup_shared_memory,
    write_output, print_statistics, calculate_weight, aggregate_values
)


# ==================== 相关性计算函数 ====================

def simple_pearson_correlation(x, y):
    """计算简单Pearson相关系数"""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator_x = np.sqrt(np.sum((x - x_mean)**2))
    denominator_y = np.sqrt(np.sum((y - y_mean)**2))
    
    if denominator_x == 0 or denominator_y == 0:
        return 0
    
    correlation = numerator / (denominator_x * denominator_y)
    return abs(correlation)


def weighted_pearson_correlation(x, y, weights):
    """计算加权Pearson相关系数（用于指数衰减）"""
    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)
    
    numerator = np.sum(weights * (x - x_mean) * (y - y_mean))
    denominator_x = np.sqrt(np.sum(weights * (x - x_mean)**2))
    denominator_y = np.sqrt(np.sum(weights * (y - y_mean)**2))
    
    if denominator_x == 0 or denominator_y == 0:
        return 0
    
    correlation = numerator / (denominator_x * denominator_y)
    return abs(correlation)


def chi_square_test(x, y):
    """计算卡方检验的统计量"""
    x = np.abs(x)
    y = np.abs(y)
    
    if np.sum(x) == 0 or np.sum(y) == 0:
        return 1.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 1.0
    
    try:
        x_median = np.median(x)
        y_median = np.median(y)
        
        n11 = np.sum((x >= x_median) & (y >= y_median))
        n12 = np.sum((x >= x_median) & (y < y_median))
        n21 = np.sum((x < x_median) & (y >= y_median))
        n22 = np.sum((x < x_median) & (y < y_median))
        
        contingency_table = np.array([[n11, n12], [n21, n22]])
        
        if np.any(contingency_table.sum(axis=0) == 0) or np.any(contingency_table.sum(axis=1) == 0):
            return 1.0
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return chi2
    except:
        return 1.0


def simple_ks_statistic(x, y):
    """计算KS检验统计量，返回(1 - D)"""
    try:
        ks_statistic, p_value = ks_2samp(x, y)
        return 1 - ks_statistic
    except:
        return 0


def calculate_mutual_information(x, y):
    """计算互信息（优化版）"""
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return 0
    
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0
    
    try:
        n_samples = len(x)
        n_bins = max(2, min(int(np.sqrt(n_samples)), 50))
        
        x_edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
        y_edges = np.linspace(np.min(y), np.max(y), n_bins + 1)
        
        x_discrete = np.digitize(x, x_edges) - 1
        y_discrete = np.digitize(y, y_edges) - 1
        
        x_discrete = np.clip(x_discrete, 0, n_bins - 1)
        y_discrete = np.clip(y_discrete, 0, n_bins - 1)
        
        mi = mutual_info_score(x_discrete, y_discrete)
        return mi
    except:
        return 0


def exponential_decay_weights(window_size, half_life):
    """计算指数衰减权重"""
    decay_constant = np.log(2) / half_life
    distances = np.arange(-window_size, window_size + 1)
    weights = np.exp(-decay_constant * np.abs(distances))
    return weights


# ==================== 多进程处理函数 ====================

def process_chunk_pearson(args):
    """处理Pearson相关性计算的数据块"""
    (start_idx, end_idx, n_bins, max_window_size, window_sizes,
     shm_name1, shm_name2, shm_shape, aggregation, weight_method) = args
    
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    chunk_results = np.zeros(end_idx - start_idx)
    
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx
        
        correlations = []
        for window_size in window_sizes:
            win_start = max(0, i - window_size)
            win_end = min(n_bins, i + window_size + 1)
            
            window_values1 = values1[win_start:win_end]
            window_values2 = values2[win_start:win_end]
            
            correlation = simple_pearson_correlation(window_values1, window_values2)
            correlations.append(correlation)
        
        aggregated_correlation = aggregate_values(correlations, aggregation)
        weight = calculate_weight(values1, values2, i, n_bins, weight_method)
        chunk_results[local_idx] = aggregated_correlation * weight
    
    shm1.close()
    shm2.close()
    
    return chunk_results


def process_chunk_pearson_exp(args):
    """处理Pearson指数衰减相关性计算的数据块"""
    (start_idx, end_idx, n_bins, window_size, half_lives,
     shm_name1, shm_name2, shm_shape, weights_dict, aggregation, weight_method) = args
    
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    chunk_results = np.zeros(end_idx - start_idx)
    
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx
        
        win_start = max(0, i - window_size)
        win_end = min(n_bins, i + window_size + 1)
        
        window_values1 = values1[win_start:win_end]
        window_values2 = values2[win_start:win_end]
        
        correlations = []
        for half_life in half_lives:
            weights_array = weights_dict[half_life]
            
            if i < window_size:
                window_weights = weights_array[window_size - i:]
            elif i >= n_bins - window_size:
                window_weights = weights_array[:window_size + (n_bins - i)]
            else:
                window_weights = weights_array
            
            correlation = weighted_pearson_correlation(window_values1, window_values2, window_weights)
            correlations.append(correlation)
        
        aggregated_correlation = aggregate_values(correlations, aggregation)
        weight = calculate_weight(values1, values2, i, n_bins, weight_method)
        chunk_results[local_idx] = aggregated_correlation * weight
    
    shm1.close()
    shm2.close()
    
    return chunk_results


def process_chunk_chi2(args):
    """处理卡方检验的数据块"""
    (start_idx, end_idx, n_bins, max_window_size, window_sizes,
     shm_name1, shm_name2, shm_shape, aggregation, weight_method) = args
    
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    chunk_results = np.zeros(end_idx - start_idx)
    
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx
        
        p_values = []
        for window_size in window_sizes:
            win_start = max(0, i - window_size)
            win_end = min(n_bins, i + window_size + 1)
            
            window_values1 = values1[win_start:win_end]
            window_values2 = values2[win_start:win_end]
            
            p_value = chi_square_test(window_values1, window_values2)
            p_values.append(p_value)
        
        aggregated_p_value = aggregate_values(p_values, aggregation)
        weight = calculate_weight(values1, values2, i, n_bins, weight_method)
        chunk_results[local_idx] = aggregated_p_value * weight
    
    shm1.close()
    shm2.close()
    
    return chunk_results


def process_chunk_ks(args):
    """处理KS检验的数据块"""
    (start_idx, end_idx, n_bins, max_window_size, window_sizes,
     shm_name1, shm_name2, shm_shape, aggregation, weight_method) = args
    
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    chunk_results = np.zeros(end_idx - start_idx)
    
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx
        
        ks_values = []
        for window_size in window_sizes:
            win_start = max(0, i - window_size)
            win_end = min(n_bins, i + window_size + 1)
            
            window_values1 = values1[win_start:win_end]
            window_values2 = values2[win_start:win_end]
            
            one_minus_d = simple_ks_statistic(window_values1, window_values2)
            ks_values.append(one_minus_d)
        
        aggregated_ks = aggregate_values(ks_values, aggregation)
        weight = calculate_weight(values1, values2, i, n_bins, weight_method)
        chunk_results[local_idx] = aggregated_ks * weight
    
    shm1.close()
    shm2.close()
    
    return chunk_results


def process_chunk_mi(args):
    """处理互信息计算的数据块"""
    (start_idx, end_idx, n_bins, max_window_size, window_sizes,
     shm_name1, shm_name2, shm_shape, aggregation, weight_method) = args
    
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    chunk_results = np.zeros(end_idx - start_idx)
    
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx
        
        mutual_informations = []
        for window_size in window_sizes:
            win_start = max(0, i - window_size)
            win_end = min(n_bins, i + window_size + 1)
            
            window_values1 = values1[win_start:win_end]
            window_values2 = values2[win_start:win_end]
            
            mi = calculate_mutual_information(window_values1, window_values2)
            mutual_informations.append(mi)
        
        aggregated_mi = aggregate_values(mutual_informations, aggregation)
        weight = calculate_weight(values1, values2, i, n_bins, weight_method)
        chunk_results[local_idx] = aggregated_mi * weight
    
    shm1.close()
    shm2.close()
    
    return chunk_results


# ==================== 主计算函数 ====================

def calculate_local_correlation_parallel(bedgraph1_path, bedgraph2_path, 
                                        output_path, method='pearson',
                                        window_sizes=[100], 
                                        aggregation='mean',
                                        weight_method='arithmetic',
                                        n_processes=None,
                                        half_lives=None):
    """
    使用多进程和共享内存计算两个bedgraph文件的局部相关性
    
    参数:
        bedgraph1_path: 第一个bedgraph文件路径
        bedgraph2_path: 第二个bedgraph文件路径
        output_path: 输出bedgraph文件路径
        method: 计算方法 ('pearson', 'pearson_exp', 'chi2', 'ks', 'mi')
        window_sizes: 窗口大小列表（单侧bin数，默认[100]）
        aggregation: 聚合方法 ('mean', 'max', 'min', 'median')
        weight_method: 权重计算方法
        n_processes: 进程数（默认为CPU核心数）
        half_lives: 半衰期列表（仅用于pearson_exp方法）
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"读取bedgraph文件...")
    bg1 = read_bedgraph(bedgraph1_path)
    bg2 = read_bedgraph(bedgraph2_path)
    
    if len(bg1) != len(bg2):
        raise ValueError(f"两个bedgraph文件的bin数不一致: {len(bg1)} vs {len(bg2)}")
    
    n_bins = len(bg1)
    print(f"总共 {n_bins} 个bins")
    print(f"使用 {n_processes} 个进程进行并行计算")
    
    values1 = bg1['value'].values.astype(np.float64)
    values2 = bg2['value'].values.astype(np.float64)
    
    print("创建共享内存...")
    shm1, shm2, shared_array1, shared_array2 = create_shared_memory(values1, values2)
    
    # 根据方法选择处理函数和参数
    if method == 'pearson_exp':
        if half_lives is None:
            half_lives = [5]
        window_size = window_sizes[0] if isinstance(window_sizes, list) else window_sizes
        
        weights_dict = {}
        for half_life in half_lives:
            weights_dict[half_life] = exponential_decay_weights(window_size, half_life)
        
        print(f"窗口大小: {2*window_size + 1} bins (中心 ± {window_size} bins)")
        print(f"半衰期: {half_lives}")
        print(f"聚合方法: {aggregation}")
        print(f"权重计算方法: {weight_method}")
        
        chunk_size = max(1, n_bins // n_processes)
        chunks = []
        for i in range(0, n_bins, chunk_size):
            start_idx = i
            end_idx = min(i + chunk_size, n_bins)
            chunks.append((start_idx, end_idx, n_bins, window_size, half_lives,
                          shm1.name, shm2.name, values1.shape, weights_dict, aggregation, weight_method))
        
        process_func = process_chunk_pearson_exp
    else:
        max_window_size = max(window_sizes)
        
        print(f"窗口大小: {window_sizes}")
        print(f"聚合方法: {aggregation}")
        print(f"权重计算方法: {weight_method}")
        
        chunk_size = max(1, n_bins // n_processes)
        chunks = []
        for i in range(0, n_bins, chunk_size):
            start_idx = i
            end_idx = min(i + chunk_size, n_bins)
            chunks.append((start_idx, end_idx, n_bins, max_window_size, window_sizes,
                          shm1.name, shm2.name, values1.shape, aggregation, weight_method))
        
        if method == 'pearson':
            process_func = process_chunk_pearson
        elif method == 'chi2':
            process_func = process_chunk_chi2
        elif method == 'ks':
            process_func = process_chunk_ks
        elif method == 'mi':
            process_func = process_chunk_mi
        else:
            raise ValueError(f"未知的方法: {method}")
    
    print(f"任务分割为 {len(chunks)} 个块")
    print("开始并行计算...")
    
    try:
        with Pool(processes=n_processes) as pool:
            if method == 'mi':
                results = list(tqdm(
                    pool.imap(process_func, chunks),
                    total=len(chunks),
                    desc="计算进度",
                    unit="块",
                    ncols=80
                ))
            else:
                results = pool.map(process_func, chunks)
        
        print("合并结果...")
        weighted_correlation = np.concatenate(results)
        
    finally:
        print("清理共享内存...")
        cleanup_shared_memory(shm1, shm2)
    
    print("计算完成，写入输出文件...")
    write_output(bg1, output_path, weighted_correlation, 'weighted_correlation')
    
    print(f"结果已保存到: {output_path}")
    print_statistics(weighted_correlation, "加权相关性")
