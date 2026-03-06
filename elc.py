
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


# def chi_square_test(x, y):
#     """卡方-老的"""
#     x = np.abs(x)
#     y = np.abs(y)
    
#     if np.sum(x) == 0 or np.sum(y) == 0:
#         return 1.0
#     if np.std(x) == 0 or np.std(y) == 0:
#         return 1.0
    
#     try:
#         x_median = np.median(x)
#         y_median = np.median(y)
        
#         n11 = np.sum((x >= x_median) & (y >= y_median))
#         n12 = np.sum((x >= x_median) & (y < y_median))
#         n21 = np.sum((x < x_median) & (y >= y_median))
#         n22 = np.sum((x < x_median) & (y < y_median))
        
#         contingency_table = np.array([[n11, n12], [n21, n22]])
        
#         if np.any(contingency_table.sum(axis=0) == 0) or np.any(contingency_table.sum(axis=1) == 0):
#             return 1.0
        
#         chi2, p_value, dof, expected = chi2_contingency(contingency_table)
#         return chi2
#     except:
#         return 1.0

def chi_square_test(x, y):
    """GPT说的：2x2 卡方统计量（与 chi2_contingency 结果一致）"""
    x = np.abs(x)
    y = np.abs(y)

    # 快速零判断
    if not x.any() or not y.any():
        return 1.0

    # 方差为 0 判断
    if x.std() == 0 or y.std() == 0:
        return 1.0

    try:
        x_median = np.median(x)
        y_median = np.median(y)

        x_bin = x >= x_median
        y_bin = y >= y_median

        a = np.sum(x_bin & y_bin)
        b = np.sum(x_bin & ~y_bin)
        c = np.sum(~x_bin & y_bin)
        d = np.sum(~x_bin & ~y_bin)

        # 边际和为 0 检查
        if (a+b)==0 or (c+d)==0 or (a+c)==0 or (b+d)==0:
            return 1.0

        n = a + b + c + d

        # 2x2 卡方公式
        chi2 = n * (a*d - b*c)**2 / (
            (a+b)*(c+d)*(a+c)*(b+d)
        )

        return chi2

    except:
        return 1.0


def simple_ks_statistic(x, y):
    try:
        ks_statistic, p_value = ks_2samp(x, y)
        return 1 - ks_statistic
    except:
        return 0


def calculate_mutual_information(x, y, n_bins=15):
    """纯 numpy"""
    if len(x) == 0 or len(x) != len(y):
        return 0.0

    # 避免无方差情况
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()

    if x_min == x_max or y_min == y_max:
        return 0.0

    # 直接构建二维直方图
    c_xy, _, _ = np.histogram2d(x, y, bins=n_bins)

    total = c_xy.sum()
    if total == 0:
        return 0.0

    p_xy = c_xy / total
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # 只计算非零位置
    nz = p_xy > 0

    return np.sum(
        p_xy[nz] *
        np.log(p_xy[nz] /
               (p_x[:, None] * p_y[None, :])[nz])
    )

def exponential_decay_weights(window_size, half_life):
    """Exponential"""
    decay_constant = np.log(2) / half_life
    distances = np.arange(-window_size, window_size + 1)
    weights = np.exp(-decay_constant * np.abs(distances))
    return weights


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
    Calculate local correlation between two bedgraph files using 
    multiprocessing and shared memory.

    Parameters:
        bedgraph1_path: Path to first bedgraph file
        bedgraph2_path: Path to second bedgraph file
        output_path: Output bedgraph file path
        method: Calculation method 
                ('pearson', 'pearson_exp', 'chi2', 'ks', 'mi')
        window_sizes: List of window sizes (number of bins on one side)
        aggregation: Aggregation method 
                    ('mean', 'max', 'min', 'median')
        weight_method: Weight calculation method
        n_processes: Number of processes (default: CPU core count)
        half_lives: Half-life list (only used for pearson_exp method)
    """

    if n_processes is None:
        n_processes = cpu_count()
    
    print("Reading bedgraph files...")
    bg1 = read_bedgraph(bedgraph1_path)
    bg2 = read_bedgraph(bedgraph2_path)
    
    if len(bg1) != len(bg2):
        raise ValueError(
            f"The two bedgraph files have different numbers of bins: "
            f"{len(bg1)} vs {len(bg2)}"
        )
    
    n_bins = len(bg1)
    print(f"Total number of bins: {n_bins}")
    print(f"Using {n_processes} processes for parallel computation")
    
    values1 = bg1['value'].values.astype(np.float64)
    values2 = bg2['value'].values.astype(np.float64)
    
    print("Creating shared memory...")
    shm1, shm2, shared_array1, shared_array2 = create_shared_memory(values1, values2)
    
    # Method selection
    if method == 'pearson_exp':
        
        half_lives = window_sizes

        window_size = 100
        
        weights_dict = {}
        for half_life in half_lives:
            weights_dict[half_life] = exponential_decay_weights(window_size, half_life)
        
        print(f"Window size: {2*window_size + 1} bins (center ± {window_size} bins)")
        print(f"Half-lives: {half_lives}")
        print(f"Aggregation method: {aggregation}")
        print(f"Weight calculation method: {weight_method}")
        
        chunk_size = max(1, n_bins // n_processes)
        chunks = []
        for i in range(0, n_bins, chunk_size):
            start_idx = i
            end_idx = min(i + chunk_size, n_bins)
            chunks.append((start_idx, end_idx, n_bins, window_size, half_lives,
                          shm1.name, shm2.name, values1.shape,
                          weights_dict, aggregation, weight_method))
        
        process_func = process_chunk_pearson_exp

    else:
        max_window_size = max(window_sizes)
        
        print(f"Window sizes: {window_sizes}")
        print(f"Aggregation method: {aggregation}")
        print(f"Weight calculation method: {weight_method}")
        
        chunk_size = max(1, n_bins // n_processes)
        chunks = []
        for i in range(0, n_bins, chunk_size):
            start_idx = i
            end_idx = min(i + chunk_size, n_bins)
            chunks.append((start_idx, end_idx, n_bins, max_window_size, window_sizes,
                          shm1.name, shm2.name, values1.shape,
                          aggregation, weight_method))
        
        if method == 'pearson':
            process_func = process_chunk_pearson
        elif method == 'chi2':
            process_func = process_chunk_chi2
        elif method == 'ks':
            process_func = process_chunk_ks
        elif method == 'mi':
            process_func = process_chunk_mi
        else:
            raise ValueError(f"Unknown method: {method}")
    
    print(f"Task divided into {len(chunks)} chunks")
    print("Starting parallel computation...")
    
    try:
        with Pool(processes=n_processes) as pool:
            if method == 'mi':
                results = list(tqdm(
                    pool.imap(process_func, chunks),
                    total=len(chunks),
                    desc="Processing",
                    unit="chunk",
                    ncols=80
                ))
            else:
                results = pool.map(process_func, chunks)
        
        print("Merging results...")
        weighted_correlation = np.concatenate(results)
        
    finally:
        print("Cleaning up shared memory...")
        cleanup_shared_memory(shm1, shm2)
    
    print("Computation completed. Writing output file...")
    write_output(bg1, output_path, weighted_correlation, 'weighted_correlation')
    
    print(f"Results saved to: {output_path}")
    print_statistics(weighted_correlation, "Weighted correlation")