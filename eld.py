"""
ELD (Epigenomic Local Difference) 模块
整合所有差异显著性计算方法
"""

import numpy as np
from scipy.stats import binomtest, poisson, norm
from statsmodels.stats.proportion import proportions_ztest
from multiprocessing import Pool, cpu_count, shared_memory

from general import (
    read_bedgraph, create_shared_memory, cleanup_shared_memory,
    write_output, print_statistics, aggregate_diff_values
)


# ==================== 权重计算函数 ====================

def calculate_weight_eld(win_count, bg_count):
    """
    计算ELD权重（基于泊松分布）
    
    参数:
        win_count: 窗口内的count
        bg_count: 背景区域的count
    
    返回:
        权重值
    """
    n = win_count + bg_count
    lam = bg_count / 5000
    
    if n > lam:
        p_value = poisson.sf(win_count - 1, mu=lam)
    else:
        p_value = poisson.cdf(win_count, mu=lam)
    
    weight = 1.0 - p_value
    return max(0.0, min(1.0, weight))


# ==================== 差异计算函数 ====================

def calculate_binomial_pvalue(x1, x2, total_count1, total_count2):
    """
    使用proportions_ztest计算p值（Binomial方法）
    
    参数:
        x1: 第一个文件窗口内的观测值总和
        x2: 第二个文件窗口内的观测值总和
        total_count1: 第一个track的全局总count数
        total_count2: 第二个track的全局总count数
    
    返回:
        差异值（基于p值的对数变换）
    """
    if x1 + x2 == 0:
        return 0
    
    if total_count1 == 0 or total_count2 == 0:
        return 0
    
    try:
        count = np.array([x1, x2])
        nobs = np.array([total_count1, total_count2])
        
        z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
        
        if p_value == 0:
            p_value = 1e-300
        
        diff_value = np.log10(p_value)
        if x1 > x2:
            diff_value = -diff_value
        return diff_value
    except:
        return 0


def calculate_poisson_pvalue(x1, x2, T1=1, T2=1):
    """
    使用二项检验计算p值（Poisson方法）
    
    参数:
        x1: 第一个文件的观测值总和
        x2: 第二个文件的观测值总和
        T1: 第一个文件的exposure（默认为1）
        T2: 第二个文件的exposure（默认为1）
    
    返回:
        差异值
    """
    n = x1 + x2
    
    if n == 0:
        return 0
    
    p = T1 / (T1 + T2)
    
    try:
        res1 = binomtest(int(x1), n=int(n), p=p, alternative="greater")
        pvalue1 = res1.pvalue
        
        if pvalue1 == 0:
            return 300
        
        res2 = binomtest(int(x1), n=int(n), p=p, alternative="less")
        pvalue2 = res2.pvalue
        if pvalue2 == 0:
            return 300
        
        diff_value = (np.log10(pvalue1)) + (-np.log10(pvalue2))
        
        return diff_value
    except:
        return 0


def calculate_negbinomial_pvalue(window_values1, bg_values1, window_values2, bg_values2):
    """
    使用负二项分布和Wald检验计算p值
    
    参数:
        window_values1: 窗口内第一个track的每-bin值
        bg_values1: 对应box（背景）中每-bin值
        window_values2, bg_values2: 同上，但用于第二个track
    
    返回:
        差异值
    """
    try:
        n1 = len(window_values1)
        n2 = len(window_values2)
        
        if n1 == 0 or n2 == 0:
            return 1.0
        
        mu1 = np.mean(window_values1)
        mu2 = np.mean(window_values2)
        
        if mu1 <= 0 or mu2 <= 0:
            return 1.0
        
        if len(bg_values1) > 1:
            var_bg1 = np.var(bg_values1, ddof=1)
        elif len(bg_values1) == 1:
            var_bg1 = 0.0
        else:
            var_bg1 = 0.0
        
        if len(bg_values2) > 1:
            var_bg2 = np.var(bg_values2, ddof=1)
        elif len(bg_values2) == 1:
            var_bg2 = 0.0
        else:
            var_bg2 = 0.0
        
        alpha1 = (var_bg1 - mu1) / (mu1 ** 2) if mu1 > 0 else 0.0
        alpha2 = (var_bg2 - mu2) / (mu2 ** 2) if mu2 > 0 else 0.0
        
        alpha1 = max(0.0, alpha1)
        alpha2 = max(0.0, alpha2)
        
        var_log_mu1 = (1.0 / n1) * (1.0 / mu1 + alpha1)
        var_log_mu2 = (1.0 / n2) * (1.0 / mu2 + alpha2)
        
        denom = var_log_mu1 + var_log_mu2
        if denom <= 0:
            return 1.0
        
        z = (np.log(mu1) - np.log(mu2)) / np.sqrt(denom)
        p_value = 2.0 * norm.sf(abs(z))
        
        p_value = max(min(p_value, 1.0), 1e-300)
        
        diff_value = np.log10(p_value)
        if mu1 > mu2:
            diff_value = -diff_value
        return diff_value
    except:
        return 1.0


def calculate_zinb_pvalue(window_values1, bg_values1, window_values2, bg_values2):
    """
    使用ZINB（零膨胀负二项）模型和Wald检验计算p值
    
    参数:
        window_values1: 窗口内第一个track的每-bin值
        bg_values1: 对应box（背景）中每-bin值
        window_values2, bg_values2: 同上，但用于第二个track
    
    返回:
        差异值
    """
    try:
        n1 = len(window_values1)
        n2 = len(window_values2)
        
        if n1 == 0 or n2 == 0:
            return 1.0
        
        mu1 = np.mean(window_values1)
        mu2 = np.mean(window_values2)
        
        if mu1 <= 0 or mu2 <= 0:
            return 1.0
        
        if len(bg_values1) > 1:
            var_bg1 = np.var(bg_values1, ddof=1)
        else:
            var_bg1 = 0.0
        
        if len(bg_values2) > 1:
            var_bg2 = np.var(bg_values2, ddof=1)
        else:
            var_bg2 = 0.0
        
        alpha1 = (var_bg1 - mu1) / (mu1 ** 2) if mu1 > 0 else 0.0
        alpha2 = (var_bg2 - mu2) / (mu2 ** 2) if mu2 > 0 else 0.0
        
        alpha1 = max(0.0, alpha1)
        alpha2 = max(0.0, alpha2)
        
        pi1 = np.mean(bg_values1 == 0) if len(bg_values1) > 0 else 0.0
        pi2 = np.mean(bg_values2 == 0) if len(bg_values2) > 0 else 0.0
        
        n_eff1 = max(1.0, n1 * (1.0 - pi1))
        n_eff2 = max(1.0, n2 * (1.0 - pi2))
        
        var_log_mu1 = (1.0 / n_eff1) * (1.0 / mu1 + alpha1)
        var_log_mu2 = (1.0 / n_eff2) * (1.0 / mu2 + alpha2)
        
        denom = var_log_mu1 + var_log_mu2
        if denom <= 0:
            return 1.0
        
        z = (np.log(mu1) - np.log(mu2)) / np.sqrt(denom)
        p_value = 2.0 * norm.sf(abs(z))
        
        p_value = max(min(p_value, 1.0), 1e-300)
        
        diff_value = np.log10(p_value)
        if mu1 > mu2:
            diff_value = -diff_value
        
        return diff_value
    except:
        return 0


# ==================== 多进程处理函数 ====================

def process_chunk_binomial(args):
    """处理Binomial方法的数据块"""
    (start_idx, end_idx, n_bins, window_sizes,
     shm_name1, shm_name2, shm_shape, aggregation,
     total_count1, total_count2) = args
    
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    chunk_results = np.zeros(end_idx - start_idx)
    
    bg_values1 = np.zeros(1)
    bg_values2 = np.zeros(1)
    bg_count1 = 0
    bg_count2 = 0
    
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx
        
        if local_idx % 5000 == 0:
            bg_values1 = values1[max(0, i - 2500):min(n_bins, i + 2500)]
            bg_values2 = values2[max(0, i - 2500):min(n_bins, i + 2500)]
            bg_count1 = np.sum(bg_values1)
            bg_count2 = np.sum(bg_values2)
        
        center_value1 = values1[i]
        center_value2 = values2[i]
        
        weight1 = calculate_weight_eld(center_value1, bg_count1)
        weight2 = calculate_weight_eld(center_value2, bg_count2)
        final_weight = max(weight1, weight2)
        
        pvalues_for_windows = []
        for window_size in window_sizes:
            win_start = max(0, i - window_size)
            win_end = min(n_bins, i + window_size + 1)
            
            window_values1 = values1[win_start:win_end]
            window_values2 = values2[win_start:win_end]
            
            x1 = np.sum(window_values1)
            x2 = np.sum(window_values2)
            
            diff_value = calculate_binomial_pvalue(x1, x2, total_count1, total_count2)
            weighted_value = diff_value * final_weight
            pvalues_for_windows.append(weighted_value)
        
        chunk_results[local_idx] = aggregate_diff_values(pvalues_for_windows, aggregation)
    
    shm1.close()
    shm2.close()
    
    return chunk_results


def process_chunk_poisson(args):
    """处理Poisson方法的数据块"""
    (start_idx, end_idx, n_bins, window_sizes,
     shm_name1, shm_name2, shm_shape, aggregation,
     total_count1, total_count2) = args
    
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    chunk_results = np.zeros(end_idx - start_idx)
    
    bg_values1 = np.zeros(1)
    bg_values2 = np.zeros(1)
    bg_count1 = 0
    bg_count2 = 0
    
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx
        
        if local_idx % 5000 == 0:
            bg_values1 = values1[max(0, i - 2500):min(n_bins, i + 2500)]
            bg_values2 = values2[max(0, i - 2500):min(n_bins, i + 2500)]
            bg_count1 = np.sum(bg_values1)
            bg_count2 = np.sum(bg_values2)
        
        center_value1 = values1[i]
        center_value2 = values2[i]
        
        weight1 = calculate_weight_eld(center_value1, bg_count1)
        weight2 = calculate_weight_eld(center_value2, bg_count2)
        final_weight = max(weight1, weight2)
        
        pvalues_for_windows = []
        for window_size in window_sizes:
            win_start = max(0, i - window_size)
            win_end = min(n_bins, i + window_size + 1)
            
            window_values1 = values1[win_start:win_end]
            window_values2 = values2[win_start:win_end]
            
            x1 = np.sum(window_values1)
            x2 = np.sum(window_values2)
            
            diff_value = calculate_poisson_pvalue(x1, x2, total_count1, total_count2)
            weighted_value = diff_value * final_weight
            pvalues_for_windows.append(weighted_value)
        
        chunk_results[local_idx] = aggregate_diff_values(pvalues_for_windows, aggregation)
    
    shm1.close()
    shm2.close()
    
    return chunk_results


def process_chunk_negbinomial(args):
    """处理NegBinomial方法的数据块"""
    (start_idx, end_idx, n_bins, window_sizes,
     shm_name1, shm_name2, shm_shape, aggregation,
     total_count1, total_count2) = args
    
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    chunk_results = np.zeros(end_idx - start_idx)
    
    bg_values1 = np.zeros(1)
    bg_values2 = np.zeros(1)
    bg_count1 = 0
    bg_count2 = 0
    
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx
        
        if local_idx % 5000 == 0:
            bg_values1 = values1[max(0, i - 2500):min(n_bins, i + 2500)]
            bg_values2 = values2[max(0, i - 2500):min(n_bins, i + 2500)]
            bg_count1 = np.sum(bg_values1)
            bg_count2 = np.sum(bg_values2)
        
        center_value1 = values1[i]
        center_value2 = values2[i]
        
        weight1 = calculate_weight_eld(center_value1, bg_count1)
        weight2 = calculate_weight_eld(center_value2, bg_count2)
        final_weight = max(weight1, weight2)
        
        pvalues_for_windows = []
        for window_size in window_sizes:
            win_start = max(0, i - window_size)
            win_end = min(n_bins, i + window_size + 1)
            
            window_values1 = values1[win_start:win_end]
            window_values2 = values2[win_start:win_end]
            
            diff_value = calculate_negbinomial_pvalue(window_values1, bg_values1, window_values2, bg_values2)
            weighted_value = diff_value * final_weight
            pvalues_for_windows.append(weighted_value)
        
        chunk_results[local_idx] = aggregate_diff_values(pvalues_for_windows, aggregation)
    
    shm1.close()
    shm2.close()
    
    return chunk_results


def process_chunk_zinb(args):
    """处理ZINB方法的数据块"""
    (start_idx, end_idx, n_bins, window_sizes,
     shm_name1, shm_name2, shm_shape, aggregation,
     total_count1, total_count2) = args
    
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    chunk_results = np.zeros(end_idx - start_idx)
    
    bg_values1 = np.zeros(1)
    bg_values2 = np.zeros(1)
    bg_count1 = 0
    bg_count2 = 0
    
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx
        
        if local_idx % 5000 == 0:
            bg_values1 = values1[max(0, i - 2500):min(n_bins, i + 2500)]
            bg_values2 = values2[max(0, i - 2500):min(n_bins, i + 2500)]
            bg_count1 = np.sum(bg_values1)
            bg_count2 = np.sum(bg_values2)
        
        center_value1 = values1[i]
        center_value2 = values2[i]
        
        weight1 = calculate_weight_eld(center_value1, bg_count1)
        weight2 = calculate_weight_eld(center_value2, bg_count2)
        final_weight = max(weight1, weight2)
        
        pvalues_for_windows = []
        for window_size in window_sizes:
            win_start = max(0, i - window_size)
            win_end = min(n_bins, i + window_size + 1)
            
            window_values1 = values1[win_start:win_end]
            window_values2 = values2[win_start:win_end]
            
            diff_value = calculate_zinb_pvalue(window_values1, bg_values1, window_values2, bg_values2)
            weighted_value = diff_value * final_weight
            pvalues_for_windows.append(weighted_value)
        
        chunk_results[local_idx] = aggregate_diff_values(pvalues_for_windows, aggregation)
    
    shm1.close()
    shm2.close()
    
    return chunk_results


# ==================== 主计算函数 ====================

def calculate_local_difference_parallel(bedgraph1_path, bedgraph2_path, 
                                       output_path, method='binomial',
                                       window_sizes=[100],
                                       aggregation='mean',
                                       n_processes=None):
    """
    使用多进程和共享内存计算两个bedgraph文件的局部差异显著性
    
    参数:
        bedgraph1_path: 第一个bedgraph文件路径
        bedgraph2_path: 第二个bedgraph文件路径
        output_path: 输出bedgraph文件路径
        method: 计算方法 ('binomial', 'poisson', 'negbinomial', 'zinb')
        window_sizes: 窗口大小列表（单侧bin数，默认[100]）
        aggregation: 聚合方法 ('mean', 'max', 'min', 'median')
        n_processes: 进程数（默认为CPU核心数）
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
    
    total_count1 = np.sum(values1)
    total_count2 = np.sum(values2)
    
    print(f"Track1 全局总count: {total_count1:.0f}")
    print(f"Track2 全局总count: {total_count2:.0f}")
    
    print("创建共享内存...")
    shm1, shm2, shared_array1, shared_array2 = create_shared_memory(values1, values2)
    
    print(f"窗口大小: {window_sizes}")
    print(f"背景区域大小: {[ws * 10 for ws in window_sizes]}")
    print(f"聚合方法: {aggregation}")
    
    chunk_size = max(1, n_bins // n_processes)
    chunks = []
    for i in range(0, n_bins, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, n_bins)
        chunks.append((start_idx, end_idx, n_bins, window_sizes,
                      shm1.name, shm2.name, values1.shape, aggregation,
                      total_count1, total_count2))
    
    if method == 'binomial':
        process_func = process_chunk_binomial
    elif method == 'poisson':
        process_func = process_chunk_poisson
    elif method == 'negbinomial':
        process_func = process_chunk_negbinomial
    elif method == 'zinb':
        process_func = process_chunk_zinb
    else:
        raise ValueError(f"未知的方法: {method}")
    
    print(f"任务分割为 {len(chunks)} 个块")
    print("开始并行计算（使用权重计算）...")
    
    try:
        with Pool(processes=n_processes) as pool:
            results = pool.map(process_func, chunks)
        
        print("合并结果...")
        weighted_diff_values = np.concatenate(results)
        
    finally:
        print("清理共享内存...")
        cleanup_shared_memory(shm1, shm2)
    
    print("计算完成，写入输出文件...")
    write_output(bg1, output_path, weighted_diff_values, 'weighted_diff_value')
    
    print(f"结果已保存到: {output_path}")
    print_statistics(weighted_diff_values, "加权差异值")
