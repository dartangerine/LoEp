import numpy as np
from scipy.stats import binomtest, poisson, norm
from statsmodels.stats.proportion import proportions_ztest
from multiprocessing import Pool, cpu_count, shared_memory

from general import (
    read_bedgraph, create_shared_memory, cleanup_shared_memory,
    write_output, print_statistics, aggregate_diff_values
)


def calculate_weight_eld(win_count, bg_count, difference_weight):
    """
    计算ELD权重 泊松分布
    """

    if difference_weight == 'none':
        return 1.0
    n = win_count + bg_count
    lam = bg_count / 5000
                                                                                                                                                                                                                                                                                                                                                                                           
    if n > lam:
        p_value = poisson.sf(win_count - 1, mu=lam)
    else:
        p_value = poisson.cdf(win_count, mu=lam)
    
    if difference_weight == 'p':
        weight = 1.0 - p_value
    elif difference_weight == 'logp':
        weight = -np.log10(p_value) if p_value > 0 else 300
    else:
        raise ValueError(f"Unknown difference_weight: {difference_weight}")

    return max(0.0, min(1.0, weight))


def calculate_binomial_pvalue(x1, x2, total_count1, total_count2):
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
     total_count1, total_count2, difference_weight) = args
    
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
        
        weight1 = calculate_weight_eld(center_value1, bg_count1, difference_weight)
        weight2 = calculate_weight_eld(center_value2, bg_count2, difference_weight)
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
     total_count1, total_count2, difference_weight) = args
    
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
        
        weight1 = calculate_weight_eld(center_value1, bg_count1, difference_weight)
        weight2 = calculate_weight_eld(center_value2, bg_count2, difference_weight)
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
     total_count1, total_count2, difference_weight) = args
    
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
        
        weight1 = calculate_weight_eld(center_value1, bg_count1, difference_weight)
        weight2 = calculate_weight_eld(center_value2, bg_count2, difference_weight)
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
     total_count1, total_count2, difference_weight) = args
    
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
        
        weight1 = calculate_weight_eld(center_value1, bg_count1, difference_weight)
        weight2 = calculate_weight_eld(center_value2, bg_count2, difference_weight)
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


def calculate_local_difference_parallel(bedgraph1_path, bedgraph2_path,
                                       output_path, method='binomial',
                                       window_sizes=[3,5,7,10],
                                       aggregation='mean',
                                       difference_weight='logp',
                                       n_processes=None,
                                       ):
    """
    ELD的主计算函数
    """
    if n_processes is None:
        n_processes = cpu_count()

    print("Reading bedgraph files...")
    bg1 = read_bedgraph(bedgraph1_path)
    bg2 = read_bedgraph(bedgraph2_path)

    if len(bg1) != len(bg2):
        raise ValueError(f"The two bedgraph files have different numbers of bins: {len(bg1)} vs {len(bg2)}")

    n_bins = len(bg1)
    print(f"Total number of bins: {n_bins}")
    print(f"Using {n_processes} processes for parallel computation")

    values1 = bg1['value'].values.astype(np.float64)
    values2 = bg2['value'].values.astype(np.float64)

    total_count1 = np.sum(values1)
    total_count2 = np.sum(values2)

    print(f"Track1 total global count: {total_count1:.0f}")
    print(f"Track2 total global count: {total_count2:.0f}")

    print("Creating shared memory...")
    shm1, shm2, shared_array1, shared_array2 = create_shared_memory(values1, values2)

    print(f"Window sizes: {window_sizes}")
    print(f"Background region sizes: {[ws * 10 for ws in window_sizes]}")
    print(f"Aggregation method: {aggregation}")

    chunk_size = max(1, n_bins // n_processes)
    chunks = []
    for i in range(0, n_bins, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, n_bins)
        chunks.append((start_idx, end_idx, n_bins, window_sizes,
                    shm1.name, shm2.name, values1.shape, aggregation,
                    total_count1, total_count2, difference_weight))

    if method == 'binomial':
        process_func = process_chunk_binomial
    elif method == 'poisson':
        process_func = process_chunk_poisson
    elif method == 'negbinomial':
        process_func = process_chunk_negbinomial
    elif method == 'zinb':
        process_func = process_chunk_zinb
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Task divided into {len(chunks)} chunks")
    print("Starting parallel computation (with weighted calculation)...")

    try:
        with Pool(processes=n_processes) as pool:
            results = pool.map(process_func, chunks)

        print("Merging results...")
        weighted_diff_values = np.concatenate(results)

    finally:
        print("Cleaning up shared memory...")
        cleanup_shared_memory(shm1, shm2)

    print("Computation completed. Writing output file...")
    write_output(bg1, output_path, weighted_diff_values, 'weighted_diff_value')

    print(f"Results saved to: {output_path}")
    print_statistics(weighted_diff_values, "Weighted difference values")