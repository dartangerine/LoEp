#!/usr/bin/env python3
"""
计算两个bedgraph文件的局部加权Pearson相关性
使用滑动窗口、指数衰减权重、多进程和共享内存优化
支持多个半衰期和聚合方法
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, shared_memory
import sys
import os

def read_bedgraph(filepath):
    """读取bedgraph文件并返回DataFrame，NaN填充为0"""
    df = pd.read_csv(filepath, sep='\t', header=None, 
                     names=['chrom', 'start', 'end', 'value'])
    # 将NaN值填充为0
    df['value'] = df['value'].fillna(0)
    return df

def exponential_decay_weights(window_size, half_life):
    """
    计算指数衰减权重
    
    参数:
        window_size: 窗口大小（单侧，不包括中心）
        half_life: 半衰期（bin数）
    
    返回:
        权重数组，长度为 2*window_size + 1
    """
    decay_constant = np.log(2) / half_life
    distances = np.arange(-window_size, window_size + 1)
    weights = np.exp(-decay_constant * np.abs(distances))
    return weights

def weighted_pearson_correlation(x, y, weights):
    """
    计算加权Pearson相关系数
    
    参数:
        x, y: 两个数据数  组
        weights: 权重数组
    
    返回:
        加权Pearson相关系数，如果无法计算则返回0
    """
    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)
    
    numerator = np.sum(weights * (x - x_mean) * (y - y_mean))
    denominator_x = np.sqrt(np.sum(weights * (x - x_mean)**2))
    denominator_y = np.sqrt(np.sum(weights * (y - y_mean)**2))
    
    if denominator_x == 0 or denominator_y == 0:
        return 0  # 返回0而不是NaN
    
    correlation = numerator / (denominator_x * denominator_y)
    correlation = abs(correlation)
    return correlation

def calculate_weight(values1, values2, bin_idx, n_bins, weight_method):
    """
    计算权重：使用当前bin及其相邻bin（共3个bin）的平均值
    
    参数:
        values1, values2: 两个数据数组
        bin_idx: 当前bin的索引
        n_bins: 总bin数
        weight_method: 权重计算方法 ('arithmetic', 'geometric', 'harmonic', 'quadratic')
    
    返回:
        权重值
    """
    # 确定3个bin的范围（当前bin及其相邻bin）
    start_idx = max(0, bin_idx - 1)
    end_idx = min(n_bins, bin_idx + 2)
    
    # 提取3个bin的数值
    local_values1 = values1[start_idx:end_idx]
    local_values2 = values2[start_idx:end_idx]
    
    # 计算两个文件的平均值
    m1 = np.mean(local_values1)
    m2 = np.mean(local_values2)
    
    # 根据方法计算权重
    if weight_method == 'arithmetic':
        # 算术平均数
        weight = (m1 + m2) / 2
    elif weight_method == 'geometric':
        # 几何平均数
        if m1 >= 0 and m2 >= 0:
            weight = np.sqrt(m1 * m2)
        else:
            weight = 0  # 如果有负值，几何平均数无意义
    elif weight_method == 'harmonic':
        # 调和平均数
        if m1 > 0 and m2 > 0:
            weight = 2 * m1 * m2 / (m1 + m2)
        else:
            weight = 0  # 如果有0或负值，调和平均数无意义
    elif weight_method == 'quadratic':
        # 平方平均数（均方根）
        weight = np.sqrt((m1**2 + m2**2) / 2)
    elif weight_method == 'minimum':
        # 最小值
        weight = min(m1, m2)
    elif weight_method == 'maximum':
        # 最大值
        weight = max(m1, m2)
    else:
        raise ValueError(f"未知的权重计算方法: {weight_method}")
    
    return weight

def process_chunk(args):
    """
    处理一个数据块的函数（用于多进程）
    
    参数:
        args: 包含所有必要参数的元组
    """
    (start_idx, end_idx, n_bins, window_size, half_lives,
     shm_name1, shm_name2, shm_shape, weights_dict, aggregation, weight_method) = args
    
    # 从共享内存中重建数组
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    # 初始化结果数组
    chunk_results = np.zeros(end_idx - start_idx)
    
    # 处理这个块中的每个bin
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx
        
        # 确定窗口范围
        win_start = max(0, i - window_size)
        win_end = min(n_bins, i + window_size + 1)
        
        # 提取窗口内的数据
        window_values1 = values1[win_start:win_end]
        window_values2 = values2[win_start:win_end]
        
        # 对每个半衰期计算相关性
        correlations = []
        for half_life in half_lives:
            weights_array = weights_dict[half_life]
            
            # 调整权重（处理边界情况）
            if i < window_size:
                window_weights = weights_array[window_size - i:]
            elif i >= n_bins - window_size:
                window_weights = weights_array[:window_size + (n_bins - i)]
            else:
                window_weights = weights_array
            
            # 计算加权Pearson相关系数
            correlation = weighted_pearson_correlation(window_values1, window_values2, 
                                                       window_weights)
            
            # 直接使用相关性（不再使用 1 - correlation）
            correlations.append(correlation)
        
        # 根据聚合方法计算聚合后的相关性
        if aggregation == 'mean':
            aggregated_correlation = np.mean(correlations)
        elif aggregation == 'max':
            aggregated_correlation = np.max(correlations)
        elif aggregation == 'min':
            aggregated_correlation = np.min(correlations)
        elif aggregation == 'median':
            aggregated_correlation = np.median(correlations)
        
        # 计算权重
        weight = calculate_weight(values1, values2, i, n_bins, weight_method)
        
        # 最终结果 = correlation * weight
        chunk_results[local_idx] = aggregated_correlation * weight
    
    # 清理共享内存引用
    shm1.close()
    shm2.close()
    
    return chunk_results

def parse_half_lives(half_life_str):
    """
    解析半衰期参数
    
    参数:
        half_life_str: 半衰期字符串，可以是单个数字或逗号分隔的列表
    
    返回:
        半衰期列表
    """
    if ',' in half_life_str:
        # 多个半衰期
        return [float(x.strip()) for x in half_life_str.split(',')]
    else:
        # 单个半衰期
        return [float(half_life_str)]

def calculate_local_correlation_parallel(bedgraph1_path, bedgraph2_path, 
                                        output_path, window_size=100, 
                                        half_lives=[5], aggregation='mean',
                                        weight_method='arithmetic',
                                        n_processes=None):
    """
    使用多进程和共享内存计算两个bedgraph文件的局部相关性
    
    参数:
        bedgraph1_path: 第一个bedgraph文件路径
        bedgraph2_path: 第二个bedgraph文件路径
        output_path: 输出bedgraph文件路径
        window_size: 窗口大小（单侧bin数，默认100）
        half_lives: 半衰期列表（默认[5]）
        aggregation: 聚合方法 ('mean', 'max', 'min', 'median')
        weight_method: 权重计算方法 ('arithmetic', 'geometric', 'harmonic', 'quadratic')
        n_processes: 进程数（默认为CPU核心数）
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"读取bedgraph文件...")
    bg1 = read_bedgraph(bedgraph1_path)
    bg2 = read_bedgraph(bedgraph2_path)
    
    # 检查两个文件的bin数是否一致
    if len(bg1) != len(bg2):
        raise ValueError(f"两个bedgraph文件的bin数不一致: {len(bg1)} vs {len(bg2)}")
    
    n_bins = len(bg1)
    print(f"总共 {n_bins} 个bins")
    print(f"使用 {n_processes} 个进程进行并行计算")
    
    # 提取数值并转换为numpy数组
    values1 = bg1['value'].values.astype(np.float64)
    values2 = bg2['value'].values.astype(np.float64)
    
    # 创建共享内存
    print("创建共享内存...")
    shm1 = shared_memory.SharedMemory(create=True, size=values1.nbytes)
    shm2 = shared_memory.SharedMemory(create=True, size=values2.nbytes)
    
    # 将数据复制到共享内存
    shared_array1 = np.ndarray(values1.shape, dtype=np.float64, buffer=shm1.buf)
    shared_array2 = np.ndarray(values2.shape, dtype=np.float64, buffer=shm2.buf)
    shared_array1[:] = values1[:]
    shared_array2[:] = values2[:]
    
    # 计算所有半衰期的指数衰减权重
    weights_dict = {}
    for half_life in half_lives:
        weights_dict[half_life] = exponential_decay_weights(window_size, half_life)
    
    print(f"窗口大小: {2*window_size + 1} bins (中心 ± {window_size} bins)")
    print(f"半衰期: {half_lives}")
    print(f"聚合方法: {aggregation}")
    print(f"权重计算方法: {weight_method}")
    
    # 将任务分割成多个块
    chunk_size = max(1, n_bins // n_processes)
    chunks = []
    for i in range(0, n_bins, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, n_bins)
        chunks.append((start_idx, end_idx, n_bins, window_size, half_lives,
                      shm1.name, shm2.name, values1.shape, weights_dict, aggregation, weight_method))
    
    print(f"任务分割为 {len(chunks)} 个块")
    print("开始并行计算...")
    
    # 使用进程池进行并行计算
    try:
        with Pool(processes=n_processes) as pool:
            results = pool.map(process_chunk, chunks)
        
        # 合并结果
        print("合并结果...")
        weighted_correlation = np.concatenate(results)
        
    finally:
        # 清理共享内存
        print("清理共享内存...")
        shm1.close()
        shm2.close()
        shm1.unlink()
        shm2.unlink()
    
    print("计算完成，写入输出文件...")
    
    # 创建输出DataFrame
    output_df = bg1[['chrom', 'start', 'end']].copy()
    output_df['weighted_correlation'] = weighted_correlation
    
    # 写入输出文件
    output_df.to_csv(output_path, sep='\t', header=False, index=False)
    
    print(f"结果已保存到: {output_path}")
    print(f"加权相关性统计:")
    print(f"  最小值: {np.min(weighted_correlation):.4f}")
    print(f"  最大值: {np.max(weighted_correlation):.4f}")
    print(f"  平均值: {np.mean(weighted_correlation):.4f}")
    print(f"  中位数: {np.median(weighted_correlation):.4f}")

def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("用法: python script.py <bedgraph1> <bedgraph2> <output> [window_size] [half_lives] [aggregation] [weight_method] [n_processes]")
        print("示例1 (单个半衰期): python script.py file1.bg file2.bg output.bg 100 5 mean arithmetic 8")
        print("示例2 (多个半衰期): python script.py file1.bg file2.bg output.bg 100 3,5,10 median geometric 8")
        print("聚合方法: mean, max, min, median")
        print("权重计算方法: arithmetic, geometric, harmonic, quadratic")
        print(f"默认进程数: {cpu_count()}")
        sys.exit(1)
    
    bedgraph1 = sys.argv[1]
    bedgraph2 = sys.argv[2]
    output = sys.argv[3]
    window_size = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    # 解析半衰期参数
    if len(sys.argv) > 5:
        half_lives = parse_half_lives(sys.argv[5])
    else:
        half_lives = [5]
    
    aggregation = sys.argv[6] if len(sys.argv) > 6 else 'mean'
    weight_method = sys.argv[7] if len(sys.argv) > 7 else 'arithmetic'
    n_processes = int(sys.argv[8]) if len(sys.argv) > 8 else None
    
    # 验证聚合方法
    valid_aggregations = ['mean', 'max', 'min', 'median']
    if aggregation not in valid_aggregations:
        print(f"错误: 聚合方法必须是 {valid_aggregations} 之一")
        sys.exit(1)
    
    # 验证权重计算方法
    valid_weight_methods = ['arithmetic', 'geometric', 'harmonic', 'quadratic', 'minimum', 'maximum']
    if weight_method not in valid_weight_methods:
        print(f"{weight_method} is not a correct weight method.")
        print(f"错误: 权重计算方法必须是 {valid_weight_methods} 之一")
        sys.exit(1)
    
    calculate_local_correlation_parallel(bedgraph1, bedgraph2, output, 
                                        window_size, half_lives, aggregation,
                                        weight_method, n_processes)

if __name__ == "__main__":
    main()
