#!/usr/bin/env python3
"""
计算两个bedgraph文件的局部差异显著性
使用滑动窗口、z检验和多进程优化
支持多个窗口大小和聚合方法
增加了基于背景的权重计算
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportions_ztest
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

import numpy as np
from scipy.stats import poisson

def calculate_weight(win_count, bg_count):
    n = win_count + bg_count

    lam = bg_count / 5000

    if n > lam:
        p_value = poisson.sf(win_count - 1, mu=lam)
    else:
        p_value = poisson.cdf(win_count, mu=lam)

    weight = 1.0 - p_value
    return max(0.0, min(1.0, weight))

def calculate_binomial_pvalue(x1, x2, total_count1, total_count2):
    """
    使用proportions_ztest计算p值
    
    参数:
        x1: 第一个文件窗口内的观测值总和
        x2: 第二个文件窗口内的观测值总和
        total_count1: 第一个track的全局总count数
        total_count2: 第二个track的全局总count数
    
    返回:
        差异值（基于p值的对数变换），如果无法计算则返回0
    """
    # 如果窗口内总和为0，返回0
    if x1 + x2 == 0:
        return 0
    
    # 如果全局总数为0，返回0
    if total_count1 == 0 or total_count2 == 0:
        return 0
    
    try:
        # 构建count和nobs数组
        count = np.array([x1, x2])
        nobs = np.array([total_count1, total_count2])
        
        z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
        
        if p_value == 0:
            p_value = 1e-300
        
        diff_value = np.log10(p_value)
        if x1 > x2:
            diff_value = -diff_value
        return diff_value
    except Exception as e:
        # 如果计算失败，返回0
        return 0

def process_chunk(args):
    """
    处理一个数据块的函数（用于多进程）
    
    参数:
        args: 包含所有必要参数的元组
    """
    (start_idx, end_idx, n_bins, window_sizes,
     shm_name1, shm_name2, shm_shape, aggregation,
     total_count1, total_count2) = args
    
    # 从共享内存中重建数组
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    shm2 = shared_memory.SharedMemory(name=shm_name2)
    
    values1 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm1.buf)
    values2 = np.ndarray(shm_shape, dtype=np.float64, buffer=shm2.buf)
    
    # 初始化
    chunk_results = np.zeros(end_idx - start_idx)

    bg_values1 = np.zeros(1)
    bg_values2 = np.zeros(1)

    bg_count1 = 0
    bg_count2 = 0
    
    # 处理这个块中的每个bin
    for i in range(start_idx, end_idx):
        local_idx = i - start_idx

        if local_idx % 5000 == 0:
            bg_values1 = values1[max(0, i - 2500):min(n_bins, i + 2500)]
            bg_values2 = values2[max(0, i - 2500):min(n_bins, i + 2500)]

            bg_count1 = np.sum(bg_values1)
            bg_count2 = np.sum(bg_values2)
        
        # 对每个窗口大小计算p值

        # center_value1 = np.sum(values1[max(0, i - 1):min(n_bins, i + 1)])
        # center_value2 = np.sum(values2[max(0, i - 1):min(n_bins, i + 1)])

        center_value1 = values1[i]
        center_value2 = values2[i]

        weight1 = calculate_weight(center_value1, bg_count1)
        weight2 = calculate_weight(center_value2, bg_count2)
        
        final_weight = max(weight1, weight2)



        pvalues_for_windows = []
        for window_size in window_sizes:
            # 确定窗口范围
            win_start = max(0, i - window_size)
            win_end = min(n_bins, i + window_size + 1)
            
            # 提取窗口内的数据
            window_values1 = values1[win_start:win_end]
            window_values2 = values2[win_start:win_end]
            
            # 计算窗口内的总和
            x1 = np.sum(window_values1)
            x2 = np.sum(window_values2)

            diff_value = calculate_binomial_pvalue(x1, x2, total_count1, total_count2)
            

            weighted_value = diff_value * final_weight
            
            pvalues_for_windows.append(weighted_value)
        
        # 根据聚合方法计算最终值
        if aggregation == 'mean':
            chunk_results[local_idx] = np.mean(pvalues_for_windows)
        elif aggregation == 'max':
            # 最远离0：绝对值最大
            abs_values = np.abs(pvalues_for_windows)
            max_idx = np.argmax(abs_values)
            chunk_results[local_idx] = pvalues_for_windows[max_idx]
        elif aggregation == 'min':
            # 最靠近0：绝对值最小
            abs_values = np.abs(pvalues_for_windows)
            min_idx = np.argmin(abs_values)
            chunk_results[local_idx] = pvalues_for_windows[min_idx]
        elif aggregation == 'median':
            chunk_results[local_idx] = np.median(pvalues_for_windows)
    
    # 清理共享内存引用
    shm1.close()
    shm2.close()
    
    return chunk_results

def parse_window_sizes(window_size_str):
    """
    解析窗口大小参数
    
    参数:
        window_size_str: 窗口大小字符串，可以是单个数字或逗号分隔的列表
    
    返回:
        窗口大小列表
    """
    if ',' in window_size_str:
        # 多个窗口大小
        return [int(x.strip()) for x in window_size_str.split(',')]
    else:
        # 单个窗口大小
        return [int(window_size_str)]

def calculate_local_difference_parallel(bedgraph1_path, bedgraph2_path, 
                                       output_path, window_sizes=[100],
                                       aggregation='mean',
                                       n_processes=None):
    """
    使用多进程和共享内存计算两个bedgraph文件的局部差异显著性
    
    参数:
        bedgraph1_path: 第一个bedgraph文件路径
        bedgraph2_path: 第二个bedgraph文件路径
        output_path: 输出bedgraph文件路径
        window_sizes: 窗口大小列表（单侧bin数，默认[100]）
        aggregation: 聚合方法 ('mean', 'max', 'min', 'median')
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
    
    # 计算全局总count数
    total_count1 = np.sum(values1)
    total_count2 = np.sum(values2)
    
    print(f"Track1 全局总count: {total_count1:.0f}")
    print(f"Track2 全局总count: {total_count2:.0f}")
    
    # 创建共享内存
    print("创建共享内存...")
    shm1 = shared_memory.SharedMemory(create=True, size=values1.nbytes)
    shm2 = shared_memory.SharedMemory(create=True, size=values2.nbytes)
    
    # 将数据复制到共享内存
    shared_array1 = np.ndarray(values1.shape, dtype=np.float64, buffer=shm1.buf)
    shared_array2 = np.ndarray(values2.shape, dtype=np.float64, buffer=shm2.buf)
    shared_array1[:] = values1[:]
    shared_array2[:] = values2[:]
    
    print(f"窗口大小: {window_sizes}")
    print(f"背景区域大小: {[ws * 10 for ws in window_sizes]}")
    print(f"聚合方法: {aggregation}")
    
    # 将任务分割成多个块
    chunk_size = max(1, n_bins // n_processes)
    chunks = []
    for i in range(0, n_bins, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, n_bins)
        chunks.append((start_idx, end_idx, n_bins, window_sizes,
                      shm1.name, shm2.name, values1.shape, aggregation,
                      total_count1, total_count2))
    
    print(f"任务分割为 {len(chunks)} 个块")
    print("开始并行计算（使用proportions_ztest和权重计算）...")
    
    # 使用进程池进行并行计算
    try:
        with Pool(processes=n_processes) as pool:
            results = pool.map(process_chunk, chunks)
        
        # 合并结果
        print("合并结果...")
        weighted_diff_values = np.concatenate(results)
        
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
    output_df['weighted_diff_value'] = weighted_diff_values
    
    # 写入输出文件
    output_df.to_csv(output_path, sep='\t', header=False, index=False)
    
    print(f"结果已保存到: {output_path}")
    print(f"加权差异值统计:")
    print(f"  最小值: {np.min(weighted_diff_values):.4f}")
    print(f"  最大值: {np.max(weighted_diff_values):.4f}")
    print(f"  平均值: {np.mean(weighted_diff_values):.4f}")
    print(f"  中位数: {np.median(weighted_diff_values):.4f}")
    

def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("用法: python script.py <bedgraph1> <bedgraph2> <output> [window_sizes] [aggregation] [n_processes]")
        print("示例1 (单个窗口): python script.py file1.bg file2.bg output.bg 100 mean 8")
        print("示例2 (多个窗口): python script.py file1.bg file2.bg output.bg 5,10,25 min 192")
        print("聚合方法: mean, max, min, median")
        print(f"默认进程数: {cpu_count()}")
        sys.exit(1)
    
    bedgraph1 = sys.argv[1]
    bedgraph2 = sys.argv[2]
    output = sys.argv[3]
    
    # 解析窗口大小参数
    if len(sys.argv) > 4:
        window_sizes = parse_window_sizes(sys.argv[4])
    else:
        window_sizes = [100]
    
    aggregation = sys.argv[5] if len(sys.argv) > 5 else 'mean'
    n_processes = int(sys.argv[6]) if len(sys.argv) > 6 else None
    
    # 验证聚合方法
    valid_aggregations = ['mean', 'max', 'min', 'median']
    if aggregation not in valid_aggregations:
        print(f"错误: 聚合方法必须是 {valid_aggregations} 之一")
        sys.exit(1)
    
    calculate_local_difference_parallel(bedgraph1, bedgraph2, output, 
                                       window_sizes, aggregation,
                                       n_processes)

if __name__ == "__main__":
    main()
