"""
计算两个bedgraph文件的局部加权Kolmogorov-Smirnov统计量
使用滑动窗口、多进程和共享内存优化
支持多个窗口大小和聚合方法
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
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

def simple_ks_statistic(x, y):
    """
    计算Kolmogorov-Smirnov检验统计量D，并返回(1 - D)
    
    参数:
        x, y: 两个数据数组
    
    返回:
        (1 - D)，其中D是KS统计量，如果无法计算则返回0
    """
    try:
        # 计算KS检验
        ks_statistic, p_value = ks_2samp(x, y)
        # 返回(1 - D)
        return 1 - ks_statistic
    except:
        return 0  # 如果计算失败，返回0

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
    (start_idx, end_idx, n_bins, max_window_size, window_sizes,
     shm_name1, shm_name2, shm_shape, aggregation, weight_method) = args
    
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
        
        # 对每个窗口大小计算KS统计量
        ks_values = []
        for window_size in window_sizes:
            # 确定窗口范围
            win_start = max(0, i - window_size)
            win_end = min(n_bins, i + window_size + 1)
            
            # 提取窗口内的数据
            window_values1 = values1[win_start:win_end]
            window_values2 = values2[win_start:win_end]
            
            # 计算(1 - D)，其中D是KS统计量
            one_minus_d = simple_ks_statistic(window_values1, window_values2)
            
            ks_values.append(one_minus_d)
        
        # 根据聚合方法计算聚合后的(1 - D)
        if aggregation == 'mean':
            aggregated_ks = np.mean(ks_values)
        elif aggregation == 'max':
            aggregated_ks = np.max(ks_values)
        elif aggregation == 'min':
            aggregated_ks = np.min(ks_values)
        elif aggregation == 'median':
            aggregated_ks = np.median(ks_values)
        
        # 计算权重
        weight = calculate_weight(values1, values2, i, n_bins, weight_method)
        
        # 最终结果 = (1 - D) * weight
        chunk_results[local_idx] = aggregated_ks * weight
    
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
        窗口大小列表（整数）
    """
    if ',' in window_size_str:
        # 多个窗口大小
        return [int(x.strip()) for x in window_size_str.split(',')]
    else:
        # 单个窗口大小
        return [int(window_size_str)]

def calculate_local_ks_parallel(bedgraph1_path, bedgraph2_path, 
                                output_path, window_sizes=[100], 
                                aggregation='mean',
                                weight_method='arithmetic',
                                n_processes=None):
    """
    使用多进程和共享内存计算两个bedgraph文件的局部KS统计量
    
    参数:
        bedgraph1_path: 第一个bedgraph文件路径
        bedgraph2_path: 第二个bedgraph文件路径
        output_path: 输出bedgraph文件路径
        window_sizes: 窗口大小列表（单侧bin数，默认[100]）
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
    
    # 获取最大窗口大小
    max_window_size = max(window_sizes)
    
    print(f"窗口大小: {window_sizes}")
    print(f"聚合方法: {aggregation}")
    print(f"权重计算方法: {weight_method}")
    print(f"使用Kolmogorov-Smirnov检验，计算(1 - D) × weight")
    
    # 将任务分割成多个块
    chunk_size = max(1, n_bins // n_processes)
    chunks = []
    for i in range(0, n_bins, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, n_bins)
        chunks.append((start_idx, end_idx, n_bins, max_window_size, window_sizes,
                      shm1.name, shm2.name, values1.shape, aggregation, weight_method))
    
    print(f"任务分割为 {len(chunks)} 个块")
    print("开始并行计算...")
    
    # 使用进程池进行并行计算
    try:
        with Pool(processes=n_processes) as pool:
            results = pool.map(process_chunk, chunks)
        
        # 合并结果
        print("合并结果...")
        weighted_ks = np.concatenate(results)
        
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
    output_df['weighted_ks'] = weighted_ks
    
    # 写入输出文件
    output_df.to_csv(output_path, sep='\t', header=False, index=False)
    
    print(f"结果已保存到: {output_path}")
    print(f"加权KS统计量统计:")
    print(f"  最小值: {np.min(weighted_ks):.4f}")
    print(f"  最大值: {np.max(weighted_ks):.4f}")
    print(f"  平均值: {np.mean(weighted_ks):.4f}")
    print(f"  中位数: {np.median(weighted_ks):.4f}")

def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("用法: python script.py <bedgraph1> <bedgraph2> <output> [window_sizes] [aggregation] [weight_method] [n_processes]")
        print("示例1 (单个窗口): python script.py file1.bg file2.bg output.bg 100 mean arithmetic 8")
        print("示例2 (多个窗口): python script.py file1.bg file2.bg output.bg 50,100,200 median geometric 8")
        print("聚合方法: mean, max, min, median")
        print("权重计算方法: arithmetic, geometric, harmonic, quadratic, minimum, maximum")
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
    weight_method = sys.argv[6] if len(sys.argv) > 6 else 'arithmetic'
    n_processes = int(sys.argv[7]) if len(sys.argv) > 7 else None
    
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
    
    calculate_local_ks_parallel(bedgraph1, bedgraph2, output, 
                                window_sizes, aggregation,
                                weight_method, n_processes)

if __name__ == "__main__":
    main()
