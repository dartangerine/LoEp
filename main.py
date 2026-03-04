#!/usr/bin/env python3
"""
LoEp - Local Epigenomic Pattern Analysis Tool
主程序入口
"""

import sys
import argparse
import os
from multiprocessing import cpu_count

from elc import calculate_local_correlation_parallel
from eld import calculate_local_difference_parallel
from general import parse_window_sizes


def generate_output_path(base_output, method_type, method_name):
    """生成输出文件路径"""
    dir_name = os.path.dirname(base_output)
    base_name = os.path.basename(base_output)
    
    if '.' in base_name:
        name_parts = base_name.rsplit('.', 1)
        name = name_parts[0]
        ext = '.' + name_parts[1]
    else:
        name = base_name
        ext = ''
    
    new_name = f"{name}_{method_type}_{method_name}{ext}"
    
    if dir_name:
        return os.path.join(dir_name, new_name)
    else:
        return new_name


def run_elc(method, input1, input2, output, window_sizes, aggregation, weight_method, processes):
    """运行ELC分析"""
    print(f"\n{'='*60}")
    print(f"Computing ELC. Method: {method}")
    print(f"{'='*60}\n")
    
    if method == 'pearson_exp':
        # pearson_exp 使用单个窗口大小和半衰期列表
        window_size = window_sizes[0] if isinstance(window_sizes, list) else window_sizes
        half_lives = [5]  # 默认半衰期
        calculate_local_correlation_parallel(
            input1, input2, output,
            method='pearson_exp',
            window_sizes=window_size,
            aggregation=aggregation,
            weight_method=weight_method,
            n_processes=processes,
            half_lives=half_lives
        )
    else:
        calculate_local_correlation_parallel(
            input1, input2, output,
            method=method,
            window_sizes=window_sizes,
            aggregation=aggregation,
            weight_method=weight_method,
            n_processes=processes
        )


def run_eld(method, input1, input2, output, window_sizes, aggregation, processes):
    """运行ELD分析"""
    print(f"\n{'='*60}")
    print(f"Running ELD analysis - Method: {method}")
    print(f"{'='*60}\n")
    
    calculate_local_difference_parallel(
        input1, input2, output,
        method=method,
        window_sizes=window_sizes,
        aggregation=aggregation,
        n_processes=processes
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='LoEp - Local Epigenomic Pattern Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:

   python main.py -i1 file1.bg -i2 file2.bg -o output.bg -cm pearson -dm binomial -w 100 -a mean
        """
    )
    
    # 必需参数
    parser.add_argument('-i1', '--input1', required=True, help='Directory path for the first bedgraph file')
    parser.add_argument('-i2', '--input2', required=True, help='Directory path for the second bedgraph file')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    
    # 方法选择
    parser.add_argument('-cm', '--correlation_method', required=True,
                       choices=['pearson', 'pearson_exp', 'chi2', 'ks', 'mi', 'none'],
                       help='Correlation computing method for ELC (set to none to skip ELC)')
    parser.add_argument('-dm', '--difference_method', required=True,
                       choices=['binomial', 'poisson', 'negbinomial', 'zinb', 'none'],
                       help='Difference computing method for ELD (set to none to skip ELD)')
    
    # 可选参数
    parser.add_argument('-w', '--window_sizes', default='100',
                       help='Window size (number of bins on each side), can be a single number or a comma-separated list, e.g., "100" or "50,100,200" (default: 100)')
    parser.add_argument('-a', '--aggregation', default='mean',
                       choices=['mean', 'max', 'min', 'median'],
                       help='Multi-window aggregation method (default: mean)')
    parser.add_argument('-wm', '--weight_method', default='arithmetic',
                       choices=['arithmetic', 'geometric', 'harmonic', 'quadratic', 'minimum', 'maximum'],
                       help='Weight computing method for ELC (default: arithmetic)')
    parser.add_argument('-p', '--processes', type=int, default=None,
                       help=f'Number of parallel processes (default: {cpu_count()})')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    if args.correlation_method == 'none' and args.difference_method == 'none':
        print("Error: Configure at least one method to run (ELC or ELD).", file=sys.stderr)
        sys.exit(1)
    
    window_sizes = parse_window_sizes(args.window_sizes)
    
    try:
        # 判断是否需要生成多个输出文件
        both_methods = (args.correlation_method != 'none' and args.difference_method != 'none')
        
        if args.correlation_method != 'none':
            if both_methods:
                elc_output = generate_output_path(args.output, 'ELC', args.correlation_method)
            else:
                elc_output = args.output
            
            run_elc(
                args.correlation_method,
                args.input1,
                args.input2,
                elc_output,
                window_sizes,
                args.aggregation,
                args.weight_method,
                args.processes
            )

        if args.difference_method != 'none':
            if both_methods:
                eld_output = generate_output_path(args.output, 'ELD', args.difference_method)
            else:
                eld_output = args.output
            
            run_eld(
                args.difference_method,
                args.input1,
                args.input2,
                eld_output,
                window_sizes,
                args.aggregation,
                args.processes
            )
        
        print(f"\n{'='*60}")
        print("All analysis completed!")
        if both_methods:
            print(f"ELC result saved as: {generate_output_path(args.output, 'ELC', args.correlation_method)}")
            print(f"ELD result saved as: {generate_output_path(args.output, 'ELD', args.difference_method)}")
        else:
            print(f"Result saved as: {args.output}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
