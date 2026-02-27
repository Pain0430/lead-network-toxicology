#!/usr/bin/env python网络毒理学 -3
"""
铅 统一命令行界面
Lead Network Toxicology - Unified CLI

用法:
    python cli.py --help
    python cli.py causal --help
    python cli.py all

作者: Pain (重庆医科大学)
日期: 2026-02-28
"""

import argparse
import sys
import os
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description='铅网络毒理学分析 - Lead Network Toxicology CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python cli.py --help                  显示帮助
    python cli.py causal                   运行因果推断分析
    python cli.py dose-response            运行剂量-反应分析
    python cli.py network                   运行生物标志物网络分析
    python cli.py ml                        运行机器学习预测
    python cli.py visualize                 生成可视化图表
    python cli.py report                    生成综合报告
    python cli.py all                       运行完整分析流程
        """
    )
    
    parser.add_argument('--input', '-i', default='data/nhanes_lead_data.csv',
                        help='输入数据文件路径 (默认: data/nhanes_lead_data.csv)')
    parser.add_argument('--output', '-o', default='output/',
                        help='输出目录 (默认: output/)')
    parser.add_argument('--config', '-c', default='chinese_config.py',
                        help='配置文件路径 (默认: chinese_config.py)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='输出详细信息')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用子命令')
    
    # 因果推断分析
    parser_causal = subparsers.add_parser('causal', help='运行因果推断分析')
    parser_causal.add_argument('--method', choices=['psm', 'iptw', 'aipw', 'all'],
                                default='all', help='因果推断方法 (默认: all)')
    parser_causal.add_argument('--exposure', default='LBXBPB',
                                help='暴露变量 (默认: LBXBPB 血铅)')
    parser_causal.add_argument('--outcome', default='CKM_risk',
                                help='结局变量 (默认: CKM_risk)')
    
    # 剂量-反应分析
    parser_dose = subparsers.add_parser('dose-response', help='运行剂量-反应分析')
    parser_dose.add_argument('--metal', default='lead',
                             choices=['lead', 'arsenic', 'cadmium', 'mercury', 'manganese'],
                             help='金属类型 (默认: lead)')
    parser_dose.add_argument('--outcome', default='hypertension',
                             help='健康结局 (默认: hypertension)')
    
    # 生物标志物网络分析
    parser_network = subparsers.add_parser('network', help='运行生物标志物网络分析')
    parser_network.add_argument('--biomarkers', nargs='+',
                                default=['LBXBPB', 'LBXSATSI', 'LBXSAPSI', 'LBXSASSI'],
                                help='生物标志物列表')
    parser_network.add_argument('--threshold', type=float, default=0.3,
                                help='相关系数阈值 (默认: 0.3)')
    
    # 机器学习预测
    parser_ml = subparsers.add_parser('ml', help='运行机器学习预测')
    parser_ml.add_argument('--model', default='rf',
                          choices=['rf', 'gb', 'lr', 'all'],
                          help='模型类型 (默认: rf RandomForest)')
    parser_ml.add_argument('--target', default='CKM_risk',
                          help='预测目标 (默认: CKM_risk)')
    
    # 可视化
    parser_viz = subparsers.add_parser('visualize', help='生成可视化图表')
    parser_viz.add_argument('--type', choices=['heatmap', 'network', 'forest', 'all'],
                            default='all', help='图表类型 (默认: all)')
    
    # 报告生成
    parser_report = subparsers.add_parser('report', help='生成综合报告')
    parser_report.add_argument('--format', choices=['txt', 'html', 'md', 'all'],
                               default='all', help='报告格式 (默认: all)')
    
    # 完整分析
    parser_all = subparsers.add_parser('all', help='运行完整分析流程')
    
    args = parser.parse_args()
    
    # 如果没有子命令，显示帮助
    if args.command is None:
        parser.print_help()
        return 0
    
    # 执行对应的分析
    if args.verbose:
        print(f"命令: {args.command}")
        print(f"输入: {args.input}")
        print(f"输出: {args.output}")
    
    # 根据子命令执行不同的分析
    if args.command == 'causal':
        return run_causal_analysis(args)
    elif args.command == 'dose-response':
        return run_dose_response(args)
    elif args.command == 'network':
        return run_network_analysis(args)
    elif args.command == 'ml':
        return run_ml_prediction(args)
    elif args.command == 'visualize':
        return run_visualization(args)
    elif args.command == 'report':
        return run_report(args)
    elif args.command == 'all':
        return run_all_analysis(args)
    
    return 0


def run_causal_analysis(args):
    """运行因果推断分析"""
    print("\n" + "="*60)
    print("运行因果推断分析...")
    print("="*60)
    
    try:
        from causal_inference import run_full_analysis
        
        result = run_full_analysis(
            exposure_var=args.exposure,
            outcome_var=args.outcome,
            output_dir=args.output
        )
        
        if result:
            print("\n✓ 因果推断分析完成!")
            print(f"  结果保存至: {args.output}")
        else:
            print("\n✗ 因果推断分析失败")
            return 1
            
    except ImportError as e:
        print(f"\n✗ 无法导入 causal_inference 模块: {e}")
        print("  请确保 causal_inference.py 存在")
        return 1
    except Exception as e:
        print(f"\n✗ 分析出错: {e}")
        return 1
    
    return 0


def run_dose_response(args):
    """运行剂量-反应分析"""
    print("\n" + "="*60)
    print(f"运行剂量-反应分析 ({args.metal})...")
    print("="*60)
    
    try:
        from dose_response_analysis import analyze_dose_response
        
        result = analyze_dose_response(
            metal=args.metal,
            outcome=args.outcome,
            output_dir=args.output
        )
        
        if result:
            print("\n✓ 剂量-反应分析完成!")
        else:
            print("\n✗ 剂量-反应分析失败")
            return 1
            
    except ImportError as e:
        print(f"\n✗ 无法导入 dose_response_analysis 模块: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 分析出错: {e}")
        return 1
    
    return 0


def run_network_analysis(args):
    """运行生物标志物网络分析"""
    print("\n" + "="*60)
    print("运行生物标志物网络分析...")
    print("="*60)
    
    try:
        from biomarker_network_analysis import build_network
        
        result = build_network(
            biomarkers=args.biomarkers,
            threshold=args.threshold,
            output_dir=args.output
        )
        
        if result:
            print("\n✓ 网络分析完成!")
        else:
            print("\n✗ 网络分析失败")
            return 1
            
    except ImportError as e:
        print(f"\n✗ 无法导入 biomarker_network_analysis 模块: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 分析出错: {e}")
        return 1
    
    return 0


def run_ml_prediction(args):
    """运行机器学习预测"""
    print("\n" + "="*60)
    print(f"运行机器学习预测 ({args.model})...")
    print("="*60)
    
    try:
        from ml_risk_prediction import train_and_evaluate
        
        result = train_and_evaluate(
            model_type=args.model,
            target=args.target,
            output_dir=args.output
        )
        
        if result:
            print("\n✓ 机器学习预测完成!")
        else:
            print("\n✗ 机器学习预测失败")
            return 1
            
    except ImportError as e:
        print(f"\n✗ 无法导入 ml_risk_prediction 模块: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 分析出错: {e}")
        return 1
    
    return 0


def run_visualization(args):
    """运行可视化"""
    print("\n" + "="*60)
    print("生成可视化图表...")
    print("="*60)
    
    try:
        from comprehensive_visualization import generate_all_figures
        
        result = generate_all_figures(
            viz_type=args.type,
            output_dir=args.output
        )
        
        if result:
            print("\n✓ 可视化完成!")
        else:
            print("\n✗ 可视化失败")
            return 1
            
    except ImportError as e:
        print(f"\n✗ 无法导入 comprehensive_visualization 模块: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 可视化出错: {e}")
        return 1
    
    return 0


def run_report(args):
    """运行报告生成"""
    print("\n" + "="*60)
    print("生成综合报告...")
    print("="*60)
    
    try:
        from generate_report import create_report
        
        result = create_report(
            format=args.format,
            output_dir=args.output
        )
        
        if result:
            print("\n✓ 报告生成完成!")
        else:
            print("\n✗ 报告生成失败")
            return 1
            
    except ImportError as e:
        print(f"\n✗ 无法导入 generate_report 模块: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 报告生成出错: {e}")
        return 1
    
    return 0


def run_all_analysis(args):
    """运行完整分析流程"""
    print("\n" + "="*60)
    print("⚡ 运行完整分析流程")
    print("="*60)
    
    # 按顺序运行各个模块
    steps = [
        ("因果推断分析", lambda: run_causal_analysis(args)),
        ("剂量-反应分析", lambda: run_dose_response(args)),
        ("网络分析", lambda: run_network_analysis(args)),
        ("机器学习预测", lambda: run_ml_prediction(args)),
        ("可视化", lambda: run_visualization(args)),
        ("报告生成", lambda: run_report(args)),
    ]
    
    results = []
    for step_name, step_func in steps:
        print(f"\n>>> 步骤: {step_name}")
        result = step_func()
        results.append((step_name, result))
    
    # 总结
    print("\n" + "="*60)
    print("📊 完整分析流程总结")
    print("="*60)
    
    for step_name, result in results:
        status = "✓ 成功" if result == 0 else "✗ 失败"
        print(f"  {step_name}: {status}")
    
    # 计算成功率
    success_count = sum(1 for _, r in results if r == 0)
    print(f"\n成功率: {success_count}/{len(results)} ({100*success_count/len(results):.0f}%)")
    
    return 0 if success_count == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
