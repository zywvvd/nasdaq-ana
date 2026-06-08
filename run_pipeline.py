#!/usr/bin/env python3
"""主控流水线 — 一键执行：下载 → 训练 → 评估 → 预测

用法:
  python run_pipeline.py                # 执行全部步骤
  python run_pipeline.py --all          # 同上
  python run_pipeline.py --download     # 仅下载数据
  python run_pipeline.py --train        # 仅训练模型
  python run_pipeline.py --evaluate     # 仅评估模型
  python run_pipeline.py --predict      # 仅运行预测
  python run_pipeline.py --train --evaluate  # 训练+评估
  python run_pipeline.py --period 5y    # 下载最近5年数据
"""
import argparse
import sys
import time


def step_download(period='15y'):
    """步骤1: 下载数据"""
    print("\n" + "=" * 60)
    print("步骤 1/4: 下载数据")
    print("=" * 60)
    from download_data import download_all
    download_all(period=period)


def step_train():
    """步骤2: 训练模型"""
    print("\n" + "=" * 60)
    print("步骤 2/4: 训练模型")
    print("=" * 60)
    from lib.trainer import train_and_evaluate
    model_top, model_bot = train_and_evaluate()
    return model_top, model_bot


def step_evaluate(model_top=None, model_bot=None):
    """步骤3: 评估模型"""
    print("\n" + "=" * 60)
    print("步骤 3/4: 评估模型")
    print("=" * 60)
    from lib.evaluator import evaluate
    results = evaluate(model_top=model_top, model_bot=model_bot)
    return results


def step_predict():
    """步骤4: 每日预测"""
    print("\n" + "=" * 60)
    print("步骤 4/4: 每日预测")
    print("=" * 60)
    from lib.predictor import predict
    result = predict()
    return result


def main():
    parser = argparse.ArgumentParser(
        description='纳斯达克 ML 模型流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--all', action='store_true', help='执行全部步骤')
    parser.add_argument('--download', action='store_true', help='下载数据')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--evaluate', action='store_true', help='评估模型')
    parser.add_argument('--predict', action='store_true', help='运行预测')
    parser.add_argument('--period', default='15y', help='下载数据周期 (默认 15y)')

    args = parser.parse_args()

    # 若无指定步骤，默认全部执行
    do_all = args.all or not any([args.download, args.train, args.evaluate, args.predict])
    do_download = args.download or do_all
    do_train = args.train or do_all
    do_evaluate = args.evaluate or do_all
    do_predict = args.predict or do_all

    start_time = time.time()

    try:
        model_top = model_bot = None

        if do_download:
            step_download(period=args.period)

        if do_train:
            model_top, model_bot = step_train()

        if do_evaluate:
            step_evaluate(model_top=model_top, model_bot=model_bot)

        if do_predict:
            step_predict()

    except Exception as e:
        print(f"\n流水线错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n流水线完成! 耗时 {elapsed:.1f} 秒")


if __name__ == '__main__':
    main()
