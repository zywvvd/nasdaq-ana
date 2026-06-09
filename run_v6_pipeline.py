"""V6 多尺度方向预测系统 — 主流水线。"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.data_fetcher import fetch_training_data
from lib.features import compute_features
from lib.v6_trainer import train_v6
from lib.v6_scorer import predict_v6, print_prediction
from lib.v6_config import MODEL_DIR


def step_train():
    print("\n[V6 训练]")
    train_v6()


def step_predict():
    print("\n[V6 预测]")
    df = fetch_training_data()
    df = compute_features(df)
    results = predict_v6(df.tail(1))
    if results:
        print_prediction(results[-1])
    else:
        print("无有效预测")


def main():
    parser = argparse.ArgumentParser(description='V6 多尺度方向预测系统')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', action='store_true', help='生成预测')
    parser.add_argument('--all', action='store_true', help='训练+预测')
    args = parser.parse_args()

    if not any([args.train, args.predict, args.all]):
        args.all = True

    if args.all or args.train:
        step_train()

    if args.all or args.predict:
        step_predict()

    print("\n完成!")


if __name__ == '__main__':
    main()
