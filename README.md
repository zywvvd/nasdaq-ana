# 纳斯达克多尺度方向预测系统

基于 15 年历史数据（2011-2026）的纳斯达克综合指数方向预测系统。V6 架构采用 5 个 Horizon LightGBM + GBC 元分类器，输出未来 1/3/5/7/10 个交易日的方向预测和顶底事件概率。

## 快速开始

```bash
# 安装依赖
pip install lightgbm scikit-learn pandas numpy matplotlib yfinance

# 训练 + 预测
python run_v6_pipeline.py --all

# 仅训练
python run_v6_pipeline.py --train

# 仅预测
python run_v6_pipeline.py --predict
```

## 预测输出示例

```
V6 多尺度方向预测 — 2026-06-08
  收盘价: 25,930

  方向预测:
     1d: -0.045 →      3d: -0.064 →      5d: -0.176 ↓
     7d: -0.159 ↓     10d: -0.124 ↓

  综合评分: -0.125 ↓
  方向一致度: 100.0%
  ML评分: 顶部概率=4.8% | 底部概率=77.2%
```

## V6 模型架构

```
179 维特征 → 5 个 LightGBM (1d/3d/5d/7d/10d) → 方向预测 [-1, +1]
                                              ↓
                              GBC 元分类器 → 顶/底事件概率
```

- **特征**：179 维（36 基础 + 21 宏观估值 + 96 窗口变化 + 26 V7 新增）
- **特征选择**：短周期 top_k=80，长周期 top_k=50（importance-based）
- **标签**：滚动 504 天百分位排名，映射到 [-1, +1]
- **验证**：非重叠 Walk-forward CV（50 折，train=504d，test=63d）

## OOS 性能

| Horizon | 方向准确率 | IC | AUC | 过拟合差距 |
|---------|----------|------|------|----------|
| 1d | **70.3%** | 0.529 | 0.760 | 4.7% |
| 3d | 63.2% | 0.370 | 0.691 | 16.4% |
| 5d | 62.4% | 0.352 | 0.680 | 16.9% |
| 7d | 64.1% | 0.378 | 0.689 | 13.6% |
| 10d | 64.9% | 0.408 | 0.715 | 14.8% |

基准线对比：永远看多 55.7% | 1 日动量 49.7% | 5 日动量 50.6%

元层 AUC：顶部 0.787 | 底部 0.707

## 项目结构

```
earning/
├── run_v6_pipeline.py          # 主流水线
├── download_data.py            # 数据下载（yfinance）
├── lib/
│   ├── config.py               # 全局配置（179 维特征列表）
│   ├── data_fetcher.py         # 数据加载与合并
│   ├── features.py             # 179 维特征计算
│   ├── v6_config.py            # V6 模型超参数
│   ├── v6_labels.py            # 滚动百分位标签
│   ├── v6_trainer.py           # 训练器（5 Horizon + 元层）
│   ├── v6_scorer.py            # 预测器
│   └── model_factory.py        # LightGBM 工厂
├── v6_models/                  # 训练好的模型文件
├── data/                       # 本地 CSV 数据
├── charts/                     # 分析图表
│   └── generate_v6_charts.py   # V6 可视化生成
└── 纳斯达克指数分析与ML模型报告.md    # 完整分析报告
```

## 数据来源

| 数据 | 来源 | 用途 |
|------|------|------|
| 纳斯达克综合指数 (^IXIC) | yfinance | OHLCV |
| VIX / VVIX / VIX9D / VIX3M | yfinance | 波动率情绪 |
| CBOE SKEW | yfinance | 尾部风险 |
| 10Y-2Y 国债利差 | yfinance | 收益率曲线 |
| HYG / LQD / TLT | yfinance | 信用利差 |
| S&P 500 PE + Shiller CAPE | multpl.com | 估值水平 |

## 免责声明

**本项目仅供研究参考，不构成任何投资建议。** 模型方向准确率 62-70%，意味着仍有 30-38% 的错误率。永远不要投入你无法承受损失的资金。
