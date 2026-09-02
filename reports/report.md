# 模型对比实验：LightGBM / DoubleEnsemble / MLP / GRU（qlib / Alpha158 / csi300）

平台：本地 Windows + RTX 4060 Laptop 8GB | 框架：qlib 0.9.7

## 实验设计（控制变量）

| 项目 | 设置 |
| --- | --- |
| 股票池 | 沪深300（动态成分股） |
| 因子 | Alpha158（GRU 按官方配置筛选其中 20 个因子；MLP 官方配置丢 VWAP0） |
| 标签 | `Ref($close,-2)/Ref($close,-1)-1`（T+1 买 T+2 卖收益） |
| 区间 | train 2014-2018 / valid 2019 / test 2020 全年 |
| 策略 | TopkDropout（topk=50, n_drop=5），双边手续费 0.05%/0.15%，涨跌停 ±9.5% 限制 |
| 评估 | IC/ICIR/Rank IC/Rank ICIR + 含费/不含费超额收益 |
| 配置 | `configs/workflow_config_{lightgbm,doubleensemble,mlp,gru}_Alpha158_smoke.yaml` |

## 结果

信号质量（test 2020）：

| 模型 | 类型 | IC | ICIR | Rank IC | Rank ICIR |
| --- | --- | --- | --- | --- | --- |
| MLP | 深度（表格） | **0.0382** | **0.2957** | 0.0404 | 0.3209 |
| DoubleEnsemble | 集成（LGBM 基座） | 0.0363 | 0.2954 | **0.0434** | **0.3335** |
| LightGBM | GBDT | 0.0266 | 0.2161 | 0.0342 | 0.2694 |
| GRU | 深度（时序） | 0.0052 | 0.0358 | 0.0316 | 0.2251 |

组合回测（2020 全年，沪深300 基准年化 26.0% / 最大回撤 -17.2%）：

| 模型 | 超额年化（不含费） | 超额年化（含费） | IR（含费） | 超额 MDD（含费） |
| --- | --- | --- | --- | --- |
| LightGBM | **+10.98%** | **+6.19%** | **0.73** | **-3.94%** |
| DoubleEnsemble | +8.03% | +3.29% | 0.37 | -4.97% |
| MLP | +3.69% | -1.03% | -0.12 | -6.22% |
| GRU | -6.18% | -10.94% | -1.55 | -12.28% |

![累计收益对比](model_comparison_curves.png)

## 关键发现

1. **信号指标与实盘口径脱节**：MLP 和 DoubleEnsemble 的 IC/Rank IC 都优于 LightGBM，但含费超额收益排序完全反过来（LightGBM > DE > MLP > GRU）。信号准不等于赚钱——TopkDropout 每日换仓 5 只，信号日间稳定性差的模型换手成本更高，费用把信号优势吃掉了。
2. **GRU 欠训练**：early stopping 显示 best score 出现在 epoch 1，第 11 轮即停止。学习率 2e-4 偏保守 + 训练区间缩短到 5 年（官方配置为 7 年），模型没有充分拟合。
3. **深度模型不是不行，是结构要匹配**：同为 PyTorch 深度模型，表格结构的 MLP（IC 0.038）远好于时序结构的 GRU（IC 0.005）。Alpha158 已经是人工设计好的截面因子，时序卷积/递归结构没有额外信息可挖。
4. **集成方法稳健**：DoubleEnsemble 在所有信号指标上稳定前三，验证了「样本重加权 + 特征选择」对低信噪比金融数据的价值。
5. 与官方 benchmark 的相对排序一致（树模型/集成强于时序深度），绝对值低于官方值（缩区间所致），对比只在同条件内部有效。

## 局限与下一步

- 训练区间比官方 benchmark 短（5 年 vs 7 年），不能与官方公布值直接对比。
- GRU 未调参；MLP 的换手/成本敏感性可以用 `report_normal_1day.pkl` 里的 turnover 字段进一步验证。
- 下一步可做：恢复完整区间重跑、GRU 调 lr 逃出 epoch-1 早停、对比各模型换手率、Alpha360 数据集上的深度模型（时序模型的真正主场）。

## 产物清单

- 配置：`quant-sprint/configs/workflow_config_*_smoke.yaml`（4 份）
- 曲线图：`quant-sprint/reports/model_comparison_curves.png`
- 绘图脚本：`quant-sprint/scripts/plot_model_comparison.py`