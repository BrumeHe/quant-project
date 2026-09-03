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

# 模型对比实验报告（qlib / Alpha158 / csi300）

> 数据源：社区 chenditc/investment_data（qlib 官方数据集已停用）。
> 所有实验同一股票池（csi300）、同一调仓策略（TopkDropout topk=50, n_drop=5）、同一手续费（买 0.05% / 卖 0.15% / 最低 5 元）、同一 label（T+1 买 T+2 卖收益）。

## 1. 实验设计（两轮，区间明示）

| 轮次 | 目的 | train | valid | test |
| --- | --- | --- | --- | --- |
| 小区间轮（smoke） | 链路验证 + 控制变量对比 | 2014-01 ~ 2018-12 | 2019 | 2020 |
| 完整区间轮（full，对齐官方） | 对齐官方 benchmark 的完整复现 | 2008-01 ~ 2014-12 | 2015 ~ 2016 | 2017-01 ~ 2020-08 |

## 2. 小区间轮结果（test=2020）

| 模型 | IC | ICIR | Rank IC | Rank ICIR | 超额年化(含费) | 信息比率(含费) | 超额最大回撤(含费) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.0266 | 0.2161 | 0.0342 | 0.2694 | 6.19% | 0.7306 | -3.94% |
| MLP | 0.0382 | 0.2957 | 0.0404 | 0.3209 | -1.03% | -0.1220 | -6.22% |
| DoubleEnsemble | 0.0363 | 0.2954 | 0.0434 | 0.3335 | 3.29% | 0.3731 | -4.97% |
| GRU | 0.0052 | 0.0358 | 0.0316 | 0.2251 | -10.94% | -1.5468 | -12.28% |

![小区间累计超额收益](figures/cum_excess_smoke.png)

## 3. 完整区间轮结果（test=2017-01 ~ 2020-08）

| 模型 | IC | ICIR | Rank IC | Rank ICIR | 超额年化(含费) | 信息比率(含费) | 超额最大回撤(含费) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.0470 | 0.3816 | 0.0487 | 0.4057 | 11.06% | 1.3051 | -8.58% |
| MLP | 0.0354 | 0.2768 | 0.0427 | 0.3354 | 5.25% | 0.6642 | -10.74% |
| DoubleEnsemble | 0.0540 | 0.4289 | 0.0511 | 0.4141 | 12.86% | 1.4358 | -8.39% |

官方 benchmark 对照（同 Alpha158 / csi300 / test 区间）：

| 模型 | 官方 IC | 本轮 IC | 偏差 | 官方 Rank IC | 本轮 Rank IC | 偏差 |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.0448 | 0.0470 | +0.0022 | 0.0469 | 0.0487 | +0.0018 |
| MLP | 0.0376 | 0.0354 | -0.0022 | 0.0429 | 0.0427 | -0.0002 |
| DoubleEnsemble | 0.0521 | 0.0540 | +0.0019 | 0.0502 | 0.0511 | +0.0009 |

![完整区间累计超额收益](figures/cum_excess_full.png)

## 4. 两层结论

### 4.1 小区间轮 vs 官方：不可比，也不用比

- 时间段不同：train/valid/test 完全不同，IC 是「模型 × 市场环境」的联合结果，不是模型固有属性。
- 小区间轮的价值在于链路验证与同区间内的控制变量对比，不作为复现对照。
- 附带发现（regime 敏感性）：LightGBM / DoubleEnsemble 的 IC 跨区间波动约 0.02，MLP 跨区间最稳（0.0382 vs 0.0354）——树模型对训练区间市场环境的敏感度高于简单神经网络。

### 4.2 完整区间轮 vs 官方：复现成功

- 同配置同区间下，IC 偏差在千分位量级（±0.002），模型排序与官方一致（DoubleEnsemble > LightGBM > MLP）。
- 偏差来源：① 数据源不同（官方数据集停用，社区数据复权/清洗/成分股快照口径不同）；② 官方表格本身是含随机性模型 20 次运行的均值 ± 标准差；③ 依赖库版本（LightGBM 4.7 / numpy 2.2）；④ GPU/cuDNN 非确定性。
- GRU 完整区间轮因笔记本内存限制（16GB）暂未跑出结果，smoke 区间结果见第 2 节；后续可用 kernels=2 单独补跑。

## 5. 复现命令

```bash
# 小区间轮
MLFLOW_ALLOW_FILE_STORE=true .venv-qlib/Scripts/qrun.exe configs/workflow_config_<model>_Alpha158_smoke.yaml
# 完整区间轮
MLFLOW_ALLOW_FILE_STORE=true .venv-qlib/Scripts/qrun.exe configs/full/workflow_config_<model>_Alpha158_full.yaml
```
