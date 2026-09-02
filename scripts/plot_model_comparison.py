# 从 mlruns 读取 LightGBM 与 GRU 两次实验的回测明细，画累计收益对比曲线
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MLRUNS = ROOT / "mlruns"

LABEL_MAP = {"LGBModel": "LightGBM", "GRU": "GRU", "DEnsembleModel": "DoubleEnsemble", "DNNModelPytorch": "MLP"}

def iter_completed_runs():
    for exp_dir in MLRUNS.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in ("0", ".trash"):
            continue
        for rec_dir in exp_dir.iterdir():
            report_pkl = rec_dir / "artifacts" / "portfolio_analysis" / "report_normal_1day.pkl"
            model_param = rec_dir / "params" / "model.class"
            if not report_pkl.exists() or not model_param.exists():
                continue  # 跳过被中断的 run
            yield model_param.read_text().strip(), report_pkl

def main():
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plotted = []
    for model_class, report_pkl in iter_completed_runs():
        label = LABEL_MAP.get(model_class, model_class)
        report = pd.read_pickle(report_pkl)
        ret_with_cost = report["return"] - report["cost"]
        bench = report["bench"]
        excess = ret_with_cost - bench
        cum_ret = (1 + ret_with_cost).cumprod()
        cum_excess = (1 + excess).cumprod()
        axes[0].plot(cum_ret.index, cum_ret.values, label=label)
        axes[1].plot(cum_excess.index, cum_excess.values, label=label)
        plotted.append(label)
        print(f"{label}: final cum_return={cum_ret.iloc[-1]:.3f}, final cum_excess={cum_excess.iloc[-1]:.3f}")

    # 基准曲线用任一 run 的 bench
    bench = pd.read_pickle(report_pkl)["bench"]
    axes[0].plot(bench.index, (1 + bench).cumprod().values, label="CSI300 benchmark", linestyle="--", color="gray")

    axes[0].set_title("Portfolio cumulative return 2020 (with cost, csi300 topk50)")
    axes[1].set_title("Cumulative excess return vs CSI300 (with cost)")
    axes[0].legend(); axes[1].legend()
    axes[1].set_xlabel("date")
    fig.tight_layout()
    out = ROOT / "reports" / "model_comparison_curves.png"
    fig.savefig(out, dpi=150)
    print("saved:", out, "| models plotted:", plotted)

if __name__ == "__main__":
    main()
