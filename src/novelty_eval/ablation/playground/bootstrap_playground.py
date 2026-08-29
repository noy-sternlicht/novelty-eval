from scipy.stats import pearsonr
import numpy as np
from scipy.stats import bootstrap

# %%
# scipy example (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
n = 100
x = np.linspace(0, 10, n)
rng = np.random.default_rng()
y = x + rng.uniform(size=n)
print(pearsonr(x, y)[0])


def my_statistic(x, y, axis=-1):
    return pearsonr(x, y, axis=axis)[0]


res = bootstrap((x, y), my_statistic, paired=True, rng=rng)
print(res.confidence_interval)


# %%
def print_bootstrap_result(label, abs_diff, res, alternative="two-sided"):
    low, high = res.confidence_interval
    if alternative == "greater":
        is_significant = np.isfinite(low) and low > 0
    elif alternative == "less":
        is_significant = np.isfinite(high) and high < 0
    else:
        is_significant = np.isfinite(low) and np.isfinite(high) and (high < 0 or low > 0)
    low_str  = f"{low * 100:.2f}%"  if np.isfinite(low)  else ("-inf%" if low  < 0 else "+inf%")
    high_str = f"{high * 100:.2f}%" if np.isfinite(high) else ("-inf%" if high < 0 else "+inf%")
    print(
        f'{label} [{alternative}]: abs_diff: {abs_diff * 100:.2f}%, low: {low_str}, high: {high_str}, is_significant: {is_significant}')
    print('----')


def pairwise_bootstrap_test(ablation: list, base: list, gt: list, alternative: str = "two-sided"):
    # We want to see whether the **difference** in accuracies is statistically significant
    def acc_diff(ablation, base, axis=-1):
        accuracy_base = np.mean(base, axis=axis)
        accuracy_ablation = np.mean(ablation, axis=axis)
        return accuracy_ablation - accuracy_base

    ablation = np.array(ablation)
    base = np.array(base)
    gt = np.array(gt)

    # accuracy, 0.5 credit for ties
    x = np.where(ablation == gt, 1, np.where(ablation == 2, 0.5, 0))
    y = np.where(base == gt, 1, np.where(base == 2, 0.5, 0))
    res = bootstrap((x, y), acc_diff, paired=True, rng=rng, n_resamples=100000, alternative=alternative)
    abs_diff = np.mean(x) - np.mean(y)
    print_bootstrap_result('ACC (w/ ties)', abs_diff, res, alternative)

    # strict accuracy, no credit for ties.
    x = (ablation == gt).astype(int)
    y = (base == gt).astype(int)
    res = bootstrap((x, y), acc_diff, paired=True, rng=rng, n_resamples=100000, alternative=alternative)
    abs_diff = np.mean(x) - np.mean(y)
    print_bootstrap_result('STRICT ACC', abs_diff, res, alternative)


def make_pointwise_statistic(metric: str):
    def _counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
        tp = int(np.sum((pred >= 0.5) & (gt >= 0.5)))
        fp = int(np.sum((pred >= 0.5) & (gt < 0.5)))
        fn = int(np.sum((pred < 0.5) & (gt >= 0.5)))
        tn = int(np.sum((pred < 0.5) & (gt < 0.5)))
        return tp, fp, fn, tn

    def _compute(pred: np.ndarray, gt: np.ndarray) -> float:
        tp, fp, fn, tn = _counts(pred, gt)
        total = tp + fp + fn + tn
        if metric == "accuracy":
            return (tp + tn) / total if total else 0.0
        p_pos = tp / (tp + fp) if (tp + fp) else 0.0
        r_pos = tp / (tp + fn) if (tp + fn) else 0.0
        f1_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) else 0.0
        p_neg = tn / (tn + fn) if (tn + fn) else 0.0
        r_neg = tn / (tn + fp) if (tn + fp) else 0.0
        f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) else 0.0
        if metric == "f1_macro":
            return (f1_pos + f1_neg) / 2
        if metric == "f1_pos":
            return f1_pos
        if metric == "f1_neg":
            return f1_neg
        if metric == "precision_pos":
            return p_pos
        if metric == "precision_neg":
            return p_neg
        if metric == "recall_pos":
            return r_pos
        if metric == "recall_neg":
            return r_neg
        raise ValueError(f"Unknown metric: {metric}")

    def statistic(ablation_pred, base_pred, gt):
        metric_ablation = _compute(ablation_pred, gt)
        metric_base = _compute(base_pred, gt)
        return metric_ablation - metric_base

    return statistic


def pointwise_bootstrap_test(ablation: list, base: list, gt: list, alternative: str = "two-sided"):
    ablation = np.array(ablation)
    base = np.array(base)
    gt = np.array(gt)

    metrics = [
        # "accuracy",
        "f1_macro",
        "f1_pos",
        "f1_neg",
        # "precision_pos",
        # "precision_neg",
        # "recall_pos",
        # "recall_neg",
    ]

    for metric in metrics:
        statistic = make_pointwise_statistic(metric)
        abs_diff = statistic(ablation, base, gt)
        res = bootstrap(
            (ablation, base, gt),
            statistic,
            vectorized=False,
            paired=True,
            rng=rng,
            n_resamples=100000,
            alternative=alternative,
        )
        print_bootstrap_result(metric, abs_diff, res, alternative)



# %%

# ----- PAIRWISE bootstrap sanity -----#
# dummy data
# print("---Dummy Pairwise Data---")
# ablation = [1, 1, 2, 1, 0]  # 0: idea 0 wins, 1: idea 1 wins, 2: tie
# base = [0, 0, 2, 1, 1]
# gt = [0, 0, 1, 1, 0]
# pairwise_bootstrap_test(ablation, base, gt)

# real data
import pandas as pd

csv_path = 'output/ablation_sweeps/my_merge/bootstrap_debug_csvs/pairwise_low_judge_reasoning__pairwise__claude-opus-4-5.csv'
df = pd.read_csv(csv_path)
ablation = df['abl_pred'].tolist()
base = df['base_pred'].tolist()
gt = df['gt'].tolist()

print("---Real Pairwise Data---")
pairwise_bootstrap_test(ablation, base, gt, alternative="two-sided")  # "two-sided" | "less" | "greater"

# %%
# ----- POINTWISE bootstrap sanity -----#
# dummy data
# print("---Dummy Pointwise Data---")
# ablation = [1, 1, 1, 1, 0]  # 0: idea is novel, 1: idea is not novel
# base = [0, 0, 0, 1, 1]
# gt = [0, 0, 1, 1, 0]
# pointwise_bootstrap_test(ablation, base, gt)

# real data
import pandas as pd

csv_path = 'output/ablation_sweeps/my_merge/bootstrap_debug_csvs/pointwise_mec_k_1__pointwise__gpt-5_4.csv'
df = pd.read_csv(csv_path)
label_map = {'POSITIVE': 1, 'NEGATIVE': 0}
ablation = df['abl_pred'].astype(int).tolist()
base = df['base_pred'].astype(int).tolist()
gt = df['gt'].map(label_map).tolist()

print("---Real Pointwise Data---")
pointwise_bootstrap_test(ablation, base, gt, alternative="less")  # "two-sided" | "less" | "greater"