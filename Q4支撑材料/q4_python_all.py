# -*- coding: utf-8 -*-
# Q4: 女胎异常判定（AB列）—— GroupKFold + XGBoost（不平衡）+ 代价阈值 + 概率校准 + SHAP + 基线L1 Logit

import json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve, \
    precision_recall_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression

from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# ----------------- 路径与常量 -----------------
DATA_XLSX = "清洗结果_Q1.xlsx"
# 如果你的女胎 sheet 名不同，这里自动探测；若都没有则从全表里用 Y列为空来筛
FEMALE_SHEET_CAND = ["女胎_清洗版", "女胎", "Female", "女胎数据"]
OUT = Path("Q4_输出");
OUT.mkdir(exist_ok=True)

# 代价：错过异常（FN）远比误报（FP）严重，数值可按你们报告口径改
COST_FN = 10.0
COST_FP = 1.0

RANDOM_STATE = 42
N_SPLITS = 5


# ----------------- 列名规范与别名 -----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str)
                  .str.strip()
                  .str.replace("\u00a0", "", regex=False)
                  .str.replace("\u3000", "", regex=False))
    return df


def force_id_col(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    id_alias = ["孕妇ID", "孕妇代码", "孕妇eid", "样本序号"]
    hit = [c for c in id_alias if c in df.columns]
    if not hit:
        raise KeyError(f"未找到ID列（候选：{id_alias}），现有列：{list(df.columns)[:20]}")
    if hit[0] != "孕妇ID":
        df = df.rename(columns={hit[0]: "孕妇ID"})
    df["孕妇ID"] = df["孕妇ID"].astype(str)
    return df


def smart_read_female(xlsx_path: str) -> pd.DataFrame:
    # 1) 优先读"女胎"sheet；2) 否则读第一个 sheet 并用 Y 列为空筛选
    xl = pd.ExcelFile(xlsx_path)
    sheet = None
    for nm in xl.sheet_names:
        if any(key in nm for key in FEMALE_SHEET_CAND):
            sheet = nm;
            break
    df = pd.read_excel(xlsx_path, sheet_name=sheet) if sheet else pd.read_excel(xlsx_path)
    df = normalize_columns(df)
    # 统一典型别名
    alias = {
        "孕妇BMI": "体质指数",
        "孕周（周）": "孕周_周",
        "Y染色体浓度": "Y浓度",
        "X染色体浓度": "X浓度",
        "13号染色体的Z值": "13号Z值",
        "18号染色体的Z值": "18号Z值",
        "21号染色体的Z值": "21号Z值",
        "X染色体的Z值": "X染色体Z值",
        "13号染色体的GC含量": "13号GC含量",
        "18号染色体的GC含量": "18号GC含量",
        "21号染色体的GC含量": "21号GC含量",
        "被过滤掉读段数的比例": "过滤比例",
        "在参考基因组上比对的比例": "比对比例",
        "重复读段的比例": "重复比例",
        "原始测序数据的总读段数": "原始读段数",
        "检测出的13号，18号，21号染色体非整倍体": "染色体的非整倍体",
    }
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})
    
    # ★ 新增：去掉重名列（"体质指数"/"孕妇BMI"等映射后很容易重名）
    df = dedupe_same_named_columns(df)
    
    df = force_id_col(df)

    # 仅保留**女胎**：Y 浓度/列为空（U/V 在女胎为空）
    y_cols = [c for c in ["Y浓度", "U", "Y 染色体浓度"] if c in df.columns]
    if y_cols:
        mask_female = df[y_cols[0]].isna()
        df = df[mask_female].copy()
    # 若没有对应列，默认全是女胎（按你们数据组织方式）
    return df


# ----------------- 构造标签与特征 -----------------
def build_label(df: pd.DataFrame) -> pd.Series:
    cand = [c for c in ["染色体的非整倍体", "AB", "非整倍体"] if c in df.columns]
    if not cand:
        raise KeyError("未找到 AB/非整倍体 列，请检查清洗表。")
    labcol = cand[0]
    # 只要含有 13/18/21 的任一异常标记即为 1
    y = df[labcol].astype(str).str.strip().replace({"nan": ""})
    y = y.apply(lambda s: 1 if (("13" in s) or ("18" in s) or ("21" in s)) else 0)
    return y.astype(int)


def pick_features(df: pd.DataFrame) -> list:
    # 题意要求：Z 值（13/18/21/X）、GC、读段/比例、BMI、年龄、X浓度等；禁止引入Y
    pref = [
        "13号Z值", "18号Z值", "21号Z值", "X染色体Z值",
        "13号GC含量", "18号GC含量", "21号GC含量", "GC含量",
        "原始读段数", "比对比例", "重复比例", "过滤比例",
        "体质指数", "年龄", "X浓度",
        # 可选的检测质量主成分
        "检测质量主成分1", "检测质量主成分2",
        # 可选：孕周（若你想纳入当次测序周数对难度的影响）
        "孕周_周"
    ]
    feats = [c for c in pref if c in df.columns]
    if not feats:
        raise RuntimeError("没有可用特征列，请检查列名是否匹配。")
    return feats


def dedupe_same_named_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    合并同名列：
    - 对于每个重复的列名，按列方向 bfill 取"最靠左的非空值"
    - 保留第一列的位置，其他同名列丢弃
    """
    if not df.columns.duplicated(keep=False).any():
        return df

    # 先保留第一列出现的位置
    out = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

    # 找出所有重复名，做合并赋值
    dup_names = pd.unique(df.columns[df.columns.duplicated(keep=False)])
    for name in dup_names:
        cols = df.columns[df.columns == name]
        merged = df.loc[:, cols].bfill(axis=1).iloc[:, 0]
        out[name] = merged

    # 仅做一次提示，便于你确认哪些列重复
    print(f"[WARN] 发现并合并了同名列：{list(map(str, dup_names))}")
    return out


def to_numeric_safe(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    更健壮的数值化：
    - 忽略缺失列
    - 若选择到的是 DataFrame（因为列名重复），先合并为一列再转数值
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        obj = df[c]
        if isinstance(obj, pd.DataFrame):
            obj = obj.bfill(axis=1).iloc[:, 0]
        df[c] = pd.to_numeric(obj, errors="coerce")
    return df


# ----------------- 训练与评估（分组CV + 概率校准 + 代价阈值） -----------------
def train_xgb_groupcv(X, y, groups, random_state=RANDOM_STATE):
    n_pos = int(y.sum());
    n_neg = int((1 - y).sum())
    spw = max(n_neg / max(n_pos, 1), 1.0)  # scale_pos_weight

    cv = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.zeros(len(y), dtype=float)
    metrics = []
    final_model = None

    for k, (tr, va) in enumerate(cv.split(X, y, groups)):
        # 每个fold创建新的模型实例，避免状态污染
        model = XGBClassifier(
            n_estimators=600, learning_rate=0.03,
            max_depth=4, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.5, min_child_weight=1.0,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=random_state, tree_method="hist",
            scale_pos_weight=spw
        )
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:, 1]
        oof_pred[va] = p
        fold = {
            "fold": k + 1,
            "roc_auc": roc_auc_score(y[va], p) if len(np.unique(y[va])) > 1 else np.nan,
            "pr_auc": average_precision_score(y[va], p) if len(np.unique(y[va])) > 1 else np.nan,
            "brier": brier_score_loss(y[va], p),
            "logloss": log_loss(y[va], p, labels=[0, 1])
        }
        metrics.append(fold)
        if final_model is None:
            final_model = model

    met = pd.DataFrame(metrics)
    return final_model, oof_pred, met


def fit_isotonic_on_oof(oof_pred, y):
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    ir.fit(oof_pred, y)
    return ir


def choose_threshold_by_cost(prob, y, c_fn=COST_FN, c_fp=COST_FP):
    grid = np.linspace(0.01, 0.99, 99)
    best = None
    for t in grid:
        yhat = (prob >= t).astype(int)
        cm = confusion_matrix(y, yhat, labels=[0, 1])  # [[TN,FP],[FN,TP]]
        TN, FP, FN, TP = cm.ravel()
        cost = c_fn * FN + c_fp * FP
        if (best is None) or (cost < best["cost"]):
            best = {"thr": float(t), "TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP), "cost": float(cost)}
    return best


def plot_calibration(prob_raw, prob_cal, y, path_png):
    def calib_points(p, y, bins=10):
        qs = np.quantile(p, np.linspace(0, 1, bins + 1))
        idx = np.digitize(p, qs[1:-1], right=True)
        df = pd.DataFrame({"p": p, "y": y, "bin": idx})
        g = df.groupby("bin", as_index=False).agg(mean_p=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
        return g

    gr = calib_points(prob_raw, y);
    gc = calib_points(prob_cal, y)
    plt.figure(figsize=(7, 6), dpi=140)
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=.6, label="理想")
    plt.plot(gr["mean_p"], gr["obs"], "-o", label="未校准")
    plt.plot(gc["mean_p"], gc["obs"], "-o", label="Isotonic后")
    plt.xlabel("预测概率");
    plt.ylabel("实际阳性率");
    plt.title("Q4 可靠度曲线")
    plt.legend();
    plt.tight_layout();
    plt.savefig(path_png);
    plt.close()


def plot_roc_pr(prob, y, thr, path_png):
    fpr, tpr, _ = roc_curve(y, prob)
    prec, rec, _ = precision_recall_curve(y, prob)
    plt.figure(figsize=(12, 5), dpi=140)

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr);
    plt.plot([0, 1], [0, 1], "k--", alpha=.5)
    plt.title("ROC (AUC=%.3f)" % roc_auc_score(y, prob));
    plt.xlabel("FPR");
    plt.ylabel("TPR")
    plt.scatter([0], [0], s=1)  # 防止空图

    plt.subplot(1, 2, 2)
    plt.plot(rec, prec);
    plt.title("PR (AP=%.3f)" % average_precision_score(y, prob))
    plt.xlabel("Recall");
    plt.ylabel("Precision")
    plt.tight_layout();
    plt.savefig(path_png);
    plt.close()


# ----------------- 主流程 -----------------
def main():
    print("[1/7] 读取女胎数据…")
    df = smart_read_female(DATA_XLSX)
    y = build_label(df)

    feats = pick_features(df)
    # 去掉特征名重复
    feats = list(dict.fromkeys(feats))
    df = to_numeric_safe(df, feats)
    df = df.dropna(subset=feats).copy()

    # 分组（防泄漏）
    groups = force_id_col(df)["孕妇ID"].astype("category").cat.codes.to_numpy()
    X = df[feats].to_numpy(dtype=float)
    yv = y.loc[df.index].to_numpy(dtype=int)

    print(f"样本量={len(yv)}, 阳性(异常)={int(yv.sum())}, 阴性={int((1 - yv).sum())}, 特征数={len(feats)}")

    print("[2/7] 分组CV训练 XGBoost（含不平衡权重）…")
    base_model, oof_raw, met = train_xgb_groupcv(X, yv, groups)
    met.to_csv(OUT / "Q4_metrics_cv.csv", index=False, encoding="utf-8-sig")
    print(met.describe().round(4))

    print("[3/7] 概率校准（Isotonic, 基于OOF）…")
    iso = fit_isotonic_on_oof(oof_raw, yv)
    oof_cal = iso.predict(oof_raw)

    # 可靠性 + ROC/PR
    plot_calibration(oof_raw, oof_cal, yv, OUT / "Q4_calibration_curve.png")
    plot_roc_pr(oof_cal, yv, 0.5, OUT / "Q4_roc_pr_curves.png")

    # 代价驱动阈值
    best = choose_threshold_by_cost(oof_cal, yv, COST_FN, COST_FP)
    with open(OUT / "Q4_best_threshold.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps({"COST_FN": COST_FN, "COST_FP": COST_FP, "best": best}, ensure_ascii=False, indent=2))
    print(
        f"[阈值] 代价最小阈值 = {best['thr']:.3f}，期望代价={best['cost']:.1f}，混淆矩阵(TN,FP,FN,TP)=({best['TN']},{best['FP']},{best['FN']},{best['TP']})")

    # 输出 OOF 预测
    oof_df = df[["孕妇ID"]].copy()
    oof_df["y_true"] = yv
    oof_df["p_raw"] = oof_raw
    oof_df["p_cal"] = oof_cal
    oof_df["y_pred_best"] = (oof_cal >= best["thr"]).astype(int)
    oof_df.to_csv(OUT / "Q4_pred_oof.csv", index=False, encoding="utf-8-sig")

    print("[4/7] 拟合全量模型并保存…")
    base_model.fit(X, yv)
    # 存模型（json）与校准器（pickle）
    Path(OUT / "Q4_model_xgb.json").write_text(base_model.get_booster().save_raw("json").decode(), encoding="utf-8")
    with open(OUT / "Q4_calibrator_isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)

    print("[5/7] SHAP 全局解释…")
    try:
        explainer = shap.TreeExplainer(base_model, feature_perturbation="tree_path_dependent")
        # 取最多 2000 条做图
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), size=min(2000, len(X)), replace=False)
        shap_values = explainer.shap_values(X[idx])
        shap.summary_plot(shap_values, features=X[idx], feature_names=feats, show=False)
        plt.tight_layout();
        plt.savefig(OUT / "Q4_shap_summary.png", dpi=140);
        plt.close()
    except Exception as e:
        print("SHAP 绘图失败：", e)

    print("[6/7] 基线对照：L1 Logistic（可解释）…")
    logit = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            penalty="l1", solver="saga", class_weight="balanced",
            C=1.0, max_iter=5000, random_state=RANDOM_STATE
        )
    )
    cv = GroupKFold(n_splits=N_SPLITS)
    oof_logit = np.zeros(len(yv))
    for k, (tr, va) in enumerate(cv.split(X, yv, groups)):
        logit.fit(X[tr], yv[tr])
        oof_logit[va] = logit.predict_proba(X[va])[:, 1]
    # 系数导出
    logit.fit(X, yv)
    clf = logit.named_steps["logisticregression"]
    coefs = pd.DataFrame({"feature": feats, "coef": clf.coef_[0]}).sort_values("coef", key=np.abs, ascending=False)
    coefs.to_csv(OUT / "Q4_logit_coefs.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(8, 6), dpi=140)
    topk = coefs.head(20).sort_values("coef")
    plt.barh(topk["feature"], topk["coef"])
    plt.title("基线 L1 Logistic 系数（Top20 | 可正可负）")
    plt.tight_layout();
    plt.savefig(OUT / "Q4_logit_coef_plot.png");
    plt.close()

    print("[7/7] 写入简要 readme…")
    with open(OUT / "Q4_readme.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Q4 女胎异常判定（AB列）\n"
            f"样本={len(yv)}, 阳性={int(yv.sum())}\n"
            f"CV 指标（均值）: ROC-AUC={met['roc_auc'].mean():.3f}, PR-AUC={met['pr_auc'].mean():.3f}, "
            f"Brier={met['brier'].mean():.4f}, LogLoss={met['logloss'].mean():.4f}\n"
            f"阈值(代价最小): {best['thr']:.3f}, 成本(FN:{COST_FN}, FP:{COST_FP}) → 期望代价={best['cost']:.1f}\n"
            f"混淆矩阵(TN,FP,FN,TP)=({best['TN']},{best['FP']},{best['FN']},{best['TP']})\n"
            f"主要产物：Q4_pred_oof.csv / Q4_metrics_cv.csv / Q4_calibration_curve.png / "
            f"Q4_roc_pr_curves.png / Q4_shap_summary.png / Q4_logit_*.csv/png / "
            f"Q4_model_xgb.json / Q4_calibrator_isotonic.pkl\n"
        )
    print("✅ 全部完成，见目录：", OUT.resolve())


if __name__ == "__main__":
    main()