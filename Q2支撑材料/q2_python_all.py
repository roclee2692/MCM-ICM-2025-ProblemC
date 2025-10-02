# Q2 · person-period + LogisticGAM(离散时间) + F(t|BMI)
# 依赖：numpy pandas openpyxl matplotlib scikit-learn pygam xlsxwriter scipy
# 运行：python q2_python_all.py

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from pygam import LogisticGAM, s, te
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
# ========== 基本参数 ==========
DATA_XLSX = "清洗结果_Q1.xlsx"     # 男胎清洗表
SHEET     = "男胎_清洗版"
OUT_DIR   = Path("输出图"); OUT_DIR.mkdir(exist_ok=True)
V_THRESH  = 0.04

# 画图中文
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 工具函数 ==========
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去不可见空格、全角空格；去两端空白"""
    df = df.copy()
    df.columns = (df.columns.astype(str)
        .str.strip()
        .str.replace('\u00a0','', regex=False)  # NBSP
        .str.replace('\u3000','', regex=False)  # 全角空格
    )
    return df

def unify_keys_and_alias(df: pd.DataFrame) -> pd.DataFrame:
    """把主键统一成『孕妇ID』；常见别名做一次映射"""
    df = normalize_columns(df)
    # 主键统一
    id_aliases = ["孕妇ID", "孕妇代码"]
    id_found = next((c for c in id_aliases if c in df.columns), None)
    if id_found is None:
        raise KeyError(f"未找到主键列（候选: {id_aliases}）。当前表头前30个：{list(df.columns)[:30]}")
    if id_found != "孕妇ID":
        df = df.rename(columns={id_found: "孕妇ID"})

    # 其他别名（按你们清洗表习惯可扩）
    if "孕妇BMI" in df.columns and "体质指数" not in df.columns:
        df = df.rename(columns={"孕妇BMI":"体质指数"})
    if "Y染色体浓度" in df.columns and "Y浓度" not in df.columns:
        df = df.rename(columns={"Y染色体浓度":"Y浓度"})
    return df

def build_person_period(dat: pd.DataFrame, col_id, col_week, col_bmi, col_v, v_thresh=0.04):
    """
    构造 person-period：人×整数周，保留『col_id』，并标注首次达标事件。
    返回列：['孕妇ID','周整数','体质指数','首次达标周','event']
    """
    d = dat.copy()
    # 整数周
    d["周整数"] = np.floor(pd.to_numeric(d[col_week], errors="raise")).astype(int)
    # 仅留必要列
    d = (d[[col_id, "周整数", col_bmi, col_v]]
         .dropna(subset=[col_id, "周整数", col_bmi, col_v])
         .sort_values([col_id, "周整数"]))

    # 每人首次达标周
    first_hit = (d.loc[d[col_v] >= v_thresh, [col_id, "周整数"]]
                   .groupby(col_id, as_index=False)["周整数"].min()
                   .rename(columns={"周整数":"首次达标周"}))

    # 人×周（去重） + 首次达标周
    pp = (d[[col_id, "周整数", col_bmi]].drop_duplicates()
          .merge(first_hit, on=col_id, how="left"))

    # 仅保留首次达标之前（含当周）的行 —— 并**强制把ID写回去**
    def _keep_rows(g: pd.DataFrame):
        fh = g["首次达标周"].iloc[0]
        if pd.isna(fh):
            g2 = g
        else:
            g2 = g[g["周整数"] <= int(fh)]
        # 关键：无论 pandas 怎么玩，手动把 ID 列补回去
        g2[col_id] = g[col_id].iloc[0]
        return g2

    # 不用 include_groups=False，且 as_index=False，最大化兼容并保留列
    try:
        pp = pp.groupby(col_id, as_index=False, group_keys=False).apply(_keep_rows)
    except TypeError:
        pp = pp.groupby(col_id, group_keys=False).apply(_keep_rows).reset_index(drop=True)

    # 首次达标事件
    pp["event"] = ((pp["首次达标周"].notna()) & (pp["周整数"] == pp["首次达标周"])).astype(int)

    # ——强制自检：ID 必须在——
    assert col_id in pp.columns, f"person-period 丢失ID列：{col_id}。现有列：{pp.columns.tolist()}"

    # 统一列顺序
    return pp[[col_id, "周整数", col_bmi, "首次达标周", "event"]]


# ========== 主流程 ==========
def main():
    # 1) 读取 & 统一列名
    df_full = pd.read_excel(DATA_XLSX, sheet_name=SHEET)
    df_full = unify_keys_and_alias(df_full)

    COL_ID   = "孕妇ID"
    COL_WEEK = "孕周_周"
    COL_BMI  = "体质指数"
    COL_V    = "Y浓度"

    # 基础列自检
    for c in [COL_ID, COL_WEEK, COL_BMI, COL_V]:
        if c not in df_full.columns:
            raise KeyError(f"缺列：{c}；表头前30个：{list(df_full.columns)[:30]}")

    # 2) person-period
    dat_core = df_full[[COL_ID, COL_WEEK, COL_BMI, COL_V]].copy()
    pp = build_person_period(dat_core, COL_ID, COL_WEEK, COL_BMI, COL_V, V_THRESH)

    # 写 CSV（确认含『孕妇ID』）
    pp.to_csv("Q2_person_period.csv", index=False, encoding="utf-8-sig")
    print("[OK] 写出 Q2_person_period.csv，列：", list(pp.columns))

    # 3) 合并『扩展因子』（可缺省，自动忽略）
    qc_cols = [
        "年龄", "IVF妊娠", "原始读段数", "在参考基因组上比对的比例",
        "重复读段的比例", "GC含量", "被过滤掉读段数的比例"
    ]
    avail_qc = [c for c in qc_cols if c in df_full.columns]
    if avail_qc:
        # 用『孕妇ID + 周整数』做键合并（先把孕周_周 → 周整数）
        df_merge = (df_full[[COL_ID, COL_WEEK] + avail_qc]
                    .rename(columns={COL_WEEK: "周整数"})
                    .drop_duplicates(subset=[COL_ID, "周整数"]))

        # ——关键：两边的键统一为 int，避免 “int 和 float 合并” 警告/错配
        df_merge["周整数"] = pd.to_numeric(df_merge["周整数"], errors="coerce").astype(int)
        pp["周整数"] = pd.to_numeric(pp["周整数"], errors="coerce").astype(int)

        # 合并前自检
        for k in [COL_ID, "周整数"]:
            assert k in pp.columns, f"pp 缺列：{k}（现有：{pp.columns.tolist()}）"
            assert k in df_merge.columns, f"df_merge 缺列：{k}（现有：{df_merge.columns.tolist()}）"

        pp = pp.merge(df_merge, on=[COL_ID, "周整数"], how="left")

        if "原始读段数" in pp.columns:
            pp["log原始读段数"] = np.log1p(pp["原始读段数"])
        if "IVF妊娠" in pp.columns:
            pp["IVF_指示"] = (pp["IVF妊娠"].astype(str) != "自然受孕").astype(int)
        print(f"[INFO] 扩展因子合并：{avail_qc}")
    else:
        print("[INFO] 未发现扩展因子列，仅用 孕周+BMI。")

    # === 关闭扩展因子（仅用孕周 + BMI） =========================================
    qc_cols: list[str] = []  # 不使用任何 QC 扩展因子
    avail_qc: list[str] = []  # 强制无可用列
    print("[INFO] 仅使用 孕周+BMI（无扩展因子）。")
    # =========================================================================

    # 4) 组装特征并拟合 LogisticGAM(离散时间危险率)
    # === 特征列（周整数 + BMI）=====================================================
    # base_feats  = ["周整数", COL_BMI]
    # extra_feats = ["年龄","log原始读段数","在参考基因组上比对的比例",
    #                "重复读段的比例","GC含量","被过滤掉读段数的比例","IVF_指示"]
    # FEATS = base_feats + [f for f in extra_feats if f in pp.columns]

    FEATS: list[str] = ["周整数", COL_BMI]  # 仅保留两列特征

    # 数值化
    for f in FEATS:
        pp[f] = pd.to_numeric(pp[f], errors="coerce")
    pp = pp.dropna(subset=FEATS + ["event"])
    X = pp[FEATS].to_numpy(dtype=float)
    y = pp["event"].to_numpy(dtype=int)
    groups = pp[COL_ID].astype("category").cat.codes.to_numpy()

    ix_week = FEATS.index("周整数")
    ix_bmi  = FEATS.index(COL_BMI)
    terms = s(ix_week, n_splines=10) + s(ix_bmi, n_splines=7) + te(ix_week, ix_bmi, n_splines=[5,5])
    for i, name in enumerate(FEATS):
        if name in ("周整数", COL_BMI):
            continue
        terms = terms + s(i, n_splines=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam = LogisticGAM(terms, verbose=False)
        # 给模型对象设置迭代上限（旧版 pyGAM 只认属性，不认 fit 的参数）
        try:
            gam.max_iter = 5000
        except Exception:
            pass
        gam.fit(X, y)

    # 分组 5 折
    cv = GroupKFold(n_splits=5)
    maes = []
    for tr, te_idx in cv.split(X, y, groups):
        m = LogisticGAM(terms, verbose=False)
        try:
            m.max_iter = 5000
        except Exception:
            pass
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te_idx])
        maes.append(np.mean(np.abs(p - y[te_idx])))
    print(f"[CV] MAE by group-5fold: mean={np.mean(maes):.4f}, std={np.std(maes):.4f}")

    # 5) 网格预测：h → S → F
    # ---------- 4) 预测网格上的 h，叠成 S、F ----------
    t_min = max(9, int(pp["周整数"].min()))
    t_max = min(26, int(pp["周整数"].max()))
    b_min = float(np.floor(pp[COL_BMI].min()))
    b_max = float(np.ceil(pp[COL_BMI].max()))

    t_seq = np.arange(t_min, t_max + 1, 1)
    b_seq = np.arange(b_min, b_max + 0.001, 0.5)

    grid = pd.DataFrame([(t, b) for b in b_seq for t in t_seq],
                        columns=["周整数", COL_BMI])

    # 其他扩展因子取中位数（保持可解释）
    med = {f: float(np.nanmedian(pp[f])) for f in FEATS if f not in ("周整数", COL_BMI)}
    for f, v in med.items():
        grid[f] = v

    grid_X = grid[FEATS].to_numpy(dtype=float)
    haz = gam.predict_proba(grid_X)  # h(t,BMI)

    df_grid = grid.copy()
    df_grid["haz"] = haz

    # ——关键修复：分组 apply 时，显式把分组列写回去 & 不用 include_groups=False
    def _acc(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("周整数").copy()
        # 保住体质指数列（有些 pandas 版本会把分组键去掉）
        g[COL_BMI] = g[COL_BMI].iloc[0]
        g["S"] = (1 - g["haz"]).cumprod()
        g["F"] = 1 - g["S"]
        return g

    df_grid = (
        df_grid
        .sort_values([COL_BMI, "周整数"])
        .groupby(COL_BMI, as_index=False, group_keys=False)  # 不加 include_groups
        .apply(_acc)
        .reset_index(drop=True)
    )

    # 导出底图（现在一定有 '体质指数' 了）
    with pd.ExcelWriter("Q2_生存底图_python.xlsx", engine="xlsxwriter") as w:
        df_grid[["周整数", COL_BMI, "haz", "S", "F"]].to_excel(
            w, index=False, sheet_name="F_grid"
        )

    # 6) 可视化
    # F 热图 + 等高线
    pivot_F = df_grid.pivot(index=COL_BMI, columns="周整数", values="F")
    plt.figure(figsize=(9,6), dpi=150)
    im = plt.imshow(pivot_F.values, aspect="auto",
                    extent=[t_seq.min(), t_seq.max(), b_seq.min(), b_seq.max()],
                    origin="lower", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="P(达标 ≤ t)")
    Z = gaussian_filter(pivot_F.values, sigma=0.6)
    Xc, Yc = np.meshgrid(t_seq, b_seq)
    CS = plt.contour(Xc, Yc, Z, levels=[.5,.7,.8,.9], colors="white", linewidths=1)
    plt.clabel(CS, inline=True, fmt="%.1f")
    plt.xlabel("孕周（整周）"); plt.ylabel("体质指数")
    plt.title("到 t 周为止已达标概率 F(t|BMI)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "Q2_F热图_python.png"); plt.close()

    # 风险集人数热图
    pp2 = pp.sort_values([COL_ID, "周整数"]).copy()
    pp2["之前已达标"] = pp2.groupby(COL_ID)["event"].cumsum().shift(1).fillna(0)
    pp2["在风险集"] = (pp2["之前已达标"] == 0).astype(int)
    pp2["_BMI档"] = np.round(pp2[COL_BMI] * 1) / 1
    support = (pp2[pp2["在风险集"] == 1]
               .groupby(["_BMI档", "周整数"], as_index=False)
               .agg(风险集人数=("在风险集", "size")))
    sup_piv = support.pivot(index="_BMI档", columns="周整数", values="风险集人数").fillna(0)

    # 风险集热图 —— 彩色连续版
    plt.figure(figsize=(9, 6), dpi=150)
    im = plt.imshow(
        sup_piv.values, aspect="auto",
        extent=[sup_piv.columns.min(), sup_piv.columns.max(),
                sup_piv.index.min(), sup_piv.index.max()],
        origin="lower",
        cmap="viridis"  # ← 连续渐变（也可改 "plasma" / "cividis"）
    )
    plt.colorbar(im, label="风险集人数")
    plt.xlabel("孕周（整周）");
    plt.ylabel("体质指数（0.5 分档）")
    plt.title("风险集热图（每周仍在风险的人数）")
    plt.tight_layout();
    plt.savefig(OUT_DIR / "Q2_风险集热图.png");
    plt.close()

    # 支撑度≥3 的 F 掩膜热图
    grid_mask = df_grid.copy()
    grid_mask["_BMI档"] = np.round(grid_mask[COL_BMI] * 2) / 2
    grid_mask = grid_mask.merge(support, on=["_BMI档", "周整数"], how="left")
    grid_mask["风险集人数"] = grid_mask["风险集人数"].fillna(0)
    grid_mask["F_masked"] = np.where(grid_mask["风险集人数"] >= 3, grid_mask["F"], np.nan)
    pm = grid_mask.pivot(index="_BMI档", columns="周整数", values="F_masked")

    plt.figure(figsize=(9, 6), dpi=150)
    im = plt.imshow(pm.values, aspect="auto",
                    extent=[pm.columns.min(), pm.columns.max(), pm.index.min(), pm.index.max()],
                    origin="lower", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="P(达标≤t)（支撑度≥3）")
    plt.xlabel("孕周（整周）");
    plt.ylabel("体质指数（0.5分档）")
    plt.title("F(t|BMI)（支撑度≥3的区域）")
    plt.tight_layout();
    plt.savefig(OUT_DIR / "Q2_F热图_masked.png");
    plt.close()

    # 经验危险率（对照）
    def empirical_hazard(pp_src, center, band=1):
        sub = pp_src[(pp_src[COL_BMI] >= center - band) & (pp_src[COL_BMI] <= center + band)].copy()
        sub = sub.sort_values([COL_ID, "周整数"])
        sub["之前已达标"] = sub.groupby(COL_ID)["event"].cumsum().shift(1).fillna(0)
        sub["在风险集"] = (sub["之前已达标"] == 0).astype(int)
        tab = (sub[sub["在风险集"] == 1]
               .groupby("周整数", as_index=False)
               .agg(风险集=("在风险集", "size"),
                    首次达标=("event", "sum")))
        tab["经验h"] = tab["首次达标"] / tab["风险集"].where(tab["风险集"] > 0, np.nan)
        return tab

    for c in [30, 35, 40]:
        emp = empirical_hazard(pp, c, band=1)
        emp.to_csv(f"Q2_经验危险率_BMI{c}±1.csv", index=False, encoding="utf-8-sig")

    # 示例导出
    def nearest(arr, v): arr = np.asarray(arr); return arr[np.argmin(np.abs(arr-v))]
    b_seq = np.arange(float(np.floor(pp[COL_BMI].min())), float(np.ceil(pp[COL_BMI].max()))+0.001, 0.5)
    b_ex = nearest(b_seq, 30)
    demo = (df_grid[(df_grid[COL_BMI]==b_ex) & (df_grid["周整数"].between(10,16))]
            .loc[:, ["周整数","haz","S","F"]]
            .rename(columns={"周整数":"周", "haz":"周危险率h", "S":"仍未达标S", "F":"已达标F"}))
    demo.to_csv("Q2_示例_BMI30_10到16周_python.csv", index=False, encoding="utf-8-sig")

    print("✅ Done. 生成：")
    print("  - Q2_person_period.csv（含『孕妇ID』）")
    print("  - Q2_生存底图_python.xlsx")
    print("  - 输出图/Q2_F热图_python.png")
    print("  - 输出图/Q2_风险集热图.png")
    print("  - 输出图/Q2_F热图_masked.png")
    print("  - Q2_经验危险率_BMI30±1.csv / 35±1.csv / 40±1.csv")
    print("  - Q2_示例_BMI30_10到16周_python.csv")

if __name__ == "__main__":
    main()