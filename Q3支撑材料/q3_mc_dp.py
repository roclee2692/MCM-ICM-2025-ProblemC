# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pygam import LogisticGAM, s, l, te
import os
import warnings
warnings.filterwarnings('ignore')

# 新增导入
from scipy.ndimage import gaussian_filter1d

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 全局参数
DATA_CSV = "Q3_person_period.csv"      # 输入文件（person-period格式）
OUT_DIR = "output"                     # 输出目录
V_THRESH = 5                          # 验证阈值

# 创建输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 固定随机种子
rng = np.random.default_rng(42)

def normalize_columns(df, cols):
    """对指定列进行标准化"""
    for col in cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def unify_keys_and_alias(df):
    """统一列名和别名"""
    rename_map = {
        "孕妇ID": "id",
        "体质指数": "BMI", 
        "周整数": "week",
        "首次达标周": "target_week",
        "event": "event"
    }
    
    df = df.rename(columns=rename_map)
    return df

def build_person_period(df, t_start=11, t_end=26):
    """构建person-period格式数据"""
    person_periods = []
    
    for person_id in df['id'].unique():
        person_data = df[df['id'] == person_id].copy()
        
        if person_data['target_week'].isna().all():
            # 未达标情况
            for t in range(t_start, t_end + 1):
                if t in person_data['week'].values:
                    row = person_data[person_data['week'] == t].iloc[0].copy()
                    row['event'] = 0
                else:
                    row = person_data.iloc[-1].copy()
                    row['week'] = t
                    row['event'] = 0
                person_periods.append(row)
        else:
            # 达标情况
            target = person_data['target_week'].iloc[0]
            for t in range(t_start, min(int(target) + 1, t_end + 1)):
                if t in person_data['week'].values:
                    row = person_data[person_data['week'] == t].iloc[0].copy()
                else:
                    row = person_data.iloc[0].copy()
                    row['week'] = t
                
                if t == int(target):
                    row['event'] = 1
                else:
                    row['event'] = 0
                person_periods.append(row)
    
    return pd.DataFrame(person_periods)

# 新增：Monte Carlo F计算函数
def mc_F_for_interval(model, FEATS, df_group, bmi_values, t_values,
                      sigma_week=0.3,    # 孕周读数标准差（周）
                      se=0.98, sp=0.99,  # 检测灵敏度/特异度（可调/敏感性分析）
                      B=200):            # 组内Z抽样次数
    feat_other = [f for f in FEATS if f not in ("week", "BMI")]
    Z_pool = df_group[feat_other].dropna()
    if Z_pool.empty:
        med = {f: float(np.nanmedian(df_group[f])) for f in feat_other}
        Z_draws = pd.DataFrame([med]*B)
    else:
        Z_draws = Z_pool.sample(n=min(B, len(Z_pool)), replace=len(Z_pool)<B,
                                random_state=42).reset_index(drop=True)
    F_out = np.zeros((len(bmi_values), len(t_values)))
    for bi, bmi in enumerate(bmi_values):
        F_acc = np.zeros(len(t_values))
        for _, z in Z_draws.iterrows():
            grid = pd.DataFrame({"week": t_values, "BMI": bmi})
            for k,v in z.items(): 
                grid[k]=v
            h = model.predict_proba(grid[FEATS].to_numpy(float))
            h = se*h + (1-sp)*(1-h)                 # 判定误差折算
            if sigma_week>0: 
                h = gaussian_filter1d(h, sigma=sigma_week)  # 孕周误差
            S = np.cumprod(1 - h + 1e-12)
            F = 1 - S
            F_acc += F
        F_out[bi,:] = F_acc / len(Z_draws)
    return F_out  # 形状: (len(bmi_values), len(t_values))

# 新增：DP分组求最优时点的三个函数
def interval_risk_and_tstar(F_sub, t_axis, alpha=0.8, c_early=1, c_late=5):
    # 可行集合: F(t) >= alpha
    feas = np.where(F_sub >= alpha)[0]
    if len(feas)==0: return None, np.inf
    # 简化风险：P(T>t)=1-F(t)；P(T<t)≈F(t-1)
    P_late  = 1 - F_sub[feas]
    P_early = np.r_[0, F_sub[feas][:-1]]  # 以 t-1 的F近似
    risks = c_early*P_late + c_late*P_early
    k = feas[np.argmin(risks)]
    return int(t_axis[k]), float(risks.min())

def precompute_R_for_all_intervals(b_bins, F_mat, t_axis,
                                   nmin_by_bin=None, wmin=2.0, alpha=0.8,
                                   c_early=1, c_late=5):
    N=len(b_bins)
    R=[[None]*N for _ in range(N)]
    Tstar=[[None]*N for _ in range(N)]
    feasible=[[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(i,N):
            if (b_bins[j]-b_bins[i]) < wmin: continue
            if nmin_by_bin is not None:
                nsum = nmin_by_bin[i:j+1].sum()
                if nsum < 30: continue
            F_sub = F_mat[i:j+1,:].mean(axis=0)
            tstar, r = interval_risk_and_tstar(F_sub, t_axis, alpha, c_early, c_late)
            if tstar is not None:
                feasible[i][j]=True
                Tstar[i][j]=tstar
                R[i][j]=r
    return feasible, Tstar, R

def dp_optimal_partition(b_bins, feasible, R, Tstar, lam=0.0):
    N=len(b_bins)
    DP=[np.inf]*N
    PRE=[-1]*N
    for j in range(N):
        best=np.inf
        bi=-1
        for i in range(j+1):
            if not feasible[i][j]: continue
            val=(DP[i-1] if i>0 else 0)+R[i][j]+lam
            if val<best: 
                best=val
                bi=i
        DP[j]=best
        PRE[j]=bi
    cuts=[]
    tstars=[]
    j=N-1
    while j>=0:
        i=PRE[j]
        cuts.append((b_bins[i], b_bins[j]))
        tstars.append(Tstar[i][j])
        j=i-1
    cuts.reverse()
    tstars.reverse()
    return cuts, tstars

def main():
    # 1. 数据读取和预处理
    print("读取person-period数据...")
    pp = pd.read_csv(DATA_CSV)
    pp = unify_keys_and_alias(pp)
    
    print(f"Person-period数据形状: {pp.shape}")
    print(f"事件发生数: {pp['event'].sum()}")
    
    # 2. 特征工程 - 扩展多因素特征
    base_feats = ["week", "BMI"]
    
    # Q3新增：扩展多因素特征
    extra_feats = [
        "年龄","log原始读段数","在参考基因组上比对的比例",
        "重复读段的比例","GC含量","被过滤掉读段数的比例","IVF_指示",
        # Q3新增：
        "身高","体重","检测质量主成分1","检测质量主成分2"
    ]
    FEATS = base_feats + [f for f in extra_feats if f in pp.columns]
    print(f"使用的特征: {FEATS}")
    
    # 数据清理
    pp_clean = pp[FEATS + ['event']].dropna()
    print(f"清理后数据形状: {pp_clean.shape}")
    
    # 3. GAM模型训练
    print("训练GAM模型...")
    
    # 构建GAM terms - 修复样条数问题
    if len(FEATS) == 2:  # 只有week和BMI
        # 简化版本，避免复杂的张量项
        terms = s(0, n_splines=8)  # week
        terms += s(1, n_splines=6)  # BMI
    else:
        # 完整版本
        terms = s(0, n_splines=8)  # week
        terms += s(1, n_splines=6)  # BMI
        terms += te(0, 1, n_splines=[5, 4])  # week*BMI交互项，确保都>3
        
        # 添加其他特征
        for i, feat in enumerate(FEATS[2:], start=2):  # 从第3个特征开始
            if pp_clean[feat].nunique() <= 2:  # 二值变量用线性项
                terms += l(i)
            else:  # 连续变量用样条
                terms += s(i, n_splines=5)
    
    gam = LogisticGAM(terms)
    X = pp_clean[FEATS].to_numpy(float)
    y = pp_clean['event'].to_numpy()
    
    gam.fit(X, y)
    print(f"GAM模型训练完成，AIC: {gam.statistics_['AIC']:.2f}")
    
    # 4. 生成F(t|BMI)网格 - 使用新的MC方法
    print("生成F(t|BMI)网格...")
    t_seq = np.arange(11, 27)  # 11-26周
    b_seq = np.arange(18, 46, 0.5)  # BMI 18-45.5，步长0.5
    
    # 🟡 替换：使用mc_F_for_interval替代旧方法
    F_mat = mc_F_for_interval(
        model=gam, FEATS=FEATS, df_group=pp_clean,
        bmi_values=b_seq, t_values=t_seq,
        sigma_week=0.3, se=0.98, sp=0.99, B=200
    )
    
    # 转换为DataFrame格式保持兼容性
    df_grid = (pd.DataFrame(F_mat, index=b_seq, columns=t_seq)
               .stack().rename("F").reset_index()
               .rename(columns={"level_0":"BMI","level_1":"week"}))
    # 保持兼容：补S/h占位（图用 F 即可）
    df_grid = df_grid.sort_values(["BMI","week"])
    df_grid["S"] = 1 - df_grid.groupby("BMI")["F"].transform(lambda x: x)
    df_grid["haz"] = np.nan
    
    print(f"网格数据形状: {df_grid.shape}")
    
    # 5. 邻域合并的支撑度计算
    print("计算邻域支撑度...")
    
    def compute_neighborhood_support(pp_clean, b_seq, t_seq, radius_bmi=1.0, radius_week=1.0):
        """计算邻域合并支撑度"""
        support_mat = np.zeros((len(b_seq), len(t_seq)))
        
        for i, bmi in enumerate(b_seq):
            for j, week in enumerate(t_seq):
                # 定义邻域
                bmi_mask = (pp_clean['BMI'] >= bmi - radius_bmi) & (pp_clean['BMI'] <= bmi + radius_bmi)
                week_mask = (pp_clean['week'] >= week - radius_week) & (pp_clean['week'] <= week + radius_week)
                support_mat[i, j] = (bmi_mask & week_mask).sum()
        
        return support_mat
    
    support_mat = compute_neighborhood_support(pp_clean, b_seq, t_seq, radius_bmi=1.0, radius_week=1.0)
    
    # 6. DP分组求最优时点
    print("执行DP分组优化...")
    
    b_bins = b_seq  # 直接使用BMI网格
    feas, Tstar, R = precompute_R_for_all_intervals(
        b_bins=b_bins, F_mat=F_mat, t_axis=t_seq,
        nmin_by_bin=None, wmin=2.0, alpha=0.8,
        c_early=1, c_late=5
    )
    cuts, tstars = dp_optimal_partition(b_bins, feas, R, Tstar, lam=0.0)
    
    # 导出DP结果
    dp_results = pd.DataFrame({
        "组序": range(1, len(cuts)+1),
        "BMI下界": [a for a,b in cuts],
        "BMI上界": [b for a,b in cuts],
        "最佳时点t*": tstars
    })
    dp_results.to_csv(f"{OUT_DIR}/Q3_DP_分组与最佳时点.csv", index=False, encoding="utf-8-sig")
    print(f"DP结果已保存: {len(cuts)}个分组")
    
    # 7. 可视化
    print("生成图表...")
    
    # 7.1 F热图
    plt.figure(figsize=(12, 8))
    F_pivot = df_grid.pivot(index="BMI", columns="week", values="F")
    im = plt.imshow(F_pivot.values, aspect='auto', origin='lower', cmap='viridis',
                   extent=[t_seq.min(), t_seq.max(), b_seq.min(), b_seq.max()])
    plt.colorbar(im, label='F(t|BMI)')
    plt.xlabel('孕周 (周)')
    plt.ylabel('BMI')
    plt.title('Q3: F(t|BMI) 热图 (误差+多因素MC)')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Q3_F热图.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7.2 风险集热图
    plt.figure(figsize=(12, 8))
    im = plt.imshow(support_mat, aspect='auto', origin='lower', cmap='viridis',
                   extent=[t_seq.min(), t_seq.max(), b_seq.min(), b_seq.max()])
    plt.colorbar(im, label='邻域样本数')
    plt.xlabel('孕周 (周)')
    plt.ylabel('BMI')
    plt.title('Q3: 邻域合并支撑度热图')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Q3_风险集热图.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7.3 掩膜热图
    plt.figure(figsize=(12, 8))
    mask = support_mat >= 30  # 支撑度阈值
    F_masked = F_mat.copy()
    F_masked[~mask] = np.nan
    
    im = plt.imshow(F_masked, aspect='auto', origin='lower', cmap='viridis',
                   extent=[t_seq.min(), t_seq.max(), b_seq.min(), b_seq.max()])
    plt.colorbar(im, label='F(t|BMI)')
    plt.xlabel('孕周 (周)')
    plt.ylabel('BMI')
    plt.title('Q3: F(t|BMI) 热图 (支撑度掩膜后)')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Q3_F热图_masked.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. 导出数据
    print("导出数据...")
    
    # F网格数据
    with pd.ExcelWriter(f"{OUT_DIR}/Q3_F_grid.xlsx", engine='openpyxl') as writer:
        df_grid.to_excel(writer, sheet_name='F_grid', index=False)
        # 添加支撑度数据
        support_df = pd.DataFrame(support_mat, index=b_seq, columns=t_seq)
        support_df.index.name = 'BMI'
        support_df.columns.name = 'week'
        support_df.reset_index().to_excel(writer, sheet_name='support', index=False)
    
    print("Q3分析完成!")
    print(f"输出文件:")
    print(f"- {OUT_DIR}/Q3_F_grid.xlsx")
    print(f"- {OUT_DIR}/Q3_F热图.png")
    print(f"- {OUT_DIR}/Q3_风险集热图.png")
    print(f"- {OUT_DIR}/Q3_F热图_masked.png")
    print(f"- {OUT_DIR}/Q3_DP_分组与最佳时点.csv")

if __name__ == "__main__":
    main()