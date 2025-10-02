# -*- coding: utf-8 -*-
"""
Q1 数据清洗 · 稳定版
修复点：
- 将写 Parquet 放在最后；写入前把所有 object 列改为 pandas 的 'string' 类型
- 就算 Parquet 失败，也不影响 Excel/CSV/描述统计产出（try/except）
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_XLSX = "附件.xlsx"
SHEET_MALE = "男胎检测数据"
SHEET_FEMALE = "女胎检测数据"

def standardize_columns(df: pd.DataFrame, is_male=True) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if not is_male:
        if 'Unnamed: 20' in df.columns and 'Y染色体的Z值' not in df.columns:
            df = df.rename(columns={'Unnamed: 20': 'Y染色体的Z值'})
        if 'Unnamed: 21' in df.columns and 'Y染色体浓度' not in df.columns:
            df = df.rename(columns={'Unnamed: 21': 'Y染色体浓度'})
    return df

def parse_gest_weeks(series: pd.Series) -> pd.Series:
    def parse_one(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if re.fullmatch(r'\d+(\.\d+)?', s):
            return float(s)
        m = re.search(r'(\d+)\s*(w|周)?\s*\+?\s*(\d+)?\s*(d|天)?', s)
        if m:
            w = float(m.group(1))
            d = m.group(3)
            d = float(d) if d is not None else 0.0
            return w + d/7.0
        m2 = re.search(r'(\d+(\.\d+)?)', s)
        return float(m2.group(1)) if m2 else np.nan
    return series.apply(parse_one)

def smithson_verkuilen_transform(v: pd.Series, n: int = None) -> pd.Series:
    v = pd.to_numeric(v, errors='coerce')
    if n is None:
        n = v.notna().sum()
    return (v*(n-1)+0.5)/n

def compute_qc_pca(df: pd.DataFrame) -> pd.DataFrame:
    qc_cols = [
        '原始读段数','在参考基因组上比对的比例','重复读段的比例','唯一比对的读段数',
        'GC含量','13号染色体的GC含量','18号染色体的GC含量','21号染色体的GC含量',
        '被过滤掉读段数的比例'
    ]
    have = [c for c in qc_cols if c in df.columns]
    if len(have) == 0:
        df['检测质量主成分1'] = np.nan
        df['检测质量主成分2'] = np.nan
        return df
    X = df[have].apply(pd.to_numeric, errors='coerce').values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Xs)
    df['检测质量主成分1'] = pcs[:, 0]
    df['检测质量主成分2'] = pcs[:, 1]
    exp = pca.explained_variance_ratio_
    print(f"[QC-PCA] 前两主成分累计解释率：{exp.sum():.3f}（PC1={exp[0]:.3f}, PC2={exp[1]:.3f}）")
    return df

def clean_sheet(df: pd.DataFrame, is_male=True) -> pd.DataFrame:
    df = standardize_columns(df, is_male=is_male)
    df = df.dropna(how="all").copy()

    if '检测孕周' in df.columns:
        df['孕周_周'] = parse_gest_weeks(df['检测孕周'])
    else:
        df['孕周_周'] = np.nan

    bmi_col = next((c for c in df.columns if 'BMI' in c), None)
    df['体质指数'] = pd.to_numeric(df[bmi_col], errors='coerce') if bmi_col else np.nan

    id_col = '孕妇代码' if '孕妇代码' in df.columns else ('孕妇编号' if '孕妇编号' in df.columns else None)
    df['孕妇ID'] = df[id_col].astype(str) if id_col else df.index.astype(str)

    for date_col in ['末次月经', '检测日期']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    numeric_cols = [
        '年龄','检测抽血次数','原始读段数','在参考基因组上比对的比例','重复读段的比例','唯一比对的读段数',
        'GC含量','被过滤掉读段数的比例','13号染色体的GC含量','18号染色体的GC含量','21号染色体的GC含量',
        'X染色体的Z值','13号染色体的Z值','18号染色体的Z值','21号染色体的Z值','Y染色体的Z值','Y染色体浓度'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if is_male and 'Y染色体浓度' in df.columns:
        df['Y浓度'] = pd.to_numeric(df['Y染色体浓度'], errors='coerce')
        df['Y浓度_Beta调整'] = smithson_verkuilen_transform(df['Y浓度'])

    df = compute_qc_pca(df)

    df['边缘样本'] = False
    df.loc[(df['孕周_周'] < 9) | (df['孕周_周'] > 26), '边缘样本'] = True
    df.loc[(df['体质指数'] < 13) | (df['体质指数'] > 55), '边缘样本'] = True

    return df

def describe_table(df: pd.DataFrame, is_male=True) -> pd.DataFrame:
    cols = ['孕妇ID','孕周_周','体质指数','年龄','IVF妊娠','检测抽血次数',
            '原始读段数','在参考基因组上比对的比例','重复读段的比例','唯一比对的读段数',
            'GC含量','被过滤掉读段数的比例','检测质量主成分1','检测质量主成分2']
    if is_male:
        cols += ['Y浓度','Y浓度_Beta调整','Y染色体的Z值']
    use = [c for c in cols if c in df.columns]
    desc = df[use].describe(include='all').T
    miss = df[use].isna().mean().rename("缺失率")
    return desc.join(miss)

def safe_to_parquet(df: pd.DataFrame, path: str):
    """将所有 object 列转为 pandas 'string'，避免 pyarrow 转型失败；失败则跳过"""
    try:
        df2 = df.copy()
        obj_cols = df2.select_dtypes(include=['object']).columns
        for c in obj_cols:
            df2[c] = df2[c].astype('string')  # 保留缺失为 <NA>
        df2.to_parquet(path, index=False)    # 需要 pyarrow 或 fastparquet
        print(f"[OK] 已写入 Parquet：{path}")
    except Exception as e:
        print(f"[跳过 Parquet] {path} 写入失败：{e}")

def main():
    xlsx = pd.ExcelFile(INPUT_XLSX)
    male_raw = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_MALE)
    female_raw = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_FEMALE)

    male_clean = clean_sheet(male_raw, is_male=True)
    female_clean = clean_sheet(female_raw, is_male=False)

    # 先确保 CSV 与 Excel 必定产出
    male_clean.to_csv("男胎_清洗版_Q1.csv", index=False, encoding="utf-8-sig")
    female_clean.to_csv("女胎_清洗版_Q1.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter("清洗结果_Q1.xlsx", engine="openpyxl") as writer:
        male_clean.to_excel(writer, sheet_name="男胎_清洗版", index=False)
        female_clean.to_excel(writer, sheet_name="女胎_清洗版", index=False)

    male_desc = describe_table(male_clean, is_male=True)
    female_desc = describe_table(female_clean, is_male=False)
    with pd.ExcelWriter("清洗_描述统计_Q1.xlsx", engine="openpyxl") as writer:
        male_desc.to_excel(writer, sheet_name="男胎_描述", index=True)
        female_desc.to_excel(writer, sheet_name="女胎_描述", index=True)

    # 再尝试写 Parquet（可选，失败不影响主流程）
    safe_to_parquet(male_clean, "男胎_清洗版_Q1.parquet")
    safe_to_parquet(female_clean, "女胎_清洗版_Q1.parquet")

    print("=== 完成 ===")
    print("男胎记录数：", male_clean.shape, "；女胎记录数：", female_clean.shape)
    print("男胎：孕周缺失率 = {:.1%}，体质指数缺失率 = {:.1%}".format(
        male_clean['孕周_周'].isna().mean(), male_clean['体质指数'].isna().mean()))
    if 'Y浓度' in male_clean.columns:
        print("男胎：Y浓度缺失率 = {:.1%}".format(male_clean['Y浓度'].isna().mean()))

if __name__ == "__main__":
    main()
