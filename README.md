# 2025年数学建模竞赛C题解决方案

## 项目简介

本仓库包含我们团队参加2025年数学建模竞赛C题的完整解决方案，包括论文、代码实现和相关数据。

**队伍信息：**
- 队伍编号：202516026047
- 参赛院校：North China University of Water Conservancy and Electric Power
- 队伍成员：见协作者列表

**竞赛信息：**
- 竞赛年份：2025年
- 题目编号：C题
- 题目名称：NIPT 动态决策优化：生存分析与风险约束

## 📂 项目结构

```
├── Q1支撑材料/          # 问题1相关代码和数据
│   ├── 输出图/          # Q1可视化结果
│   ├── q1_clean.py     # 数据清洗
│   ├── q1_gamm_all.R   # GAMM统计模型
│   └── ...
├── Q2支撑材料/          # 问题2相关代码和数据
│   ├── 输出图/          # Q2可视化结果
│   ├── q2_python_all.py # 完整分析代码
│   └── ...
├── Q3支撑材料/          # 问题3相关代码和数据
│   ├── output/         # Q3输出结果
│   ├── q3_mc_dp.py     # 主程序
│   └── ...
├── Q4支撑材料/          # 问题4相关代码和数据
│   ├── Q4_输出/        # Q4输出结果
│   ├── q4_python_all.py # 主程序
│   └── ...
├── AI工具使用详情.docx  # AI工具使用说明
├── 参赛论文.pdf         # 最终提交论文
└── README.md

```

## 快速开始

### 方法一：一键运行（推荐）

如果已在Windows系统中配置R语言编译器，并将其所在路径正确添加到系统环境变量：

```bash
# 直接双击项目根目录下的批处理文件
run_all.bat
```

### 方法二：手动运行

#### Q1 - R语言程序运行

**环境要求：**
- R 4.x
- RStudio（推荐）
- 所需R包：mgcv, gratia, rsample, writexl

**运行步骤：**

1. 将 `q1_gamm_all.R` 和 `清洗结果_Q1.xlsx` 放到同一目录（如 `Q1支撑材料/`）

2. 在RStudio中打开 `q1_gamm_all.R`

3. 设置工作目录：
   - 方法：Session → Set Working Directory → To Source File Location
   - 此步骤很关键，确保读表时能找到文件

4. 运行脚本：
   - 点击脚本编辑器右上角的 **Source** 按钮
   - 或使用快捷键 `Ctrl + Shift + Enter`
   - 脚本会自动安装缺失的包，然后依次执行

5. 查看输出：
   - 控制台提示：`[Q1] 拟合 Beta-GAMM ...`、`[Q1] 已导出：Q1_分组CV_指标.xlsx` 等
   - 输出文件：
     - `Q1_模型摘要.txt`
     - `输出图/Q1_部分效应.png`
     - `Q1_关键点_预测.xlsx`
     - `Q2_等高线底图.xlsx`
     - `输出图/Q2_达标概率_等高线.png`
     - `Q1_分组CV_指标.xlsx`

#### Q2-Q4 - Python程序运行

**Python脚本说明：**
- Q1的Python脚本（`q1_clean.py`）仅用于数据清洗，非核心代码
- Q2-Q4为核心分析代码
- 推荐使用虚拟环境运行

**虚拟环境配置（以Q4为例）：**

在PowerShell中执行以下命令：

```powershell
# 1. 进入项目目录
cd D:\DpanPython\C-Q4

# 2. 创建虚拟环境
python -m venv .venv

# 3. 临时放开脚本执行限制（避免"running scripts is disabled"报错）
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. 激活虚拟环境（PowerShell使用.ps1）
.\.venv\Scripts\Activate.ps1

# 5. 更新pip（可选）
python -m pip install --upgrade "pip==25.1.1"

# 6. 安装依赖
python -m pip install -r .\requirements_q4.txt

# 7. 运行代码
python q4_python_all.py
```

**检查命令：**
- 查看Python版本：`python -V` 或 `py -V`
- 查看执行策略：`Get-ExecutionPolicy`

**其他问题运行方法类似：**
- Q1: `python q1_clean.py`
- Q2: `python q2_python_all.py`
- Q3: `python q3_mc_dp.py`
- Q4: `python q4_python_all.py`

## 📊 问题概述

### 问题1：胎儿生长曲线建模
- **支撑材料位置**: `Q1支撑材料/`
- **主要方法**: GAMM（广义可加混合模型）
- **核心代码**: `q1_clean.py`, `q1_gamm_all.R`

### 问题2：生存分析
- **支撑材料位置**: `Q2支撑材料/`
- **主要方法**: 生存分析、BMI分组分析
- **核心代码**: `q2_python_all.py`

### 问题3：[问题描述]
- **支撑材料位置**: `Q3支撑材料/`
- **核心代码**: `q3_mc_dp.py`

### 问题4：[问题描述]
- **支撑材料位置**: `Q4支撑材料/`
- **核心代码**: `q4_python_all.py`

## 📈 主要结果

各问题的结果图表分别保存在对应的输出文件夹中：
- Q1: `Q1支撑材料/输出图/`
- Q2: `Q2支撑材料/输出图/`
- Q3: `Q3支撑材料/output/`
- Q4: `Q4支撑材料/Q4_输出/`

完整论文请查看：[参赛论文.pdf](参赛论文.pdf)

## 依赖环境

### Q1依赖

**Python部分（数据清洗）：**
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
openpyxl==3.1.5
pyarrow==15.0.2
fastparquet==2024.11.0
matplotlib==3.8.4
```

**R部分（核心建模）：**
```r
mgcv      # GAMM模型
gratia    # GAMM可视化
rsample   # 交叉验证
writexl   # 导出Excel
```

### Q2依赖

```
# Python 3.11.x
numpy==1.26.4
pandas==2.3.2
scipy==1.11.4
matplotlib==3.10.6
scikit-learn==1.7.1
openpyxl==3.1.5
xlsxwriter==3.2.5

# Q2核心：pyGAM + 进度条（Windows友好）
pygam==0.9.1
progressbar2==4.5.0
python-utils==3.9.1

# 画图/读写常用
pillow==11.3.0
tzdata==2025.2
```

### Q3依赖

```
# Python 3.11.x
numpy==1.26.4
pandas==2.3.2
scipy==1.11.4
matplotlib==3.10.6
scikit-learn==1.7.1
openpyxl==3.1.5
xlsxwriter==3.2.5

# Q3核心：pyGAM + 进度条（Windows友好）
pygam==0.9.1
progressbar2==4.5.0
python-utils==3.9.1

# 画图/读写常用
pillow==11.3.0
tzdata==2025.2
```

### Q4依赖

```
# Python 3.11.x
numpy==1.26.4
pandas==2.3.2
scikit-learn==1.7.1
xgboost==2.1.1
shap==0.45.1
matplotlib==3.10.0
openpyxl==3.1.5
xlsxwriter==3.2.5
lightgbm==4.5.0
```

## 团队成员

本项目由3人团队协作完成，分工如下：

**成员1：**
- 数据清洗与预处理
- 代码实现与可视化
- 审查论文提出修改建议
- 摘要撰写
- AI使用报告撰写

**成员2：**
- 统计建模与分析
- 此部分论文撰写
- 审查论文提出修改建议
- 摘要优化
- AI使用报告润色

**成员3：**
- 论文绝大部分撰写与图文排版
- AI使用报告排版

---

**注意**：本仓库仅供学习交流使用，请遵守竞赛规则和学术诚信原则。

**最后更新**：2025年9月
