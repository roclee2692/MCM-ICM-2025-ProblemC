# 2025 Mathematical Contest in Modeling - Problem C

[简体中文](README_CN.md) | **English**

## Project Overview

This repository contains our team's complete solution for Problem C of the 2025 Mathematical Contest in Modeling, including paper, code implementation, and related data.

**Team Information:**
- Team Number: 202516026047
- Institution: North China University of Water Conservancy and Electric Power
- Team Members: See collaborators list

**Competition Information:**
- Year: 2025
- Problem: C
- Title: NIPT Dynamic Decision Optimization: Survival Analysis and Risk Constraints

## Project Structure

```
├── Q1支撑材料/          # Problem 1 Code and Data
│   ├── 输出图/          # Q1 Visualization Results
│   ├── q1_clean.py     # Data Cleaning
│   ├── q1_gamm_all.R   # GAMM Statistical Model
│   └── ...
├── Q2支撑材料/          # Problem 2 Code and Data
│   ├── 输出图/          # Q2 Visualization Results
│   ├── q2_python_all.py # Complete Analysis Code
│   └── ...
├── Q3支撑材料/          # Problem 3 Code and Data
│   ├── output/         # Q3 Output Results
│   ├── q3_mc_dp.py     # Main Program
│   └── ...
├── Q4支撑材料/          # Problem 4 Code and Data
│   ├── Q4_输出/        # Q4 Output Results
│   ├── q4_python_all.py # Main Program
│   └── ...
├── AI工具使用详情.docx  # AI Tool Usage Details
├── 参赛论文.pdf         # Final Submission Paper
└── README.md

```

## Quick Start

### Method 1: One-Click Run (Recommended)

If you have configured the R language compiler on Windows and added it to the system environment variables:

```bash
# Double-click the batch file in the project root directory
run_all.bat
```

### Method 2: Manual Run

#### Q1 - R Language Program

**Requirements:**
- R 4.x
- RStudio (Recommended)
- Required R packages: mgcv, gratia, rsample, writexl

**Steps:**

1. Place `q1_gamm_all.R` and `清洗结果_Q1.xlsx` in the same directory (e.g., `Q1支撑材料/`)

2. Open `q1_gamm_all.R` in RStudio

3. Set working directory:
   - Method: Session → Set Working Directory → To Source File Location
   - This step is crucial for the script to locate files

4. Run the script:
   - Click the **Source** button in the upper right corner of the script editor
   - Or use shortcut `Ctrl + Shift + Enter`
   - The script will automatically install missing packages and execute sequentially

5. Check output:
   - Console prompts: `[Q1] Fitting Beta-GAMM ...`, `[Q1] Exported: Q1_分组CV_指标.xlsx`, etc.
   - Output files:
     - `Q1_模型摘要.txt`
     - `输出图/Q1_部分效应.png`
     - `Q1_关键点_预测.xlsx`
     - `Q2_等高线底图.xlsx`
     - `输出图/Q2_达标概率_等高线.png`
     - `Q1_分组CV_指标.xlsx`

#### Q2-Q4 - Python Programs

**Python Script Notes:**
- Q1's Python script (`q1_clean.py`) is only for data cleaning, not core code
- Q2-Q4 are core analysis codes
- Virtual environment is recommended

**Virtual Environment Setup (Using Q4 as example):**

Execute the following commands in PowerShell:

```powershell
# 1. Navigate to project directory
cd D:\DpanPython\C-Q4

# 2. Create virtual environment
python -m venv .venv

# 3. Temporarily allow script execution (avoid "running scripts is disabled" error)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Activate virtual environment (PowerShell uses .ps1)
.\.venv\Scripts\Activate.ps1

# 5. Update pip (optional)
python -m pip install --upgrade "pip==25.1.1"

# 6. Install dependencies
python -m pip install -r .\requirements_q4.txt

# 7. Run code
python q4_python_all.py
```

**Check commands:**
- Check Python version: `python -V` or `py -V`
- Check execution policy: `Get-ExecutionPolicy`

**Run other problems similarly:**
- Q1: `python q1_clean.py`
- Q2: `python q2_python_all.py`
- Q3: `python q3_mc_dp.py`
- Q4: `python q4_python_all.py`

## Problem Overview

### Problem 1: Fetal Growth Curve Modeling
- **Materials location**: `Q1支撑材料/`
- **Main method**: GAMM (Generalized Additive Mixed Models)
- **Core code**: `q1_clean.py`, `q1_gamm_all.R`

### Problem 2: Survival Analysis
- **Materials location**: `Q2支撑材料/`
- **Main method**: Survival analysis, BMI grouping analysis
- **Core code**: `q2_python_all.py`

### Problem 3: [Problem Description]
- **Materials location**: `Q3支撑材料/`
- **Core code**: `q3_mc_dp.py`

### Problem 4: [Problem Description]
- **Materials location**: `Q4支撑材料/`
- **Core code**: `q4_python_all.py`

## Main Results

Result charts for each problem are saved in corresponding output folders:
- Q1: `Q1支撑材料/输出图/`
- Q2: `Q2支撑材料/输出图/`
- Q3: `Q3支撑材料/output/`
- Q4: `Q4支撑材料/Q4_输出/`

Complete paper: [参赛论文.pdf](参赛论文.pdf)

## Dependencies

### Q1 Dependencies

**Python Part (Data Cleaning):**
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
openpyxl==3.1.5
pyarrow==15.0.2
fastparquet==2024.11.0
matplotlib==3.8.4
```

**R Part (Core Modeling):**
```r
mgcv      # GAMM models
gratia    # GAMM visualization
rsample   # Cross-validation
writexl   # Export Excel
```

### Q2 Dependencies

```
# Python 3.11.x
numpy==1.26.4
pandas==2.3.2
scipy==1.11.4
matplotlib==3.10.6
scikit-learn==1.7.1
openpyxl==3.1.5
xlsxwriter==3.2.5

# Q2 Core: pyGAM + progress bar (Windows-friendly)
pygam==0.9.1
progressbar2==4.5.0
python-utils==3.9.1

# Plotting/IO utilities
pillow==11.3.0
tzdata==2025.2
```

### Q3 Dependencies

```
# Python 3.11.x
numpy==1.26.4
pandas==2.3.2
scipy==1.11.4
matplotlib==3.10.6
scikit-learn==1.7.1
openpyxl==3.1.5
xlsxwriter==3.2.5

# Q3 Core: pyGAM + progress bar (Windows-friendly)
pygam==0.9.1
progressbar2==4.5.0
python-utils==3.9.1

# Plotting/IO utilities
pillow==11.3.0
tzdata==2025.2
```

### Q4 Dependencies

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

## Team Members

This project was completed by a 3-person team with the following division of labor:

**Member 1:**
- Data cleaning and preprocessing
- Code implementation and visualization
- Paper review and suggestions
- Abstract writing
- AI usage report writing

**Member 2:**
- Statistical modeling and analysis
- Paper writing for this section
- Paper review and suggestions
- Abstract optimization
- AI usage report polishing

**Member 3:**
- Majority of paper writing and layout
- AI usage report layout

---

**Note**: This repository is for learning and communication purposes only. Please comply with competition rules and academic integrity principles.

**Last Updated**: September 2025
2025年数学建模竞赛C题解决方案
简体中文 | English

项目简介
本仓库包含我们团队参加2025年数学建模竞赛C题的完整解决方案，包括论文、代码实现和相关数据。

队伍信息：

队伍编号：202516026047
参赛院校：North China University of Water Conservancy and Electric Power
队伍成员：见协作者列表
竞赛信息：

竞赛年份：2025年
题目编号：C题
题目名称：NIPT 动态决策优化：生存分析与风险约束
📂 项目结构
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
快速开始
方法一：一键运行（推荐）
如果已在Windows系统中配置R语言编译器，并将其所在路径正确添加到系统环境变量：

bash
# 直接双击项目根目录下的批处理文件
run_all.bat
方法二：手动运行
Q1 - R语言程序运行
环境要求：

R 4.x
RStudio（推荐）
所需R包：mgcv, gratia, rsample, writexl
运行步骤：

将 q1_gamm_all.R 和 清洗结果_Q1.xlsx 放到同一目录（如 Q1支撑材料/）
在RStudio中打开 q1_gamm_all.R
设置工作目录：
方法：Session → Set Working Directory → To Source File Location
此步骤很关键，确保读表时能找到文件
运行脚本：
点击脚本编辑器右上角的 Source 按钮
或使用快捷键 Ctrl + Shift + Enter
脚本会自动安装缺失的包，然后依次执行
查看输出：
控制台提示：[Q1] 拟合 Beta-GAMM ...、[Q1] 已导出：Q1_分组CV_指标.xlsx 等
输出文件：
Q1_模型摘要.txt
输出图/Q1_部分效应.png
Q1_关键点_预测.xlsx
Q2_等高线底图.xlsx
输出图/Q2_达标概率_等高线.png
Q1_分组CV_指标.xlsx
Q2-Q4 - Python程序运行
Python脚本说明：

Q1的Python脚本（q1_clean.py）仅用于数据清洗，非核心代码
Q2-Q4为核心分析代码
推荐使用虚拟环境运行
虚拟环境配置（以Q4为例）：

在PowerShell中执行以下命令：

powershell
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
检查命令：

查看Python版本：python -V 或 py -V
查看执行策略：Get-ExecutionPolicy
其他问题运行方法类似：

Q1: python q1_clean.py
Q2: python q2_python_all.py
Q3: python q3_mc_dp.py
Q4: python q4_python_all.py
📊 问题概述
问题1：胎儿生长曲线建模
支撑材料位置: Q1支撑材料/
主要方法: GAMM（广义可加混合模型）
核心代码: q1_clean.py, q1_gamm_all.R
问题2：生存分析
支撑材料位置: Q2支撑材料/
主要方法: 生存分析、BMI分组分析
核心代码: q2_python_all.py
问题3：[问题描述]
支撑材料位置: Q3支撑材料/
核心代码: q3_mc_dp.py
问题4：[问题描述]
支撑材料位置: Q4支撑材料/
核心代码: q4_python_all.py
📈 主要结果
各问题的结果图表分别保存在对应的输出文件夹中：

Q1: Q1支撑材料/输出图/
Q2: Q2支撑材料/输出图/
Q3: Q3支撑材料/output/
Q4: Q4支撑材料/Q4_输出/
完整论文请查看：参赛论文.pdf

依赖环境
Q1依赖
Python部分（数据清洗）：

pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
openpyxl==3.1.5
pyarrow==15.0.2
fastparquet==2024.11.0
matplotlib==3.8.4
R部分（核心建模）：

r
mgcv      # GAMM模型
gratia    # GAMM可视化
rsample   # 交叉验证
writexl   # 导出Excel
Q2依赖
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
Q3依赖
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
Q4依赖
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
团队成员
本项目由3人团队协作完成，分工如下：

成员1：

数据清洗与预处理
代码实现与可视化
审查论文提出修改建议
摘要撰写
AI使用报告撰写
成员2：

统计建模与分析
此部分论文撰写
审查论文提出修改建议
摘要优化
AI使用报告润色
成员3：

论文绝大部分撰写与图文排版
AI使用报告排版
注意：本仓库仅供学习交流使用，请遵守竞赛规则和学术诚信原则。

最后更新：2025年9月

