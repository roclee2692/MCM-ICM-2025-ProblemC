# 2025 全国大学生数学建模竞赛 C 题 / 2025 CUMCM Problem C

> 本项目参加的是中国全国大学生数学建模竞赛（CUMCM），不是美国 MCM/ICM。仓库名称沿用历史地址。
>
> [参赛论文 PDF](Competition_Paper.pdf) · [Q1](Q1支撑材料/) · [Q2](Q2支撑材料/) · [Q3](Q3支撑材料/) · [Q4](Q4支撑材料/)

<details>
<summary><b>🇨🇳 点击查看中文版 / Click for Chinese Version</b></summary>

## 项目简介

本仓库包含三人团队参加 2025 年全国大学生数学建模竞赛 C 题的论文、Q1–Q4 代码和相关数据，获得河南赛区省级一等奖。

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
├── AI 工具使用详情.docx  # AI工具使用说明
├── Competition_Paper.pdf         # 最终提交论文
└── README.md
```

## 🚀 快速开始

### 方法一：运行 Q1 的 Windows 脚本

`Q1支撑材料/run_all.bat` 位于 Q1 目录，只运行 Q1 的 Python 清洗和 R 建模；它会创建虚拟环境并安装依赖，需要预先安装 Python 和 R。它不是 Q1–Q4 的一键运行脚本，本次未重新执行验证。

### 方法二：手动运行

#### Q1 - R语言程序运行

**环境要求：**
- R 4.x
- RStudio（推荐）
- 所需R包：mgcv, gratia, rsample, writexl

**运行步骤：**

1. 将 `q1_gamm_all.R` 和 `清洗结果_Q1.xlsx` 放到同一目录（如 `Q1支撑材料/`）
2. 在RStudio中打开 `q1_gamm_all.R`
3. 设置工作目录：Session → Set Working Directory → To Source File Location
4. 运行脚本：点击 **Source** 按钮或按 `Ctrl + Shift + Enter`
5. 查看输出文件

#### Q2-Q4 - Python程序运行

**虚拟环境配置（以Q4为例）：**

```powershell
# 1. 进入项目目录
cd .\Q4支撑材料

# 2. 创建虚拟环境
python -m venv .venv

# 3. 临时放开脚本执行限制
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 5. 安装依赖
python -m pip install -r .\requirements_q4.txt

# 6. 运行代码
python q4_python_all.py
```

## 📊 问题概述

### 问题1：胎儿生长曲线建模
- **支撑材料位置**: `Q1支撑材料/`
- **主要方法**: GAMM（广义可加混合模型）
- **核心代码**: `q1_clean.py`, `q1_gamm_all.R`

### 问题2：生存分析
- **支撑材料位置**: `Q2支撑材料/`
- **主要方法**: 生存分析、BMI分组分析
- **核心代码**: `q2_python_all.py`

### 问题3：蒙特卡洛与动态规划
- **支撑材料位置**: `Q3支撑材料/`
- **核心代码**: `q3_mc_dp.py`

### 问题4：机器学习预测
- **支撑材料位置**: `Q4支撑材料/`
- **核心代码**: `q4_python_all.py`

## 📈 主要结果

各问题的结果图表分别保存在对应的输出文件夹中：
- Q1: `Q1支撑材料/输出图/`
- Q2: `Q2支撑材料/输出图/`
- Q3: `Q3支撑材料/output/`
- Q4: `Q4支撑材料/Q4_输出/`

完整论文请查看：[Competition_Paper.pdf](Competition_Paper.pdf)

## 👥 团队成员

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

</details>

---

## 🇺🇸 English Version

### Project Overview

This repository contains our team's paper, Q1–Q4 code and related data for Problem C of the 2025 China Undergraduate Mathematical Contest in Modeling (CUMCM). The team received a provincial first prize in Henan.

**Team Information:**
- Team Number: 202516026047
- Institution: North China University of Water Conservancy and Electric Power
- Team Members: See collaborators list

**Competition Information:**
- Year: 2025
- Problem: C
- Title: NIPT Dynamic Decision Optimization: Survival Analysis and Risk Constraints

### 📂 Project Structure

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
├── AI 工具使用详情.docx  # AI Tool Usage Details
├── Competition_Paper.pdf         # Final Submission Paper
└── README.md
```

### 🚀 Quick Start

#### Method 1: Run the Q1 Windows script

`Q1支撑材料/run_all.bat` runs Q1 preprocessing and R modelling only; it creates a virtual environment and installs dependencies. Python and R are required. It does not run all four questions and was not rerun during this documentation review.

#### Method 2: Manual Run

**Q1 - R Language Program:**

Requirements: R 4.x, RStudio, Required packages: mgcv, gratia, rsample, writexl

Steps:
1. Place `q1_gamm_all.R` and `清洗结果_Q1.xlsx` in same directory
2. Open `q1_gamm_all.R` in RStudio
3. Set working directory: Session → Set Working Directory → To Source File Location
4. Run script: Click **Source** button or press `Ctrl + Shift + Enter`
5. Check output files

**Q2-Q4 - Python Programs:**

Virtual environment setup (using Q4 as example):

```powershell
# 1. Navigate to project directory
cd .\Q4支撑材料

# 2. Create virtual environment
python -m venv .venv

# 3. Allow script execution
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 5. Install dependencies
python -m pip install -r .\requirements_q4.txt

# 6. Run code
python q4_python_all.py
```

### 📊 Problem Overview

- **Problem 1**: Fetal Growth Curve Modeling (GAMM)
- **Problem 2**: Survival Analysis (BMI grouping)
- **Problem 3**: Monte Carlo & Dynamic Programming
- **Problem 4**: Machine Learning Prediction

### 📈 Main Results

Result charts are saved in corresponding output folders:
- Q1: `Q1支撑材料/输出图/`
- Q2: `Q2支撑材料/输出图/`
- Q3: `Q3支撑材料/output/`
- Q4: `Q4支撑材料/Q4_输出/`

Complete paper: [Competition_Paper.pdf](Competition_Paper.pdf)

### 👥 Team Members

**Member 1:** Data cleaning, code implementation, visualization, paper review, abstract writing, AI report writing

**Member 2:** Statistical modeling, analysis, paper writing, abstract optimization, AI report polishing

**Member 3:** Paper writing, layout, AI report layout

---

**Note**: This repository is for learning and communication purposes only. Please comply with competition rules and academic integrity principles.

**Last Updated**: September 2025
