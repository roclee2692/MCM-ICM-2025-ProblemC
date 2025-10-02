@echo off
REM === 自动运行 C-Q1 全流程 ===
set ROOT=%~dp0
cd /d "%ROOT%"

echo [1/4] 创建并激活虚拟环境...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/4] 安装依赖...
pip install -r requirements.txt

echo [3/4] 运行 Python 清洗脚本...
python q1_clean.py

echo [4/4] 调用 R 脚本建模...
Rscript q1_gamm_all.R

echo === 全流程结束 ===
pause
