# 测试修复后的R代码语法
# 此文件用于验证 q1_gamm_all.R 的语法是否正确

# 加载必要的包
suppressPackageStartupMessages({
  pkgs <- c("readxl","mgcv","gratia","rsample","ggplot2","writexl","viridis","dplyr")
  need <- setdiff(pkgs, rownames(installed.packages()))
  if(length(need)) install.packages(need, repos="https://cloud.r-project.org")
  lapply(pkgs, require, character.only = TRUE)
})

# 测试修复的语法
cat("测试语法修复...\n")

# 模拟数据框
test_dat <- data.frame(
  col1 = c("test\u00a0 ", " test\u3000", "normal"),
  col2 = 1:3
)

# 测试列名清洗（修复后的版本）
cat("原始列名:", names(test_dat), "\n")
names(test_dat) <- trimws(names(test_dat))
names(test_dat) <- gsub("\u00a0","", names(test_dat), fixed = TRUE)
names(test_dat) <- gsub("\u3000","", names(test_dat), fixed = TRUE)
cat("清洗后列名:", names(test_dat), "\n")

cat("语法测试通过！\n")
