# ==== q1_gamm_all.R：Q1建模 + 图 + 等高线 + 分组CV（多因子可选） ====
suppressPackageStartupMessages({
  pkgs <- c("readxl","mgcv","gratia","rsample","ggplot2","writexl","viridis","dplyr")
  need <- setdiff(pkgs, rownames(installed.packages()))
  if(length(need)) install.packages(need, repos="https://cloud.r-project.org")
  lapply(pkgs, require, character.only = TRUE)
})

dir.create("输出图", showWarnings = FALSE)

# ---------- 1) 读取清洗数据 ----------
if (!file.exists("清洗结果_Q1.xlsx")) {
  stop("未找到 清洗结果_Q1.xlsx。请先运行 q1_clean.py 或把清洗结果放到当前目录。")
}
dat <- readxl::read_excel("清洗结果_Q1.xlsx", sheet = "男胎_清洗版")

# 列名去不可见空格
names(dat) <- trimws(names(dat))
names(dat) <- gsub("\u00a0","", names(dat), fixed = TRUE)
names(dat) <- gsub("\u3000","", names(dat), fixed = TRUE)

# 主键列兼容（孕妇代码 -> 孕妇ID）
if(!"孕妇ID" %in% names(dat) && "孕妇代码" %in% names(dat)){
  dat <- dplyr::rename(dat, 孕妇ID = 孕妇代码)
}

# 变量类型
dat$孕妇ID   <- factor(dat$孕妇ID)
dat$孕周_周  <- as.numeric(dat$孕周_周)
dat$体质指数 <- as.numeric(dat$体质指数)

# 响应列
ycol <- "Y浓度_Beta调整"
if(!ycol %in% names(dat)) stop("缺失列：Y浓度_Beta调整（请确认Python清洗已生成）。")
dat[[ycol]] <- as.numeric(dat[[ycol]])
dat <- dplyr::filter(dat, is.finite(dat[[ycol]]), dat[[ycol]] > 0, dat[[ycol]] < 1)

# ---------- 1a) 多因子（自动可选） ----------
# 可选协变量：有则纳入，无则忽略
qc_candidates <- c(
  "年龄","IVF妊娠","原始读段数","在参考基因组上比对的比例",
  "重复读段的比例","GC含量","被过滤掉读段数的比例",
  "检测质量主成分1","检测质量主成分2"
)
have_qc <- qc_candidates[qc_candidates %in% names(dat)]

# IVF妊娠 -> 因子（IVF vs 自然受孕）
if("IVF妊娠" %in% have_qc){
  dat$IVF妊娠 <- factor(ifelse(as.character(dat$IVF妊娠) %in% c("IVF","试管婴儿","体外受精","是","Yes","TRUE"), "IVF","自然受孕"))
  
  # 检查是否有足够的变异性（至少两个水平）
  if(length(levels(dat$IVF妊娠)) < 2 || min(table(dat$IVF妊娠)) < 2){
    message("[警告] IVF妊娠变量缺乏变异性，从模型中排除")
    have_qc <- setdiff(have_qc, "IVF妊娠")
  }
}

# 原始读段数对数化
if("原始读段数" %in% have_qc){
  dat$log原始读段数 <- log1p(as.numeric(dat$原始读段数))
  have_qc <- unique(c(have_qc, "log原始读段数"))
}

# ---------- 2) 拟合 Beta(logit)-GAMM（加入交互 + 可收缩平滑 + 随机效应） ----------
message("[Q1] 拟合 Beta-GAMM ...")

# 基础平滑 + 二元交互 + 随机效应
sm_terms <- c(
  "s(孕周_周, k=9, bs='tp')",
  "s(体质指数, k=7, bs='tp')",
  "ti(孕周_周, 体质指数, k=c(6,5), bs=c('tp','tp'))",
  "s(孕妇ID, bs='re')"
)

# 数值类可选变量 -> 收缩样条（bs='ts' 带select）
num_qc <- intersect(have_qc, c(
  "年龄","在参考基因组上比对的比例","重复读段的比例","GC含量",
  "被过滤掉读段数的比例","log原始读段数","检测质量主成分1","检测质量主成分2"
))
if(length(num_qc)){
  sm_terms <- c(sm_terms, sprintf("s(%s, k=5, bs='ts')", num_qc))
}

# IVF 作为参数项（存在才加入）
param_terms <- if("IVF妊娠" %in% have_qc) "IVF妊娠" else NULL

rhs  <- paste(c(param_terms, sm_terms), collapse = " + ")
form <- as.formula(paste(ycol, "~", rhs))
message("[Q1] 公式：\n", deparse(form))

m0 <- mgcv::gam(
  form,
  family = betar(link="logit"),
  method  = "REML",
  select  = TRUE,
  data    = dat
)

# 模型摘要 & 随机效应方差
summ <- summary(m0)
vc   <- gam.vcomp(m0)
capture.output({
  cat("=== 多因子 Beta-GAMM 摘要 ===\n")
  print(summ)
  cat("\n=== 随机效应方差 ===\n")
  print(vc)
  cat("\n=== 并曲性（concurvity）===\n")
  print(concurvity(m0, full = TRUE))
  cat("\n=== 诊断（gam.check）===\n")
  print(gam.check(m0))
}, file = "Q1_模型摘要.txt")
message("[Q1] 已写入：Q1_模型摘要.txt")

# ---------- 3) 部分效应曲线（变量多会自动多面板） ----------
p_eff <- gratia::draw(m0, residuals = FALSE)
ggsave("输出图/Q1_部分效应.png", p_eff, width=11, height=8, dpi=150)
message("[Q1] 已导出：输出图/Q1_部分效应.png")

# ---------- 4) 关键点预测（排除随机效应；其他协变量取样本中位数/众数） ----------
ref_id <- levels(dat$孕妇ID)[1]
kp <- expand.grid(
  `孕周_周`  = c(10,12,14,16),
  `体质指数` = as.numeric(quantile(dat$`体质指数`, probs=c(.2,.5,.8), na.rm=TRUE))
)
# 其他数值协变量取中位数
for(v in num_qc){
  kp[[v]] <- suppressWarnings(stats::median(dat[[v]], na.rm=TRUE))
}
# IVF 取最常见类别
if("IVF妊娠" %in% have_qc){
  ivf_ref <- names(sort(table(dat$IVF妊娠), decreasing = TRUE))[1]
  kp$IVF妊娠 <- factor(ivf_ref, levels = levels(dat$IVF妊娠))
}
kp$孕妇ID <- factor(ref_id, levels = levels(dat$孕妇ID))

pr <- predict(m0, newdata = kp, type = "link", se.fit = TRUE, exclude = "s(孕妇ID)")
kp$预测 <- plogis(pr$fit)
kp$下界 <- plogis(pr$fit - 1.96*pr$se.fit)
kp$上界 <- plogis(pr$fit + 1.96*pr$se.fit)
writexl::write_xlsx(list("关键点预测"=kp), "Q1_关键点_预测.xlsx")
message("[Q1] 已导出：Q1_关键点_预测.xlsx")

# ---------- 5) 等高线底图：P(Y≥4%)（Beta阈值校正 φ） ----------
# 取 Beta 精度参数 φ
phi <- tryCatch(m0$family$getTheta(TRUE),
                error = function(e) if(!is.null(m0$family$theta)) m0$family$theta else 50)
# 4% 的 SV 阈值（样本量校正）
n_obs <- sum(is.finite(dat[[ycol]]))
v_star <- (0.04*(n_obs-1) + 0.5)/n_obs

grid <- expand.grid(
  `孕周_周`  = seq(9, 26, by = 0.1),
  `体质指数` = seq(16, 40, by = 0.5)
)
# 其他数值协变量取中位数
for(v in num_qc){
  grid[[v]] <- suppressWarnings(stats::median(dat[[v]], na.rm=TRUE))
}
# IVF 取最常见类别
if("IVF妊娠" %in% have_qc){
  ivf_ref <- names(sort(table(dat$IVF妊娠), decreasing = TRUE))[1]
  grid$IVF妊娠 <- factor(ivf_ref, levels = levels(dat$IVF妊娠))
}
grid$孕妇ID <- factor(ref_id, levels = levels(dat$孕妇ID))

mu <- as.numeric(predict(m0, newdata=grid, type="response", exclude="s(孕妇ID)"))
a <- mu * phi; b <- (1 - mu) * phi
grid$达标概率 <- pmax(0, pmin(1, 1 - pbeta(v_star, a, b)))

# 导出（不带ID）
writexl::write_xlsx(
  list("BMIx孕周_达标概率"=dplyr::select(grid, `孕周_周`,`体质指数`,达标概率)),
  "Q2_等高线底图.xlsx"
)

g <- ggplot(grid, aes(x=`孕周_周`, y=`体质指数`, fill=达标概率)) +
  geom_raster() +
  viridis::scale_fill_viridis(name="P(Y≥4%)", limits=c(0,1)) +
  geom_contour(aes(z=达标概率), breaks=c(0.5,0.7,0.8,0.9),
               color="white", linewidth=0.4) +
  labs(x="孕周（周）", y="体质指数", title="达标概率 等高线底图（供Q2）") +
  theme_minimal(base_size = 12)
ggsave("输出图/Q2_达标概率_等高线.png", g, width=9, height=6, dpi=150)
message("[Q1→Q2] 已导出：Q2_等高线底图.xlsx、输出图/Q2_达标概率_等高线.png")

# ---------- 6) 分组CV：按孕妇ID分组的 v 折 ----------
set.seed(42)
v <- min(5, nlevels(dat$孕妇ID))
folds <- rsample::group_vfold_cv(dat, group=孕妇ID, v=v)

cv_rows <- lapply(seq_along(folds$splits), function(i){
  sp <- folds$splits[[i]]
  tr <- rsample::analysis(sp)
  te <- rsample::assessment(sp)

  tr$孕妇ID <- droplevels(tr$孕妇ID)
  te$孕妇ID <- factor(as.character(te$孕妇ID), levels = levels(tr$孕妇ID))

  fit <- mgcv::gam(formula(m0), family=betar(link="logit"),
                   method="REML", select=TRUE, data=tr)

  # 合法ID占位 + 排除随机效应
  ref_id2 <- levels(tr$孕妇ID)[1]
  te2 <- te; te2$孕妇ID <- factor(ref_id2, levels=levels(tr$孕妇ID))
  te_pred <- predict(fit, newdata=te2, type="response", exclude="s(孕妇ID)")

  ok  <- is.finite(te_pred) & is.finite(te[[ycol]])
  nOK <- sum(ok)
  mae <- if(nOK>0) mean(abs(te_pred[ok] - te[[ycol]][ok])) else NA_real_
  rho <- if(nOK>1) suppressWarnings(cor(te_pred[ok], te[[ycol]][ok], method="spearman")) else NA_real_

  data.frame(折=i, 训练量=nrow(tr), 测试量=nrow(te),
             有效对数=nOK, MAE=mae, Spearman=rho)
})

cv_tab <- dplyr::bind_rows(cv_rows)
cv_summary <- data.frame(
  折数     = nrow(cv_tab),
  MAE均值   = mean(cv_tab$MAE, na.rm=TRUE),
  MAE标准差 = sd(cv_tab$MAE,   na.rm=TRUE),
  ρ均值     = mean(cv_tab$Spearman, na.rm=TRUE),
  ρ标准差   = sd(cv_tab$Spearman,   na.rm=TRUE)
)
writexl::write_xlsx(list("分组CV_明细"=cv_tab, "分组CV_汇总"=cv_summary),
                    "Q1_分组CV_指标.xlsx")
message("[Q1] 已导出：Q1_分组CV_指标.xlsx")

message("🎉 全部完成：模型摘要、部分效应、关键点预测、等高线底图、分组CV（多因子可选）。")
# ==== END ====
