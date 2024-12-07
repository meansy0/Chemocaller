

# script2.R
# 加载 script1.R 中的函数
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source("my_function.r")

# PCA

# 获取当前脚本文件的目录并设置为工作目录

# type
# merge_embedding merge_chemConstruction4 merge_chemConstruction5 merge_sigs
csv_file="/public/home/xiayini/project/NewL/Decoder/results_newdata_Process3.0/C_5mC_T/epoch100_model_noseqs_lstm2/train_C_5mC_T/chemConstruction5.csv"
# PCA_plot(csv_file) #or
umap_plot(csv_file)
ggsave('/public/home/xiayini/project/NewL/Decoder/scripts/val/co5.png')
selected_umap_plot(csv_file, c("5mC", "T", "label3"))

