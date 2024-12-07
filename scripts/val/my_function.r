# 加载指定的 R 包
load_package <- function(package_name) {
  if (!(package_name %in% loadedNamespaces())) {
    library(package_name, character.only = TRUE)
    message(paste(package_name, "successful!"))
  } else {
    message(paste(package_name, " has been loaded!"))
  }
}
# # 调用函数，传递包名
# load_package("ggplot2")


# csv file format:
    # 1:type
    # 2-:data
PCA_plot <- function(csvfile) {
    load_package("ggplot2")
    # 读取CSV文件
    data <- read.csv(csvfile)
    # 提取type列
    type <- data[, 1]
    # 提取数值数据，从第二列开始
    numeric_data <- data[, -1]
    # 对数据进行标准化
    scaled_data <- scale(numeric_data)

    # 进行PCA降维
    pca_result <- prcomp(numeric_data, center = TRUE, scale. = TRUE)
    # 显示PCA的摘要信息
    summary(pca_result)
    # 查看主成分的载荷（变量对主成分的贡献）
    print(pca_result$rotation)
    library(reshape2)# 查看数据在主成分空间中的坐标
    # print(pca_result$x)

    # 将PCA结果转换为数据框
    pca_data <- data.frame(pca_result$x)

    # 将type列添加到PCA结果数据框中
    pca_data$type <- type

    # 绘制主成分1和主成分2的散点图
    ggplot(pca_data, aes(x = PC1, y = PC2, color = type)) +
    geom_point() +
    labs(title = "PCA Plot", x = "Principal Component 1", y = "Principal Component 2") +
    theme_minimal()
    # # 如果需要提取前两个主成分的数据
    # pca_reduced_data <- pca_data$x[, 1:2]
}


# csv file format:
    # 1:type
    # 2-:data
umap_plot <- function(csvfile) {
    load_package("ggplot2")
    load_package("umap")
    data <- read.csv(csv_file)

    # 去除重复数据（假设 data_unique 是你想用的唯一数据）
    data_unique <- data[!duplicated(data), ]

    # 执行 UMAP
    umap_result <- umap(data_unique[, -1], n_neighbors = 15, min_dist = 0.1, metric = "euclidean")

    # 将 UMAP 结果转换为数据框
    umap_data <- data.frame(umap_result$layout)
    umap_data$type <- data_unique[, 1]

    # 使用 ggplot2 可视化 UMAP 结果
    ggplot(umap_data, aes(x = X1, y = X2, color = as.factor(type))) +
    geom_point() +
    labs(title = "UMAP Plot", x = "UMAP Dimension 1", y = "UMAP Dimension 2") +
    theme_minimal()


}

selected_umap_plot <- function(csvfile, selected_labels) {
    load_package("ggplot2")
    load_package("umap")
    
    # 读取 CSV 文件
    data <- read.csv(csvfile)

    # 去除重复数据
    data_unique <- data[!duplicated(data), ]

    # 筛选出指定的标签
    data_filtered <- data_unique[data_unique$type %in% selected_labels, ]

    # 检查是否有足够的数据进行 UMAP
    if (nrow(data_filtered) < 2) {
        stop("筛选后的数据点不足，无法执行 UMAP。")
    }

    # 执行 UMAP
    umap_result <- umap(data_filtered[, -1], n_neighbors = 15, min_dist = 0.1, metric = "euclidean")

    # 将 UMAP 结果转换为数据框
    umap_data <- data.frame(umap_result$layout)
    umap_data$type <- data_filtered$type

    # 使用 ggplot2 可视化 UMAP 结果
    ggplot(umap_data, aes(x = X1, y = X2, color = as.factor(type))) +
        geom_point() +
        labs(title = "UMAP Plot", x = "UMAP Dimension 1", y = "UMAP Dimension 2") +
        theme_minimal()
}

# 使用示例
# umap_plot("your_data.csv", c("label1", "label2", "label3"))
