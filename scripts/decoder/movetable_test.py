import numpy as np
import torch
def load_data_to_tensor(file_path):
    """
    从txt文件中读取数据，并转换为 1x36x100 的 tensor。
    """
    data = np.loadtxt(file_path)
    
    # 确保数据形状正确
    assert data.shape == (36, 100), "数据形状不正确，应该是 36x100"
    
    # 将numpy数组转换为torch的tensor，并调整为1x36x100的形状
    tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    
    return tensor_data

def parse_kmer_from_encoder(enc_kmers, kmer_length=9):
    """
    解析 encoder 中的 kmer 序列，区分连续相同碱基和同一个位置的重复测序。
    """
    parsed_kmers = []

    # 解析每个数据点的 kmer 编码
    for data_idx in range(enc_kmers.shape[0]):  # N 个数据点
        kmer_seq_per_sample = []  # 存储该数据点的所有 kmer 序列
        for kmer_idx in range(0, enc_kmers.shape[1], 4):  # 每4行代表一个碱基的one-hot编码
            base_seq = []
            for col in range(enc_kmers.shape[2]):  # 对每个样本列遍历
                one_hot = enc_kmers[data_idx, kmer_idx:kmer_idx+4, col].argmax().item()
                base = 'ACGT'[one_hot]  # 根据最大值的索引判断是哪个碱基
                base_seq.append(base)
            kmer_seq_per_sample.append(''.join(base_seq))
        
        # 每个样本的 kmer 序列是 9 x 100
        parsed_kmers.append(kmer_seq_per_sample)
    
    return parsed_kmers  # 大小为 [N, 9, 100] 的结构

def simplify_sequence(expanded_matrix):
    """
    按列判断重复碱基，并用 '-' 替代重复的碱基。
    """
    all_processed_matrices = []

    # 遍历每个数据点
    for dataset in expanded_matrix:  # expanded_matrix 大小为 [N, 9, 100]，每个数据点是9行的字符串
        processed_matrix = [list(row) for row in dataset]  # 将每行字符串转为字符列表以便修改
        
        num_rows = len(processed_matrix)  # 行数应为9
        num_cols = len(processed_matrix[0])  # 列数应为100

        # 遍历每一列
        for col in range(num_cols):
            base = processed_matrix[0][col]  # 基准是第一行的碱基
            # 从第二行开始检查是否重复

            for row in range(1, num_rows):
                if processed_matrix[row][col] == base:
                    # 如果当前行和基准相同，替换为 '-'
                    processed_matrix[row][col] = '-'
                else:
                    # 如果遇到不同碱基，更新基准
                    base = processed_matrix[row][col]

        # 将处理后的矩阵行重新组合成字符串
        processed_matrix = [''.join(row) for row in processed_matrix]
        all_processed_matrices.append(processed_matrix)  # 保存该数据点的处理结果

    return all_processed_matrices


def remove_center_base(enc_kmers, parsed_kmers, kmer_length=9):
    """
    移除中心碱基的信息。
    """
    center_idx = kmer_length // 2  # 中心碱基索引为4

    for data_idx in range(enc_kmers.shape[0]):
        for col in range(enc_kmers.shape[2]):
            # 获取当前列对应的 kmer 序列
            kmer_seq = parsed_kmers[data_idx][col]

            # 中心碱基是第 center_idx 个碱基
            center_base = kmer_seq[center_idx]

            # 对于当前的 one-hot 编码行，找到中心碱基所在的4行并将其置零
            for kmer_idx in range(0, enc_kmers.shape[1], 4):
                # 判断该行是否是中心碱基的编码
                one_hot = enc_kmers[data_idx, kmer_idx:kmer_idx+4, col].argmax().item()
                base = 'ACGT'[one_hot]

                if base == center_base:
                    # 将该 one-hot 编码置零
                    enc_kmers[data_idx, kmer_idx:kmer_idx+4, col] = torch.zeros(4)

    return enc_kmers


def process_kmers(enc_kmers, kmer_length=9):
    """
    执行解析和移除操作。
    """
    # 第一步：解析 kmer 序列，区分重复碱基与重复测序
    parsed_kmers = parse_kmer_from_encoder(enc_kmers, kmer_length)


    # 将第一个数据点的内容输出到 txt 文件中
    output_path = "/public/home/xiayini/project/NewL/Decoder/scripts/decoder/tensor_output2.txt"
    with open(output_path, "w") as f:
        for row in parsed_kmers[0]:
            f.write(row+"\n")
    # 处理 parsed_kmers 数据，确保传入的是字符列表
    repetition_results = simplify_sequence(parsed_kmers)
    
    for result in repetition_results[0]:  # 仅输出第一个数据点的结果
        print(result)

    return repetition_results


# 假设 enc_kmers 是一个大小为 [2048, 36, 100] 的 Tensor
# 测试加载数据的函数
file_path = "/public/home/xiayini/project/NewL/Decoder/scripts/decoder/tensor_output.txt"  # 替换为你的txt文件路径
enc_kmers = load_data_to_tensor(file_path)

# 调用函数处理数据
processed_kmers = process_kmers(enc_kmers)
