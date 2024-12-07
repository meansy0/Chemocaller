import os
import json
import atexit
from shutil import copyfile
from itertools import islice
from sklearn.metrics import confusion_matrix

import torch
import numpy as np
from tqdm import tqdm
from thop import profile
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from data_chunks import (
    RemoraDataset,
    CoreRemoraDataset,
    load_dataset,
    dataloader_worker_init,
)
from remora import (
    constants,
    util,
    log,
    RemoraError,
    model_util,
    validate,
)

def load_model_for_evaluation(finetune_path, device):
    model_path='/public/home/xiayini/project/NewL/Decoder/scripts/decoder/model_Pyrimidine.py'
    ckpt, model = model_util.continue_from_checkpoint(
                finetune_path, model_path=model_path
            )
            
    model = model.to(device)
    return model


def loaderData(remora_dataset_path):
    # 路径设置
    # remora_dataset_path='/public/home/xiayini/project/NewL/Decoder/data/merge/T_5mC_train_dataset.jsn'
    paths, props, hashes = load_dataset(remora_dataset_path)
    dataset = RemoraDataset(
        [
            CoreRemoraDataset(
                path,
                override_metadata=None,
            )
            for path in paths
        ],
        props,
        hashes,
        batch_size=2048,
        super_batch_size=100000,
        super_batch_sample_frac=1.0,
    )

    kmer_context_bases=None
    chunk_context=[50,50]

    override_metadata = {"extra_arrays": {}}
    if kmer_context_bases is not None:
        override_metadata["kmer_context_bases"] = kmer_context_bases
    if chunk_context is not None:
        override_metadata["chunk_context"] = chunk_context

    trn_ds, val_ds = dataset.train_test_split(
        10,
        override_metadata=override_metadata,
    )
    trn_loader = DataLoader(
        trn_ds,
        batch_size=None,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        worker_init_fn=dataloader_worker_init,
    )
    return trn_loader,dataset.metadata.mod_long_names



def evaluate_model(model, dataLoader, device,basetype_list):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    total_correctly_labeled=0
    total_sam=0
    print(basetype_list)
    with torch.no_grad():  # 不计算梯度，节省内存
        for enc_kmers, sigs, labels in islice(dataLoader, 10):
            outputs = model(sigs.to(device), enc_kmers.to(device))
            # 根据你模型输出的逻辑，计算损失和准确率
            classifier_typeNum_list = [1, 1, 1, 2, 3, 1]
            # 甲基化类型映射表
            chemo_base_list = {
                "C":   (0, 0, 0, 0, 0, 0),
                "T":   (0, 0, 0, 1, 0, 0),
                "5mU": (0, 0, 0, 1, 1, 0),
                "5mC": (0, 0, 0, 0, 1, 0),
                "5hmC": (0, 0, 0, 0, 2, 0),
            }

            # 将标签映射为对应的化学特性
            location_labels = []
            for label in labels:
                str_label = str(label.item())  # 将Tensor中的标签转换为字符串
                n = len(basetype_list)  # 获取 basetype_list 的长度
                if str_label.isdigit() and 1 <= int(str_label) <= n:
                    basename = str(basetype_list[int(str_label) - 1])  # 获取对应的 basetype 名称
                    # print(basename)
                else:
                    print("aa")

                location_label = chemo_base_list[basename]  # 获取化学特性
                location_labels.append(location_label)
            # 将 location_labels 转换为 tensor
            location_labels = torch.tensor(location_labels).to(device)

            # 初始化损失和正确分类标记

            num_o = 0
            location_labels_cpu = location_labels.cpu().numpy()
            correctly_labeled = np.ones(location_labels_cpu.shape[0], dtype=bool)

            # 遍历分类器类型和输出，计算损失并判断分类准确性
            for lo, c in enumerate(classifier_typeNum_list):
                if c > 1:
                    Zn = outputs[:, num_o:num_o + c]  # 获取第 lo 个分类器的输出

                    correctly_labeled &= (np.argmax(Zn.detach().cpu().numpy(), axis=1) == location_labels_cpu[:, lo])

                    num_o += c  # 更新输出索引

            acc = correctly_labeled.sum() / len(location_labels_cpu)
            print(f"acc:{str(correctly_labeled.sum())}\t{len(location_labels_cpu)}")
            total_correctly_labeled+=correctly_labeled.sum()
            total_sam+=len(location_labels_cpu)
            # print(acc)
            
            # total_loss += loss.item() * labels.size(0)
            # correct_predictions += correctly_labeled.sum()
            # total_samples += labels.size(0)

    avg_acc = total_correctly_labeled / total_sam
    # accuracy = correct_predictions / total_samples

    return avg_acc


if __name__ == "__main__":
    # 路径设置

    mod_type_name='T_5mC'
    model_name='model_final'

    path=f"/public/home/xiayini/project/NewL/Decoder/results/{mod_type_name}/epoch10_test"
    
    finetune_path="/public/home/xiayini/project/NewL/Decoder/results/T_5mC/epoch4_test/model_000004.checkpoint"
    # finetune_path = f"{path}/{model_name}.checkpoint"  # 训练好的模型checkpoint路径
    output_path=f'{path}/val'
    if(os.path.exists(output_path)==False):
        os.system("mkdir -p "+output_path)
    # copy_model_path=util.resolve_path(os.path.join(output_path, "model_Pyrimidine.py"))
    # f"{output_path}/{model_name}.checkpoint"
    dataset_path="/public/home/xiayini/project/NewL/Decoder/data/merge/C_5mC_T_train_dataset.jsn"
    # dataset_path = f"/public/home/xiayini/project/NewL/Decoder/data/merge/{mod_type_name}_train_dataset.jsn"  # 测试数据集路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU

    # 加载模型
    model = load_model_for_evaluation(finetune_path, device)

    dataLoader,basetype_list=loaderData(dataset_path)
    accuracy = evaluate_model(model, dataLoader, device, basetype_list)

    # 评估模型
    print(f"Test Accuracy: {accuracy:10f}")
    # print(f"Test Average Loss: {avg_loss:.4f}")