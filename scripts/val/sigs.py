import os
import json
import atexit
from shutil import copyfile
from itertools import islice
from sklearn.metrics import confusion_matrix
import pandas as pd
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


def train_model(
    device,
    remora_dataset_path,
    batch_size,
    model_path,
    chunks_per_epoch,
    finetune_path,
    super_batch_size,
    super_batch_sample_frac,
    output_path,
    location_name
):
    kmer_context_bases=None
    chunk_context=[50,50]

    override_metadata = {"extra_arrays": {}}
    if kmer_context_bases is not None:
        override_metadata["kmer_context_bases"] = kmer_context_bases
    if chunk_context is not None:
        override_metadata["chunk_context"] = chunk_context

    paths, props, hashes = load_dataset(remora_dataset_path)
    dataset = RemoraDataset(
        [
            CoreRemoraDataset(
                path,
                override_metadata=override_metadata,
            )
            for path in paths
        ],
        props,
        hashes,
        batch_size=batch_size,
        super_batch_size=super_batch_size,
        super_batch_sample_frac=super_batch_sample_frac,
    )
    
    ckpt, model = model_util.continue_from_checkpoint(
        finetune_path, model_path
    )
    model = model.to(device)
    trn_ds = dataset
    trn_loader = DataLoader(
        trn_ds,
        batch_size=None,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        worker_init_fn=dataloader_worker_init,
    )


    batches_per_epoch = int(np.ceil(chunks_per_epoch / batch_size))

    model.eval()
    basetype_list=dataset.metadata.mod_long_names
    for epoch_i, (enc_kmers, sigs, labels) in enumerate(
        islice(trn_loader, batches_per_epoch)
    ):
   
        embeddings = model.embedding(sigs.to(device), enc_kmers.to(device))

        # Convert the embeddings to a numpy array
        # embeddings_np = embeddings.cpu().numpy()
        embeddings_np = embeddings.cpu().detach().numpy()


        # print(f"check base label:{labels[0]} and {labels[-1]}")
        basenames=[]
        for label in labels:
            str_label = str(label.item())  # 将Tensor中的单个值转换为Python整数
            n = len(basetype_list)  # 假设 n 是 basetype_list 的长度
            if str_label.isdigit() and 1 <= int(str_label) <= n:
                basename = str(basetype_list[int(str_label) - 1])
                basenames.append(basename)


        def getCsv(basenames,):
                
            embeddings_df = pd.DataFrame(embeddings_np)
            embeddings_df.insert(0, "basename", basenames)  # Insert basenames as the first column
            embeddings_df.to_csv(f"{output_path}/{location_name}_embedding.csv", index=False, mode='a', header=False)  # Append without header

            embeddings_np = embeddings.cpu().detach().numpy()
            
            embeddings_df = pd.DataFrame(embeddings_np)
            embeddings_df.insert(0, "basename", basenames)  # Insert basenames as the first column

            embeddings_df.to_csv(f"{output_path}/{location_name}_sigs.csv", index=False, mode='a', header=False)  # Append without header



if __name__ == "__main__":

    version=''

    train_mod="T_5mC"
    # train_mod="C_5mC_T"
    # 
    train_path=f"/public/home/xiayini/project/NewL/Decoder/results_newdata/{train_mod}/epoch200_trainDataset"
    finetune_path=f"{train_path}/model_final.checkpoint"
    # finetune_path=f"{train_path}/model_000036.checkpoint"
    # finetune_path="/public/home/xiayini/project/NewL/Decoder/results2/T_5mC/epoch50_Tanh/model_000044.checkpoint"

    
    mod_type_name='T_5mC' # C_5mC_T T_5mC
    location_name="merge"
    dataset_name="val" # train val
    dataset_path = f"/public/home/xiayini/project/NewL/Decoder/newdata/{location_name}/{dataset_name}/{mod_type_name}_dataset.jsn"  # 测试数据集路径


    # finetune_path = f"{path}/{model_name}.checkpoint"  # 训练好的模型checkpoint路径
    output_path=f'{train_path}/val_{mod_type_name}'
    if(os.path.exists(output_path)==False):
        os.system("mkdir -p "+output_path)
    output_csv_file=f'{output_path}'
    
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU
    # model_path="/public/home/xiayini/project/NewL/Decoder/scripts/decoderV2.0/model_Pyrimidine.py"
    model_path="/public/home/xiayini/project/NewL/Decoder/scripts/decoder/model_Pyrimidine.py"

    os.system("rm "+f"{output_path}/{location_name}_*.csv")
    train_model(device,dataset_path,2048,model_path,10000,finetune_path,20000,0.1,output_path,location_name)