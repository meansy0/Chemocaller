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
    output_path
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
    epoch_summ = trn_ds.epoch_summary(batches_per_epoch)
    model.eval()
    basetype_list=dataset.metadata.mod_long_names
    for epoch_i, (enc_kmers, sigs, labels) in enumerate(
        islice(trn_loader, batches_per_epoch)
    ):
        # print(f'label1: {np.sum(labels.cpu().numpy()==1)}')
        # print(f'label2: {np.sum(labels.cpu().numpy()==2)}')
        classifier_typeNum_list=[1,1,1,2,3,1]
        classifier_num=6
        vaild_classifier_num=2

        outputs= model(sigs.to(device), enc_kmers.to(device))
        
        embeddings = model.embedding(sigs.to(device), enc_kmers.to(device))

        latent4,latent5=model.latent(sigs.to(device), enc_kmers.to(device))
        # Convert the embeddings to a numpy array
        # embeddings_np = embeddings.cpu().numpy()
        embeddings_np = embeddings.cpu().detach().numpy()
        
        sigs_np = sigs.squeeze(1).cpu().detach().numpy()
        latent4_np=latent4.cpu().detach().numpy()
        latent5_np=latent5.cpu().detach().numpy()

        # 转换为locatioin标签

        location_labels = []
        chemo_base_list = {
            "C":   (0,0,0,0,0,0),
            "T":   (0,0,0,1,0,0),
            "5mU": (0,0,0,1,1,0),
            "5mC": (0,0,0,0,1,0),
            "5hmC":(0,0,0,0,2,0),

        }
        dic_chem_to_features = {
            "1": (0),         # 分类器1 null
            "2": (1),         # 分类器2 =O
            "3": (0),         # 分类器3 null
            "4": (0,1),       # 分类器4 =O     -NH2      
            "5": (0,1,2),     # 分类器5 null   -CH3   -CH₂-OH
            "6": (0),         # 分类器6 null
        }

        # print(f"check base label:{labels[0]} and {labels[-1]}")
        basenames=[]
        for label in labels:
            str_label = str(label.item())  # 将Tensor中的单个值转换为Python整数

            n = len(basetype_list)  # 假设 n 是 basetype_list 的长度
            if str_label.isdigit() and 1 <= int(str_label) <= n:
                basename = str(basetype_list[int(str_label) - 1])
                basenames.append(basename)

            location_label=chemo_base_list[basename]

            location_labels.append(location_label)

        # 转换为张量
        location_labels = torch.tensor(location_labels)
        loss=0
        lo=0
        num_o=0
        location_labels_cpu = location_labels.cpu().numpy()
        correctly_labeled = np.ones(location_labels_cpu.shape[0], dtype=bool)
        
        for c in classifier_typeNum_list:
            if(c>1):
                Zn=outputs[:, num_o:num_o+c]

                max_positions = torch.argmax(Zn, dim=1).detach().cpu().numpy()

                correctly_labeled &= (np.argmax(Zn.detach().cpu().numpy(), axis=1) == location_labels_cpu[:, lo])

                outputzn_np=np.argmax(Zn.detach().cpu().numpy(), axis=1)

                # outputzn_np = Zn.cpu().detach().numpy()
                outputzn_df = pd.DataFrame(outputzn_np)
                outputzn_df.insert(0, "basename", basenames)  # Insert basenames as the first column
                outputzn_df.to_csv(f"{output_path}/outputz{str(lo+1)}.csv", index=False, mode='a', header=False)  # Append without header

                num_o=num_o+c             
            lo=lo+1

        # Save the embeddings to a CSV file for use in R
        embeddings_df = pd.DataFrame(embeddings_np)
        embeddings_df.insert(0, "basename", basenames)  # Insert basenames as the first column

        embeddings_df.to_csv(f"{output_path}/embedding.csv", index=False, mode='a', header=False)  # Append without header

        latent4_df=pd.DataFrame(latent4_np)     
        latent5_df=pd.DataFrame(latent5_np)  

        latent4_df.insert(0, "basename", basenames)  # Insert basenames as the first column
        latent4_df.to_csv(f"{output_path}/chemConstruction4.csv", index=False, mode='a', header=False)  # Append without header
  
        latent5_df.insert(0, "basename", basenames)  # Insert basenames as the first column
        latent5_df.to_csv(f"{output_path}/chemConstruction5.csv", index=False, mode='a', header=False)  # Append without header
        acc = correctly_labeled.sum() / len(location_labels_cpu)

        
        
        sigs_df = pd.DataFrame(sigs_np)
        sigs_df.insert(0, "basename", basenames)  # Insert basenames as the first column

        sigs_df.to_csv(f"{output_path}/sigs.csv", index=False, mode='a', header=False)  # Append without header


        print(f"batch dataset {str(epoch_i)}: {acc}")

    os.system("paste "+f"{output_path}/outputz4.csv {output_path}/outputz5.csv"+" -d '\t' |tr ',' '\t' |cut -f 1,2,4 >"+f"{output_path}/output.csv")
    os.system(f"sort -k1,1 {output_path}/output.csv -o {output_path}/output.csv")

def main(train_path,dataset_name,finetune_path,dataset_path,val_dataset_mod):
    model_path=f"{train_path}/model_Pyrimidine.py"
    # finetune_path = f"{path}/{model_name}.checkpoint"  # 训练好的模型checkpoint路径
    output_path=f'{train_path}/{dataset_name}_{val_dataset_mod}'
    if(os.path.exists(output_path)==False):
        os.system("mkdir -p "+output_path)
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU

    # model_path="/public/home/xiayini/project/NewL/Decoder/scripts/decoder/model_Pyrimidine.py"
    # model_file=
    os.system("rm "+f"{output_path}/merge_*.csv")
    # train_model(device,dataset_path,2048,model_path,10000000,finetune_path,20000,0.1,output_path)
    train_model(device,dataset_path,2048,model_path,10000,finetune_path,20000,0.1,output_path)



if __name__ == "__main__":
    # in_data_path="/public/home/xiayini/project/NewL/Decoder/newdata_Process3.0/merge"
    # out_path="/public/home/xiayini/project/NewL/Decoder/results_newdata_Process3.0"

    in_data_path="/public/home/xiayini/project/NewL/Decoder/newdata_Process3.0/merge"
    out_path="/public/home/xiayini/project/NewL/Decoder/results_newdata_Process3.0"

    # train dataset 
    train_dataset_mod="C_5mC_T" # T_5mC C_5mC_T 
    # test dataset 
    val_dataset_mod="C_5mC_T" # T_5mC C_5mC_T 

    # model_noseqs_lstm3 model_noseqs_lstm2 model_seqs_lstm2
    epochs_num="100"
    model_type="model_noseqs_lstm2"
    train_epoch_name=f"epoch{epochs_num}_{model_type}"


    model_name="model_000056" # model_final model_000060... 

    train_path=f"{out_path}/{train_dataset_mod}/{train_epoch_name}"
    finetune_path=f"{train_path}/{model_name}.checkpoint"

    dataset_name="train" # train val
    dataset_path = f"{in_data_path}/{dataset_name}/{val_dataset_mod}_dataset.jsn"  # 测试数据集路径

    main(train_path,dataset_name,finetune_path,dataset_path,val_dataset_mod)
