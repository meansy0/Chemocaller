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




if __name__ == "__main__":

    for enc_kmers, sigs, labels in islice(trn_loader, 1024):
        print('ww')