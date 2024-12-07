import torch
from torch import nn
import torch.nn.functional as F

from remora import constants
from remora.activations import swish


class network(nn.Module):
    _variable_width_possible = False

    def __init__(
        self,
        size=constants.DEFAULT_NN_SIZE,
        kmer_len=constants.DEFAULT_KMER_LEN,
        num_out=2,
    ):
        super().__init__()
        self.sig_conv1 = nn.Conv1d(1, 4, 5)
        self.sig_bn1 = nn.BatchNorm1d(4)
        self.sig_conv2 = nn.Conv1d(4, 16, 5)
        self.sig_bn2 = nn.BatchNorm1d(16)
        self.sig_conv3 = nn.Conv1d(16, size, 9, 3)
        self.sig_bn3 = nn.BatchNorm1d(size)

        self.seq_conv1 = nn.Conv1d(kmer_len * 4, 16, 5)
        self.seq_bn1 = nn.BatchNorm1d(16)
        self.seq_conv2 = nn.Conv1d(16, size, 13, 3)
        self.seq_bn2 = nn.BatchNorm1d(size)

        self.merge_conv1 = nn.Conv1d(size, size, 5)
        self.merge_bn = nn.BatchNorm1d(size)

        self.lstm1 = nn.LSTM(size, size, 1)
        self.lstm2 = nn.LSTM(size, size, 1)

        latent_size=32
        self.latent_fc4 = nn.Linear(size, latent_size)  # 全连接层，输出维度是 32
        self.latent_fc5 = nn.Linear(size, latent_size)  # 全连接层，输出维度是 32
        
        self.activation = torch.nn.Tanh()  # 你可以根据需求使用其他激活函数，如 Sigmoid 或 Tanh nn.Sigmoid()
        

        # 定义6个分类器
        self.classifier_typeNum_list=[1,1,1,2,3,1]
        self.fc1 = nn.Linear(latent_size, self.classifier_typeNum_list[0])  
        self.fc2 = nn.Linear(latent_size, self.classifier_typeNum_list[1])
        self.fc3 = nn.Linear(latent_size, self.classifier_typeNum_list[2])
        self.fc4 = nn.Linear(latent_size, self.classifier_typeNum_list[3])
        self.fc5 = nn.Linear(latent_size, self.classifier_typeNum_list[4])
        self.fc6 = nn.Linear(latent_size, self.classifier_typeNum_list[5])


        self.dropout = nn.Dropout(p=0.3)

    def forward(self, sigs, seqs):
        # inputs are BFT (batch, feature, time)
        sigs_x = swish(self.sig_bn1(self.sig_conv1(sigs)))
        sigs_x = swish(self.sig_bn2(self.sig_conv2(sigs_x)))
        sigs_x = swish(self.sig_bn3(self.sig_conv3(sigs_x)))

        seqs_x = swish(self.seq_bn1(self.seq_conv1(seqs)))
        seqs_x = swish(self.seq_bn2(self.seq_conv2(seqs_x)))

        # z = torch.cat((sigs_x, seqs_x), 1)
        z = sigs_x

        z = swish(self.merge_bn(self.merge_conv1(z)))
        z = z.permute(2, 0, 1)
        z = swish(self.lstm1(z)[0])
        z = torch.flip(swish(self.lstm2(torch.flip(z, (0,)))[0]), (0,))
        z = z[-1].permute(0, 1)

        latent_z4=self.activation(self.latent_fc4(z))

        latent_z5=self.activation(self.latent_fc5(z))
        
        # z1 = self.fc1(z)
        # z2 = self.fc2(z)
        # z3 = self.fc3(z)
        z4 = self.fc4(latent_z4)
        z5 = self.fc5(latent_z5)
        # z6 = self.fc6(z)

        output = torch.cat((z4,z5), dim=1)
        # output = torch.cat((z1, z2,z3,z4,z5,z6), dim=1)
        return output
    
    def embedding(self, sigs, seqs):
        # inputs are BFT (batch, feature, time)
        sigs_x = swish(self.sig_bn1(self.sig_conv1(sigs)))
        sigs_x = swish(self.sig_bn2(self.sig_conv2(sigs_x)))
        sigs_x = swish(self.sig_bn3(self.sig_conv3(sigs_x)))

        seqs_x = swish(self.seq_bn1(self.seq_conv1(seqs)))
        seqs_x = swish(self.seq_bn2(self.seq_conv2(seqs_x)))

        z = sigs_x

        z = swish(self.merge_bn(self.merge_conv1(z)))
        z = z.permute(2, 0, 1)
        z = swish(self.lstm1(z)[0])
        z = torch.flip(swish(self.lstm2(torch.flip(z, (0,)))[0]), (0,))
        z = z[-1].permute(0, 1)
        
        # z4 = self.fc4(z)
        # z5 = self.fc5(z)
        # output = torch.cat((z4,z5), dim=1)
        return z

    def latent(self, sigs, seqs):
        # inputs are BFT (batch, feature, time)
        sigs_x = swish(self.sig_bn1(self.sig_conv1(sigs)))
        sigs_x = swish(self.sig_bn2(self.sig_conv2(sigs_x)))
        sigs_x = swish(self.sig_bn3(self.sig_conv3(sigs_x)))

        seqs_x = swish(self.seq_bn1(self.seq_conv1(seqs)))
        seqs_x = swish(self.seq_bn2(self.seq_conv2(seqs_x)))

        z = sigs_x
        
        z = swish(self.merge_bn(self.merge_conv1(z)))
        z = z.permute(2, 0, 1)
        z = swish(self.lstm1(z)[0])
        z = torch.flip(swish(self.lstm2(torch.flip(z, (0,)))[0]), (0,))
        z = z[-1].permute(0, 1)


        latent_z4=self.activation(self.latent_fc4(z))

        latent_z5=self.activation(self.latent_fc5(z))
        # z4 = self.fc4(z)
        # z5 = self.fc5(z)
        # output = torch.cat((z4,z5), dim=1)
        return latent_z4,latent_z5