import torch.nn as nn
import torch, random
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
import math
import torch.nn as nn
from torch import nn
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Config():
    def __init__(self) -> None:
        self.hidden_dim = 64
        self.bias = True
        self.path = "PEMS08_adj.csv"   
        self.Dinput_size = 1  # 特征的个数
        self.Dhidden_size = 64
        self.n_pred = 3
        self.node = 170
class GCNGRUCell(nn.Module):
    def __init__(self, config):
        super(GCNGRUCell, self).__init__()
        self.GCN1 = GCN(config.hidden_dim, config.hidden_dim * 2, config.path, bias=0.4)  
        self.GCN2 = GCN(config.hidden_dim, config.hidden_dim, config.path, bias=0.6)  
    def forward(self, input_tensor, hidden):
        batch, Node = input_tensor.shape
        hidden = hidden.reshape(batch, -1)
        combinedGCN = torch.sigmoid(self.GCN1(input_tensor, hidden))  
        r, u = torch.chunk(combinedGCN, chunks=2, dim=1)
        c = torch.tanh(self.GCN2(input_tensor, r * hidden)) 
        h_next = u * hidden + (1.0 - u) * c  
        h_next = h_next.reshape(input_tensor.size(0), input_tensor.size(1), -1)
        return h_next, h_next
    def init_hidden(self, batch_size, node, hidden):  
        init_h = torch.zeros(batch_size, node, hidden).to(device)
        return init_h
class EndGCNGRU(nn.Module):
    def __init__(self, config):
        super(EndGCNGRU, self).__init__()
        self.cell_list = GCNGRUCell(config)
        self.hidden_dim = config.hidden_dim
    def forward(self, input_tensor):
        b, n, seq_len = input_tensor.size() 
        hidden_state = self._init_hidden(batch_size=b, node=n, hidden=self.hidden_dim)
        cur_layer_input = input_tensor
        for t in range(seq_len):  
            output, hidden_state = self.cell_list(input_tensor=cur_layer_input[:, :, t], hidden=hidden_state)
            output = output.unsqueeze(1)
            if t == 0:
                outs_dec = output
            else:
                outs_dec = torch.cat([outs_dec, output], dim=1)
        outs_dec = outs_dec.reshape(outs_dec.shape[0], outs_dec.shape[1], -1)  
        hidden_state = hidden_state.reshape(outs_dec.shape[0], -1).unsqueeze(1)  
        return outs_dec, hidden_state
    def _init_hidden(self, batch_size, node, hidden): 
        X = self.cell_list.init_hidden(batch_size, node, hidden)
        return X  
class GCN(nn.Module): 
    def __init__(self, input_size, hidden_size, adjpath, bias):
        super(GCN, self).__init__() 
        self.adj = self.ADJ(adjpath)
        self._num_gru_units = input_size 
        self._output_dim = hidden_size 
        self._bias_init_value = bias
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()
    def reset_parameters(self):  
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)
    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        graph_data = self.adj.to(device)  
        graph_data = GCN.process_graph(graph_data) 
        inputs = inputs.unsqueeze(2) 
        hidden_state = hidden_state.reshape(batch_size, num_nodes, self._num_gru_units)
        batch_size, num_nodes, _ = inputs.shape 
        inputs = inputs.to(torch.device("cuda"))
        hidden_state = hidden_state.to(torch.device("cuda"))
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size))
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = graph_data @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size))
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1))
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        # print("outputs.shape", outputs.shape)
        return outputs
    @staticmethod
    def process_graph(graph_data):  
        N = graph_data.size(0)  
        matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device) 
        graph_data = graph_data + matrix_i 
        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  
        degree_matrix = torch.pow(degree_matrix, -0.5).flatten() 
        degree_matrix[degree_matrix == float("inf")] = 0. 
        degree_matrix = torch.diag(degree_matrix) 
        out = graph_data.matmul(degree_matrix).transpose(0, 1).matmul(degree_matrix)
        return out  
    def ADJ(self, adjpath):
        df = pd.read_csv(adjpath, encoding='utf-8', header=None)
        adj = np.array(df, dtype=np.float32)
        adj = torch.from_numpy(adj)  
        return adj
class Attention(nn.Module):  
    def __init__(self, config):
        super(Attention, self).__init__()
        self.attn = nn.Linear(config.hidden_dim + config.Dhidden_size, config.Dhidden_size, bias=False)  # 输出的维度是任意的
        self.v = nn.Linear(config.Dhidden_size, 1, bias=False)  # 将输出维度置为1
        self.node = config.node
    def forward(self, kfc,enc_output):
        energy = torch.tanh(self.attn(kfc))
        attention = self.v(
            energy) 
        return attention, enc_output
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.decoder = nn.GRU(int(config.Dhidden_size +1), config.Dhidden_size, batch_first=True)
        self.liner = nn.Linear(config.Dhidden_size, 1)
        self.Dhidden_size = config.Dhidden_size
    def forward(self, target, hidden):
        out, hidden = self.decoder(target, hidden)
        return out, hidden
class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionLayer, self).__init__()
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
    def forward(self, input_data):
        queries = self.query(input_data) 
        keys = self.key(input_data) 
        values = self.value(input_data) 
        scores = torch.matmul(queries, keys.transpose(1, 2)) 
        attention_weights = nn.functional.softmax(scores, dim=-1) 
        attended_values = torch.matmul(attention_weights, values)
        output_data = attended_values + input_data  
        return output_data
class AttentionMode(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMode, self).__init__()
        self.attention_layer = nn.Linear(input_dim, 64)
        self.softmax = nn.Softmax(dim=0)
        self.layer2 = nn.Sequential(
            nn.Linear(64, 64 // 8),
            nn.GELU(approximate='none'),
            nn.Linear(64 // 8, 1),
        )
    def forward(self, inputs):
        out = self.layer2(inputs)
        return out    
class RAGCRN(nn.Module):  
    def __init__(self, config):
        super(RAGCRN, self).__init__()
        self.enc = EndGCNGRU(config)
        self.dec = Decoder(config)
        self.n_pred = config.n_pred
        self.hidden = config.hidden_dim
        self.node = config.node
        self.attention = Attention(config)
        self.ll = nn.Linear(config.Dhidden_size + config.Dhidden_size, config.Dinput_size)
        self.dog=AttentionMode(input_dim=64)    
    def reset(self):
                nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)
                nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)
    def forward(self, train, target, teacher_forcing_ratio=0.5):
        target = target.transpose(1, 2)
        outputenc, hidden = self.enc(train)
        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(1, hidden.shape[1], int(hidden.shape[2] / self.hidden), -1)
        hidden = hidden.reshape(1, hidden.shape[1] * hidden.shape[2], -1)
        output = torch.zeros_like(target[:, 0:1, :])
        output = output.transpose(1, 2)
        output = output.reshape(output.shape[0] * output.shape[1], -1).unsqueeze(2)
        output = output.repeat(1, 1, 1 * self.hidden+1)
        kvv2=outputenc
        for t in range(self.n_pred):
            if t == 0:
                input = output
            else:
                teacher_force = random.random() < teacher_forcing_ratio
                input = (target[:, t - 1:t, :] if teacher_force else output)
            outputenc=kvv2
            output, hidden_dec = self.dec(input, hidden)
            end_hidden=hidden_dec
            enc_output=outputenc
            enc_output=enc_output.view(12,-1,64)
            attention_weights = self.dog(enc_output)
            result = torch.sum(attention_weights, dim=(1, 2)).view(12, 1)  
            topk_values, topk_indices = torch.topk(result.view(-1), 9)
            enc_output = enc_output[topk_indices].view(-1,9,64)
            batch2, seq_len2, _ = enc_output.shape
            _, _, hidden2 = end_hidden.shape
            enc_output = enc_output.reshape(batch2, seq_len2, hidden2, -1)
            enc_output = enc_output.permute(0, 3, 1, 2)
            enc_output = enc_output.reshape(-1, seq_len2, hidden2)
            s = end_hidden.repeat(seq_len2, 1, 1)
            s = s.transpose(0, 1)
            DH=torch.cat((s, enc_output), dim=2)
            outputsource, outputenc = self.attention(DH,enc_output)
            outputsource=outputsource.view(-1,9,1)
            #outputsource=self.tktk.forward(outputsource)
            outputsource = F.softmax(outputsource, dim=1)
            outputsource=outputsource.to(torch.float32)
            outputenc=outputenc.to(torch.float32)
            C = torch.bmm(outputsource.transpose(1, 2), outputenc)
            out = torch.tanh(torch.cat((C, hidden_dec.transpose(0, 1)), dim=2))
            pred = self.ll(out)
            fedback = torch.cat((C, pred), dim=2)
            if t == 0:
                outs_dec = pred
            else:
                outs_dec = torch.cat([outs_dec, pred], dim=1)
            hidden = hidden_dec
            output = fedback
        outs_dec = outs_dec.squeeze(2)
        outs_dec = outs_dec.reshape(-1, self.node, self.n_pred)
        return outs_dec