import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from torch import nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        self.hidden_dim = 64
        self.bias = True
        self.path = "PEMS08_adj.csv"
        self.Dinput_size = 1  # 特征的个数
        self.Dhidden_size = 64
        self.n_pred = 12
        self.node = 170

class GCNGRUCell(nn.Module):
    def __init__(self, config):
        super(GCNGRUCell, self).__init__()
        self.GCN1 = GCN(config.hidden_dim, config.hidden_dim * 2, config.path, bias=0.4)
        self.GCN2 = GCN(config.hidden_dim, config.hidden_dim, config.path, bias=0.6)

    def forward(self, input_tensor, hidden):
        batch, node = input_tensor.shape
        hidden = hidden.reshape(batch, -1)
        combined_gcn = torch.sigmoid(self.GCN1(input_tensor, hidden))
        r, u = torch.chunk(combined_gcn, chunks=2, dim=1)
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
        return self.cell_list.init_hidden(batch_size, node, hidden)

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, adj_path, bias):
        super(GCN, self).__init__()
        self.adj = self.load_adj(adj_path)
        self._num_gru_units = input_size
        self._output_dim = hidden_size
        self._bias_init_value = bias
        self.weights = nn.Parameter(torch.FloatTensor(self._num_gru_units + 1, self._output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        graph_data = self.adj.to(device)
        graph_data = self.process_graph(graph_data)
        inputs = inputs.unsqueeze(2)
        hidden_state = hidden_state.reshape(batch_size, num_nodes, self._num_gru_units)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        concatenation = concatenation.reshape((num_nodes, (self._num_gru_units + 1) * batch_size))
        a_times_concat = graph_data @ concatenation
        a_times_concat = a_times_concat.reshape((num_nodes, self._num_gru_units + 1, batch_size))
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        a_times_concat = a_times_concat.reshape((batch_size * num_nodes, self._num_gru_units + 1))
        outputs = a_times_concat @ self.weights + self.biases
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
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
        return graph_data.matmul(degree_matrix).transpose(0, 1).matmul(degree_matrix)

    def load_adj(self, adj_path):
        df = pd.read_csv(adj_path, encoding='utf-8', header=None)
        adj = np.array(df, dtype=np.float32)
        return torch.from_numpy(adj)

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.attn = nn.Linear(config.hidden_dim + config.Dhidden_size, config.Dhidden_size, bias=False)
        self.v = nn.Linear(config.Dhidden_size, 1, bias=False)
        self.node = config.node

    def forward(self, kfc, enc_output):
        energy = torch.tanh(self.attn(kfc))
        attention = self.v(energy)
        return attention, enc_output

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.decoder = nn.GRU(int(config.Dhidden_size + 1), config.Dhidden_size, batch_first=True)
        self.linear = nn.Linear(config.Dhidden_size, 1)
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
        return attended_values + input_data

class AttentionMode(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMode, self).__init__()
        self.attention_layer = nn.Linear(input_dim, 64)
        self.softmax = nn.Softmax(dim=0)
        self.layer2 = nn.Sequential(
            nn.Linear(64, 64 // 8),
            nn.GELU(),
            nn.Linear(64 // 8, 1),
        )

    def forward(self, inputs):
        return self.layer2(inputs)

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
        self.att = AttentionMode(input_dim=64)

    def forward(self, train, target, teacher_forcing_ratio=0.5):
        target = target.transpose(1, 2)
        output_enc, hidden = self.enc(train)
        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(1, hidden.shape[1], int(hidden.shape[2] / self.hidden), -1)
        hidden = hidden.reshape(1, hidden.shape[1] * hidden.shape[2], -1)
        output = torch.zeros_like(target[:, 0:1, :])
        output = output.transpose(1, 2)
        output = output.reshape(output.shape[0] * output.shape[1], -1).unsqueeze(2)
        output = output.repeat(1, 1, 1 * self.hidden + 1)
        output__ = output_enc
        for t in range(self.n_pred):
            if t == 0:
                input = output
            else:
                teacher_force = random.random() < teacher_forcing_ratio
                input = (target[:, t - 1:t, :] if teacher_force else output)
            output_enc = output__
            output, hidden_dec = self.dec(input, hidden)
            end_hidden = hidden_dec
            enc_output = output_enc
            enc_output = enc_output.view(12, -1, 64)
            attention_weights = self.att(enc_output)
            result = torch.sum(attention_weights, dim=(1, 2)).view(12, 1)
            topk_values, topk_indices = torch.topk(result.view(-1), 9)
            enc_output = enc_output[topk_indices].view(-1, 9, 64)
            batch2, seq_len2, _ = enc_output.shape
            _, _, hidden2 = end_hidden.shape
            enc_output = enc_output.reshape(batch2, seq_len2, hidden2, -1)
            enc_output = enc_output.permute(0, 3, 1, 2)
            enc_output = enc_output.reshape(-1, seq_len2, hidden2)
            s = end_hidden.repeat(seq_len2, 1, 1)
            s = s.transpose(0, 1)
            DH = torch.cat((s, enc_output), dim=2)
            output_source, output_enc = self.attention(DH, enc_output)
            output_source = output_source.view(-1, 9, 1)
            output_source = F.softmax(output_source, dim=1)
            output_source = output_source.to(torch.float32)
            output_enc = output_enc.to(torch.float32)
            C = torch.bmm(output_source.transpose(1, 2), output_enc)
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

def test_model():
    config = Config()
    model = RAGCRN(config).to(device)
    batch_size = 64
    seq_len = 12
    node = config.node
    train_data = torch.randn(batch_size, node, seq_len).to(device)
    target_data = torch.randn(batch_size, node, config.n_pred).to(device)
    output = model(train_data, target_data, 0)  
    
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test_model()