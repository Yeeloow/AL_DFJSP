# networks.py (최종 수정본)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from config import *
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters(); self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in, epsilon_out = self._scale_noise(self.in_features), self._scale_noise(self.out_features)
        # .outer()는 numpy, .ger()이 pytorch의 외적(outer product)입니다.
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return F.linear(x, self.weight_mu, self.bias_mu)

class Actor(nn.Module):
    def __init__(self, in_dim, gat_dim, n_heads, graph_embedding_dim):
        super(Actor, self).__init__()
        self.encoder = GATConv(in_dim, gat_dim, heads=n_heads, dropout=GAT_DROPOUT)
        self.decoder = nn.Sequential(nn.Linear(gat_dim * n_heads + graph_embedding_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, op_features, edge_index, batch_index):
        # --- 디버깅용 print 추가 ---
        
        try:
            op_embedding = F.elu(self.encoder(op_features, edge_index))
            graph_embedding = global_mean_pool(op_embedding, batch_index)
            graph_embedding_expanded = graph_embedding[batch_index]
            combined_features = torch.cat([op_embedding, graph_embedding_expanded], dim=1)
            scores = self.decoder(combined_features).squeeze(-1)
            
            # --------------------------
            
            return scores, graph_embedding

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None # 예외 발생 시 명시적으로 None 반환 (오류 확인용)

class Critic(nn.Module):
    def __init__(self, graph_embedding_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(graph_embedding_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(graph_embedding_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, graph_embedding, action):
        sa = torch.cat([graph_embedding, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

class DuelingD5QN(nn.Module):
    def __init__(self, machine_feature_dim, op_machine_pair_dim, hidden_size, n_heads, n_layers):
        super(DuelingD5QN, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # --- 이 부분이 수정되었습니다 ---
        # 기계 특징 입력 차원이 2만큼 증가 (earliest_fit_time, num_available_slots)
        self.machine_feature_proj = nn.Linear(machine_feature_dim + 2, hidden_size)

        self.advantage = nn.Sequential(NoisyLinear(hidden_size + op_machine_pair_dim, 128), nn.ReLU(), NoisyLinear(128, 1))
        self.value = nn.Sequential(NoisyLinear(hidden_size + op_machine_pair_dim, 128), nn.ReLU(), NoisyLinear(128, 1))


    def forward(self, machine_features, op_machine_pairs):
        proj_machine_features = self.machine_feature_proj(machine_features)
        machine_embedding = self.transformer_encoder(proj_machine_features)
        combined_features = torch.cat([machine_embedding, op_machine_pairs], dim=-1)
        adv = self.advantage(combined_features)
        val = self.value(combined_features)
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values.squeeze(-1)
        
    def reset_noise(self):
        for name, module in self.named_children():
            if isinstance(module, nn.Sequential):
                for sub_module in module.children():
                    if hasattr(sub_module, 'reset_noise'):
                        sub_module.reset_noise()