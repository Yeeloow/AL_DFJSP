# agents.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from networks import Actor, Critic, DuelingD5QN
from config import *
import numpy as np
# --- 이 부분이 수정되었습니다 ---
# torch_geometric.data에서 Data와 Batch 클래스를 import 합니다.
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

class Agent1_SAC:
    def __init__(self, op_feature_dim, max_ops, gat_dim=GAT_OUT_DIM, n_heads=GAT_N_HEADS):
        self.graph_embedding_dim = 32
        self.max_ops = max_ops
        self.actor = Actor(op_feature_dim, gat_dim, n_heads, self.graph_embedding_dim).to(DEVICE)
        self.critic = Critic(self.graph_embedding_dim, self.max_ops).to(DEVICE)
        self.critic_target = Critic(self.graph_embedding_dim, self.max_ops).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=SAC_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=SAC_LR)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=SAC_LR)
        self.target_entropy = -np.log(1.0 / self.max_ops) * 0.98

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            if not isinstance(state, (Data, Batch)):
                raise TypeError(f"Expected Data or Batch object, but got {type(state)}")

            state_batch = state if isinstance(state, Batch) else Batch.from_data_list([state])
            state_batch = state_batch.to(DEVICE)
            
            if state_batch.x is None:
                raise ValueError("Actor received state with None features.")

            scores, _ = self.actor(state_batch.x, state_batch.edge_index, state_batch.batch)
            dense_scores, node_mask = to_dense_batch(scores, state_batch.batch, fill_value=-float('inf'))
            
            for i in range(state_batch.num_graphs):
                eligible_ops = state_batch[i].eligible_ops
                mask = torch.ones(dense_scores.size(1), device=DEVICE, dtype=torch.bool)
                if len(eligible_ops) > 0:
                    mask_indices = torch.tensor(eligible_ops, device=DEVICE)
                    if mask_indices.numel() > 0 and mask_indices.max() < mask.size(0):
                         mask[mask_indices] = False
                dense_scores[i, mask] = -float('inf')
            
            dense_scores = torch.nan_to_num(dense_scores, neginf=-1e9)
            
            probs = F.softmax(dense_scores, dim=1)
            probs = probs + 1e-8
            probs = probs / probs.sum(dim=-1, keepdim=True)

            dist = torch.distributions.Categorical(probs)
            
            action = torch.argmax(probs, dim=1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        if not isinstance(state, Batch) or state_batch.num_graphs == 1:
            return action.item(), log_prob.item(), entropy.item()
        
        return action, log_prob, entropy

    def update(self, replay_buffer, batch_size):
        sampled_data = replay_buffer.sample(batch_size)
        if sampled_data is None:  # sample()이 None을 반환하면 학습을 건너뜀
            return None
        states, actions, rewards, next_states, dones, idxs, is_weights = sampled_data

        state_batch = Batch.from_data_list(states).to(DEVICE)
        next_state_batch = Batch.from_data_list(next_states).to(DEVICE)
        
        op_actions = torch.tensor([a[0] for a in actions], device=DEVICE)
        action_batch_one_hot = F.one_hot(op_actions, num_classes=self.max_ops).float()
        reward_batch = torch.tensor(rewards, dtype=torch.float, device=DEVICE).unsqueeze(1)
        done_batch = torch.tensor(dones, dtype=torch.float, device=DEVICE).unsqueeze(1)

        with torch.no_grad():
            _, next_graph_embedding = self.actor(next_state_batch.x, next_state_batch.edge_index, next_state_batch.batch)
            next_actions_dist, next_log_pi, _ = self.select_action(next_state_batch)
            next_action_one_hot = F.one_hot(next_actions_dist, num_classes=self.max_ops).float()
            target_q1, target_q2 = self.critic_target(next_graph_embedding, next_action_one_hot)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi.unsqueeze(1)
            y = reward_batch + (1 - done_batch) * D5QN_GAMMA * target_q
        _, graph_embedding = self.actor(state_batch.x, state_batch.edge_index, state_batch.batch)
        current_q1, current_q2 = self.critic(graph_embedding, action_batch_one_hot)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        self.critic_optimizer.zero_grad(); critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        pred_actions, log_pi, entropy = self.select_action(state_batch)
        pred_action_one_hot = F.one_hot(pred_actions, num_classes=self.max_ops).float()
        q1_policy, q2_policy = self.critic(graph_embedding.detach(), pred_action_one_hot)
        min_q_policy = torch.min(q1_policy, q2_policy)
        actor_loss = (self.alpha * log_pi - min_q_policy.squeeze()).mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * SAC_TAU + target_param.data * (1.0 - SAC_TAU))
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha_value': self.alpha.item(),
            'policy_entropy': entropy.mean().item()
        }

class Agent2_D5QN:
    def __init__(self, machine_feature_dim, op_machine_pair_dim, **kwargs):
        # 네트워크 입력 차원은 이전과 동일 (기본 7개 + 통계 2개)
        self.q_network = DuelingD5QN(machine_feature_dim + 2, op_machine_pair_dim, **kwargs).to(DEVICE)
        self.target_network = DuelingD5QN(machine_feature_dim + 2, op_machine_pair_dim, **kwargs).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=D5QN_LR)
        self.update_counter = 0

    def _calculate_statistical_features(self, state, op_action_idx):
        earliest_times = []; num_slots = []
        op_proc_times = state.om_features[op_action_idx]
        prev_op_comp_time = state.eligible_ops_details[op_action_idx].item()
        
        for machine_idx in range(N_MACHINES):
            proc_time = op_proc_times[machine_idx][0].item()
            if proc_time == 0:
                earliest_times.append(float('inf')); num_slots.append(0)
                continue
                
            busy_intervals = state.op_histories[machine_idx] # op_histories는 이제 busy_intervals
            current_check_time = prev_op_comp_time
            available_slots_count = 0; earliest_fit_time = float('inf')
            
            if not busy_intervals:
                earliest_fit_time = current_check_time
                available_slots_count = 1
            else:
                # 첫 작업 시작 전
                if busy_intervals[0][0] - 0 >= proc_time:
                    start = max(0, current_check_time)
                    if start + proc_time <= busy_intervals[0][0]:
                        earliest_fit_time = min(earliest_fit_time, start)
                        available_slots_count += 1

                # 작업들 사이
                for i in range(len(busy_intervals) - 1):
                    idle_start, idle_end = busy_intervals[i][1], busy_intervals[i+1][0]
                    if idle_end - idle_start >= proc_time:
                        start = max(idle_start, current_check_time)
                        if start + proc_time <= idle_end:
                            earliest_fit_time = min(earliest_fit_time, start)
                            available_slots_count += 1
                
                # 마지막 작업 이후
                last_op_end = busy_intervals[-1][1]
                start = max(last_op_end, current_check_time)
                earliest_fit_time = min(earliest_fit_time, start)
                available_slots_count += 1

            earliest_times.append(earliest_fit_time); num_slots.append(available_slots_count)
        return torch.tensor([earliest_times, num_slots], dtype=torch.float, device=DEVICE).t()



    def select_action(self, state, selected_op_idx):
        with torch.no_grad():
            self.q_network.eval()
            stat_features = self._calculate_statistical_features(state, selected_op_idx)
            combined_m_features = torch.cat([state.m_features.to(DEVICE), stat_features], dim=-1).unsqueeze(0)
            op_machine_pairs = state.om_features[selected_op_idx].unsqueeze(0).to(DEVICE)
            q_values = self.q_network(combined_m_features, op_machine_pairs).squeeze(0)
            mask = (state.om_features[selected_op_idx].sum(axis=1) == 0)
            q_values[mask] = -float('inf')
            action = q_values.argmax().item()
            self.q_network.train()
        return action
        
    def update(self, replay_buffer, batch_size):
        sampled_data = replay_buffer.sample(batch_size)
        if sampled_data is None:  # sample()이 None을 반환하면 학습을 건너뜀
            return None
        states, actions, rewards, next_states, dones, idxs, is_weights = sampled_data

        state_batch = Batch.from_data_list(states).to(DEVICE)
        next_state_batch = Batch.from_data_list(next_states).to(DEVICE)
        
        op_indices_local = torch.tensor([a[0] for a in actions], device=DEVICE)
        action_batch = torch.tensor([a[1] for a in actions], dtype=torch.long, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float, device=DEVICE).unsqueeze(1)
        done_batch = torch.tensor(dones, dtype=torch.float, device=DEVICE).unsqueeze(1)
        is_weights_batch = is_weights.unsqueeze(1).to(DEVICE)

        stat_features_list = [self._calculate_statistical_features(s, a[0]) for s, a in zip(states, actions)]
        stat_features_batch = torch.stack(stat_features_list)
        combined_m_features = torch.cat([state_batch.m_features.view(batch_size, N_MACHINES, -1), stat_features_batch], dim=-1)

        next_op_indices_local = torch.tensor([s.eligible_ops[0] if len(s.eligible_ops) > 0 else 0 for s in next_states], device=DEVICE)
        next_stat_features_list = [self._calculate_statistical_features(ns, op_idx.item()) for ns, op_idx in zip(next_states, next_op_indices_local)]
        next_stat_features_batch = torch.stack(next_stat_features_list)
        next_combined_m_features = torch.cat([next_state_batch.m_features.view(batch_size, N_MACHINES, -1), next_stat_features_batch], dim=-1)
        
        op_indices_global = state_batch.ptr[:-1] + op_indices_local
        om_features_dim = 2 
        om_pairs_selected = state_batch.om_features[op_indices_global].view(batch_size, N_MACHINES, om_features_dim)

        next_op_indices_global = next_state_batch.ptr[:-1] + next_op_indices_local
        next_om_pairs_selected = next_state_batch.om_features[next_op_indices_global].view(batch_size, N_MACHINES, om_features_dim)

        with torch.no_grad():
            self.q_network.eval(); self.target_network.eval()
            next_q_values_current_net = self.q_network(next_combined_m_features, next_om_pairs_selected)
            best_next_actions = next_q_values_current_net.argmax(dim=1, keepdim=True)
            next_q_values_target_net = self.target_network(next_combined_m_features, next_om_pairs_selected)
            target_q_next = next_q_values_target_net.gather(1, best_next_actions)
            self.q_network.train(); self.target_network.train()
        
        y = reward_batch + (1 - done_batch) * D5QN_GAMMA * target_q_next
        
        current_q_values = self.q_network(combined_m_features, om_pairs_selected)
        current_q = current_q_values.gather(1, action_batch)
        td_errors = (y - current_q).abs()
        
        loss = (F.mse_loss(current_q, y, reduction='none') * is_weights_batch).mean()
        
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        replay_buffer.update_priorities(idxs, td_errors.squeeze().detach().cpu().numpy())
        
        self.update_counter += 1
        if self.update_counter % 200 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return {'d5qn_loss': loss.item(), 'avg_q_value': current_q.mean().item(), 'avg_td_error': td_errors.mean().item()}