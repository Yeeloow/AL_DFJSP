# agents.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from networks import Actor, Critic, DuelingD5QN
from config import *
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import logging # [추가됨] logging 모듈 임포트

# [추가됨] fjsp_environment.py에서 설정한 것과 동일한 이름으로 로거를 가져옵니다.
validation_logger = logging.getLogger('validation_logger')

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
        if sampled_data is None:
            return None
        states, actions, rewards, next_states, dones, idxs, is_weights = sampled_data

        # --- ▼▼▼ 수정된 부분 ▼▼▼ ---
        # Agent1이 처리할 수 없는 복잡한 객체를 Data에서 제거하는 클렌징 과정
        keys_to_remove = ['jobs', 'op_map_rev', 'op_histories', 'travel_times']
        
        clean_states = []
        for s in states:
            clean_s_dict = {k: v for k, v in s.to_dict().items() if k not in keys_to_remove}
            clean_states.append(Data(**clean_s_dict))

        clean_next_states = []
        for s in next_states:
            clean_s_dict = {k: v for k, v in s.to_dict().items() if k not in keys_to_remove}
            clean_next_states.append(Data(**clean_s_dict))
            
        # 정제된 데이터를 사용하여 배치 생성
        state_batch = Batch.from_data_list(clean_states).to(DEVICE)
        next_state_batch = Batch.from_data_list(clean_next_states).to(DEVICE)
        # --- ▲▲▲ 수정 끝 ▲▲▲ ---
        
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
    # --- ▼▼▼ 수정된 부분 ▼▼▼ ---
    # __init__ 메서드가 인자를 명시적으로 받도록 수정하여 중복 문제를 해결합니다.
    def __init__(self, machine_feature_dim, op_machine_pair_dim, **kwargs):
        # 이제 train.py에서 전달된 machine_feature_dim 값을 사용합니다.
        self.q_network = DuelingD5QN(
            machine_feature_dim=machine_feature_dim, 
            op_machine_pair_dim=op_machine_pair_dim, 
            **kwargs
        ).to(DEVICE)
        self.target_network = DuelingD5QN(
            machine_feature_dim=machine_feature_dim, 
            op_machine_pair_dim=op_machine_pair_dim, 
            **kwargs
        ).to(DEVICE)
    # --- ▲▲▲ 수정 끝 ▲▲▲ ---
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=D5QN_LR)
        self.update_counter = 0

    # --- ▼▼▼ 이 아래의 다른 메서드들은 변경 없습니다 ▼▼▼ ---
    def _find_first_fit_for_machine(self, busy_intervals, ready_time, proc_time):
        """ Agent 내부에서 대기 시간을 계산하기 위해 환경의 함수를 재현 """
        if not busy_intervals:
            return ready_time
        if ready_time + proc_time <= busy_intervals[0][0]:
            return ready_time
        for i in range(len(busy_intervals) - 1):
            idle_start = busy_intervals[i][1]
            idle_end = busy_intervals[i+1][0]
            potential_start = max(idle_start, ready_time)
            if potential_start + proc_time <= idle_end:
                return potential_start
        last_available_time = busy_intervals[-1][1] if busy_intervals else 0
        return max(last_available_time, ready_time)

    def _calculate_features_for_one_item(self, state, selected_op_idx):
        """ 단일 state 객체에 대해 새로운 5차원 머신 피쳐를 계산 """
        # ... (이하 _calculate_features_for_one_item 메서드 내용은 이전과 동일)
        jobs = state.jobs
        op_map_rev = state.op_map_rev
        job_id, op_id = op_map_rev[selected_op_idx]
        op = jobs[job_id].ops[op_id]
        
        predecessor_op = jobs[job_id].ops[op.id - 1] if op.id > 0 else None

        all_workloads = state.m_features[:, 0]
        avg_workload = torch.mean(all_workloads) if len(all_workloads) > 0 else 0.0

        wait_times = []
        for machine_idx in range(N_MACHINES):
            final_ready_time = 0.0
            if predecessor_op and predecessor_op.is_scheduled:
                travel_time = state.travel_times[predecessor_op.scheduled_machine][machine_idx].item()
                final_ready_time = predecessor_op.completion_time + travel_time
            
            processing_time = op.machine_map.get(machine_idx, float('inf'))
            if processing_time != float('inf'):
                busy_intervals = state.op_histories[machine_idx]
                actual_start_time = self._find_first_fit_for_machine(busy_intervals, final_ready_time, processing_time)
                wait_time = actual_start_time - final_ready_time
            else:
                wait_time = float('inf')
            wait_times.append(wait_time)

        valid_wait_times = [t for t in wait_times if t != float('inf')]
        max_wait_time = max(valid_wait_times) if valid_wait_times else 1.0
        max_workload = torch.max(all_workloads).item() if len(all_workloads) > 0 else 1.0
        max_proc_time = max(op.processing_times) if op.processing_times else 1.0
        
        new_features_list = []
        for machine_idx in range(N_MACHINES):
            workload = state.m_features[machine_idx, 0].item()
            num_candidates = state.m_features[machine_idx, 1].item()
            processing_time = op.machine_map.get(machine_idx, 0)

            wait_time_norm = wait_times[machine_idx] / (max_wait_time + 1e-8) if wait_times[machine_idx] != float('inf') else 1.0
            workload_norm = workload / (max_workload + 1e-8)
            workload_vs_avg_norm = (workload - avg_workload) / (max_workload + 1e-8)
            proc_time_norm = processing_time / (max_proc_time + 1e-8)
            num_candidates_norm = num_candidates / N_MACHINES
            
            new_features_list.append([
                wait_time_norm, workload_norm, workload_vs_avg_norm,
                proc_time_norm, num_candidates_norm
            ])
        
        return torch.tensor(new_features_list, dtype=torch.float, device=DEVICE)


    def select_action(self, state, selected_op_idx, log_q_values=False, current_step=None):
    # --- ▲▲▲ 수정 끝 ▲▲▲ ---
        with torch.no_grad():
            self.q_network.eval()
            new_m_features = self._calculate_features_for_one_item(state, selected_op_idx).unsqueeze(0)
            op_machine_pairs = state.om_features[selected_op_idx].unsqueeze(0).to(DEVICE)
            q_values = self.q_network(new_m_features, op_machine_pairs).squeeze(0)

            mask = (state.om_features[selected_op_idx].sum(axis=1) == 0)

            if log_q_values:
                eligible_machines = np.where(mask.cpu().numpy() == False)[0]
                q_values_str = np.round(q_values.detach().cpu().numpy(), 2)
                
                # --- ▼▼▼ 수정된 부분 (로그 메시지 형식 변경) ▼▼▼ ---
                log_message = (
                    f"[Step {current_step}] Q-Values for Op index: {selected_op_idx}\n"
                    f"  - Eligible Machines: {eligible_machines}\n"
                    f"  - Q-Values: {q_values_str}"
                )
                validation_logger.info(log_message)
                # --- ▲▲▲ 수정 끝 ▲▲▲ ---

            q_values[mask] = -float('inf')
            action = q_values.argmax().item()
            self.q_network.train()
        return action
        
    def update(self, replay_buffer, batch_size):
        sampled_data = replay_buffer.sample(batch_size)
        if sampled_data is None:
            return None
        states, actions, rewards, next_states, dones, idxs, is_weights = sampled_data
        
        op_indices = torch.tensor([a[0] for a in actions], device=DEVICE)
        action_batch = torch.tensor([a[1] for a in actions], dtype=torch.long, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float, device=DEVICE).unsqueeze(1)
        done_batch = torch.tensor(dones, dtype=torch.float, device=DEVICE).unsqueeze(1)
        is_weights_batch = torch.tensor(is_weights, dtype=torch.float, device=DEVICE).unsqueeze(1)

        # 배치 내 각 항목에 대해 실시간으로 특징 계산
        current_m_features_batch = []
        for i in range(batch_size):
            features = self._calculate_features_for_one_item(states[i], op_indices[i].item())
            current_m_features_batch.append(features)
        current_m_features_tensor = torch.stack(current_m_features_batch)

        next_m_features_batch = []
        next_op_indices = torch.tensor([s.eligible_ops[0] if len(s.eligible_ops) > 0 else 0 for s in next_states], device=DEVICE)
        for i in range(batch_size):
            features = self._calculate_features_for_one_item(next_states[i], next_op_indices[i].item())
            next_m_features_batch.append(features)
        next_m_features_tensor = torch.stack(next_m_features_batch)

        # om_features 또한 배치에 맞게 재구성
        om_pairs_list = [states[i].om_features[op_indices[i]] for i in range(batch_size)]
        om_pairs_tensor = torch.stack(om_pairs_list).to(DEVICE)

        next_om_pairs_list = [next_states[i].om_features[next_op_indices[i]] for i in range(batch_size)]
        next_om_pairs_tensor = torch.stack(next_om_pairs_list).to(DEVICE)

        with torch.no_grad():
            self.q_network.eval(); self.target_network.eval()
            next_q_values_current_net = self.q_network(next_m_features_tensor, next_om_pairs_tensor)
            best_next_actions = next_q_values_current_net.argmax(dim=1, keepdim=True)
            next_q_values_target_net = self.target_network(next_m_features_tensor, next_om_pairs_tensor)
            target_q_next = next_q_values_target_net.gather(1, best_next_actions)
            self.q_network.train(); self.target_network.train()
        
        y = reward_batch + (1 - done_batch) * D5QN_GAMMA * target_q_next
        
        current_q_values = self.q_network(current_m_features_tensor, om_pairs_tensor)
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