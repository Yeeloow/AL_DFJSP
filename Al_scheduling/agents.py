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
        """ 
        [전면 개편] 새로운 5차원 머신 특징을 계산합니다.
        1. 장비 가용 시간 (Time to Availability)
        2. 이동 시간 (Travel Time)
        3. 처리 시간 (Processing Time)
        4. 장비별 작업 완료율 (Work Completion Rate)
        5. 대기 시간 (Wait Time)
        """
        num_machines = state.m_features.shape[0]
        jobs = state.jobs
        op_map_rev = state.op_map_rev
        job_id, op_id = op_map_rev[selected_op_idx]
        op = jobs[job_id].ops[op_id]
        
        predecessor_op = jobs[job_id].ops[op.id - 1] if op.id > 0 else None
        
        # 각 머신별로 특징의 원본 값을 계산합니다.
        time_to_availability_list = []
        travel_times_list = []
        processing_times_list = []
        wait_times_list = []

        current_time_approx = predecessor_op.completion_time if (predecessor_op and predecessor_op.is_scheduled) else 0

        for m_idx in range(num_machines):
            # 특징 1: 장비 가용 시간
            avail_time = state.machine_availability_times[m_idx].item()
            time_to_availability = max(0, avail_time - current_time_approx)
            time_to_availability_list.append(time_to_availability)

            # 특징 2: 이동 시간
            travel_time = 0.0
            if predecessor_op and predecessor_op.is_scheduled:
                if predecessor_op.scheduled_machine != m_idx:
                    travel_time = state.travel_times[predecessor_op.scheduled_machine][m_idx].item()
            travel_times_list.append(travel_time)

            # 특징 3 & 5: 처리 시간 및 대기 시간
            proc_time = op.machine_map.get(m_idx, float('inf'))
            processing_times_list.append(proc_time if proc_time != float('inf') else 0)

            if proc_time != float('inf'):
                ready_time = current_time_approx + travel_time
                busy_intervals = state.op_histories[m_idx]
                actual_start_time = self._find_first_fit_for_machine(busy_intervals, ready_time, proc_time)
                wait_time = actual_start_time - ready_time
                wait_times_list.append(wait_time)
            else:
                wait_times_list.append(float('inf'))

        # 특징 4: 장비별 작업 완료율 (이 특징은 이미 [0, 1] 범위이므로 추가 정규화 불필요)
        work_completion_rate = state.completed_ops_per_machine / (state.total_potential_ops_per_machine + 1e-8)
        
        # 정규화를 위한 최대값 계산 (Global-Fixed Maximum 방식 적용)
        max_time_to_avail = max(time_to_availability_list or [1.0])
        max_travel_time = torch.max(state.travel_times).item() if state.travel_times.numel() > 0 else 1.0
        max_proc_time = max(op.processing_times or [1.0])
        max_wait_time = max([t for t in wait_times_list if t != float('inf')] or [1.0])

        # 최종 5차원 특징 벡터 생성
        new_features_list = []
        for m_idx in range(num_machines):
            # 각 특징을 [0, 1] 사이 값으로 정규화
            avail_norm = time_to_availability_list[m_idx] / (max_time_to_avail + 1e-8)
            travel_norm = travel_times_list[m_idx] / (max_travel_time + 1e-8)
            proc_norm = processing_times_list[m_idx] / (max_proc_time + 1e-8)
            wait_norm = wait_times_list[m_idx] / (max_wait_time + 1e-8) if wait_times_list[m_idx] != float('inf') else 1.0

            # 5가지 특징을 최종 벡터로 조합
            new_features_list.append([
                avail_norm,
                travel_norm,
                proc_norm,
                work_completion_rate[m_idx].item(),
                wait_norm
            ])
        
        return torch.tensor(new_features_list, dtype=torch.float, device=DEVICE)


    def select_action(self, state, selected_op_idx, log_q_values=False, current_step=None):
        with torch.no_grad():
            self.q_network.eval()
            new_m_features = self._calculate_features_for_one_item(state, selected_op_idx).unsqueeze(0)
            op_machine_pairs = state.om_features[selected_op_idx].unsqueeze(0).to(DEVICE)
            q_values = self.q_network(new_m_features, op_machine_pairs).squeeze(0)

            # --- ▼▼▼ [수정됨] 마스킹 로직 전체 수정 ▼▼▼ ---

            # 1. Eligible machines 정보를 먼저 가져옵니다.
            job_id, op_id = state.op_map_rev[selected_op_idx]
            op = state.jobs[job_id].ops[op_id]
            eligible_machines = [int(m) for m in op.eligible_machines]

            # 2. 올바른 마스크를 생성합니다: 모든 장비를 True(마스킹)로 초기화하고,
            #    eligible_machines에 해당하는 장비만 False(마스킹 해제)로 변경합니다.
            num_machines = state.m_features.shape[0]
            mask = torch.ones(num_machines, dtype=torch.bool, device=DEVICE)
            if eligible_machines:
                mask[eligible_machines] = False

            # 3. 마스킹 *전* Q-Value를 로깅합니다 (기존과 동일).
            if log_q_values:
                q_values_str = np.round(q_values.detach().cpu().numpy(), 2)
                log_message = (
                    f"[Step {current_step}] Q-Values for Op index: {selected_op_idx}\n"
                    f"  - Eligible Machines: {eligible_machines}\n"
                    f"  - Q-Values: {q_values_str}"
                )
                validation_logger.info(log_message)

            # 4. 올바른 마스크를 적용합니다.
            q_values[mask] = -float('inf')
            action = q_values.argmax().item()

            # 5. [추가] 선택된 action이 올바른지 확인하고, 아닐 경우 로그를 남깁니다.
            if action not in eligible_machines:
                wrong_choice_log = (
                    f"[Step {current_step}] Wrong machine choice! "
                    f"Selected M{action}, but eligible are {eligible_machines}."
                )
                validation_logger.info(wrong_choice_log)
            # --- ▲▲▲ 수정 끝 ▲▲▲ ---
            
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
        is_weights_batch = is_weights.to(DEVICE).unsqueeze(1)

        current_m_features_batch = []
        for i in range(batch_size):
            # next_state의 eligible_ops가 비어있을 경우를 대비한 예외 처리
            if states[i].num_nodes > 0 and op_indices[i].item() < states[i].num_nodes:
                 features = self._calculate_features_for_one_item(states[i], op_indices[i].item())
                 current_m_features_batch.append(features)
            else: # Fallback: 만약 op_indices가 유효하지 않으면, 0 벡터 추가
                 current_m_features_batch.append(torch.zeros((N_MACHINES, 5), device=DEVICE))

        current_m_features_tensor = torch.stack(current_m_features_batch)

        next_m_features_batch = []
        next_op_indices = torch.tensor([s.eligible_ops[0] if (hasattr(s, 'eligible_ops') and len(s.eligible_ops) > 0) else 0 for s in next_states], device=DEVICE)
        
        for i in range(batch_size):
            if next_states[i].num_nodes > 0 and next_op_indices[i].item() < next_states[i].num_nodes:
                features = self._calculate_features_for_one_item(next_states[i], next_op_indices[i].item())
                next_m_features_batch.append(features)
            else: # Fallback
                next_m_features_batch.append(torch.zeros((N_MACHINES, 5), device=DEVICE))

        next_m_features_tensor = torch.stack(next_m_features_batch)

        om_pairs_list = [states[i].om_features[op_indices[i]] if op_indices[i] < len(states[i].om_features) else torch.zeros(N_MACHINES, 2) for i in range(batch_size)]
        om_pairs_tensor = torch.stack(om_pairs_list).to(DEVICE)

        next_om_pairs_list = [next_states[i].om_features[next_op_indices[i]] if next_op_indices[i] < len(next_states[i].om_features) else torch.zeros(N_MACHINES, 2) for i in range(batch_size)]
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
        
        # --- ▼▼▼ 수정된 부분 (손실 함수 변경) ▼▼▼ ---
        # MSE Loss 대신 Huber Loss(Smooth L1 Loss)를 사용하여 그라디언트 폭발 방지
        loss = (F.smooth_l1_loss(current_q, y, reduction='none') * is_weights_batch).mean()
        # --- ▲▲▲ 수정 끝 ▲▲▲ ---
        
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        replay_buffer.update_priorities(idxs, td_errors.squeeze().detach().cpu().numpy())
        
        self.update_counter += 1
        # --- ▼▼▼ 수정된 부분 (업데이트 주기 변경) ▼▼▼ ---
        # 타겟 네트워크 업데이트 주기를 200에서 50으로 단축
        if self.update_counter % 50 == 0:
        # --- ▲▲▲ 수정 끝 ▲▲▲ ---
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return {'d5qn_loss': loss.item(), 'avg_q_value': current_q.mean().item(), 'avg_td_error': td_errors.mean().item()}
