# train.py

import torch
from tqdm import tqdm
from config import *
from fjsp_environment import FJSPEnv
from agents import Agent1_SAC, Agent2_D5QN
from replay_buffer import PrioritizedReplayBuffer
import numpy as np
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
import warnings
import logging
import os
from datetime import datetime

warnings.filterwarnings("ignore", message="gemm_and_bias error:.*")

# --- 로거 설정 함수 ---
def setup_logger():
    """로그 파일과 콘솔 출력을 동시에 처리하는 로거를 설정합니다."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = datetime.now().strftime('train_%Y%m%d_%H%M%S.log')
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger('FJSP_Logger')
    logger.setLevel(logging.INFO)

    # 핸들러가 이미 추가되었는지 확인하여 중복 추가 방지
    if not logger.handlers:
        # 파일 핸들러: 로그를 파일에 저장
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)

        # 스트림 핸들러: 로그를 콘솔에 출력
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s')) # 콘솔에는 메시지만 깔끔하게 출력
        logger.addHandler(stream_handler)
        
    return logger

def validate(agent1, agent2, validation_instances, max_ops):
    """
    별도의 검증 데이터셋으로 현재 에이전트들의 성능을 순차적으로 평가합니다.
    """
    total_makespan = 0
    agent1.actor.eval(); agent2.q_network.eval()

    with torch.no_grad():
        for instance_data_list in validation_instances:
            env = FJSPEnv()
            env.instance_jobs_data = instance_data_list
            state_dict = env.reset()
            done = False
            
            while not done:
                state_data = Data(**state_dict)
                op_action, _, _ = agent1.select_action(state_data, deterministic=True)
                machine_action = agent2.select_action(state_data, op_action)
                action = (op_action, machine_action)
                state_dict, _, done = env.step(action)
                
            total_makespan += env._calculate_makespan()
    
    agent1.actor.train(); agent2.q_network.train()
    return total_makespan / len(validation_instances)


def main():
    logger = setup_logger()
    writer = SummaryWriter()

    envs = [FJSPEnv(n_jobs=N_JOBS, n_machines=N_MACHINES) for _ in range(BATCH_SIZE)]
    validation_instances = [FJSPEnv(n_jobs=N_JOBS, n_machines=N_MACHINES).instance_jobs_data for _ in range(100)]
    
    max_ops_in_dataset = max(sum(len(job_data) for job_data in env.instance_jobs_data) for env in envs)

    agent1 = Agent1_SAC(op_feature_dim=10, max_ops=max_ops_in_dataset)
    agent2 = Agent2_D5QN(
        machine_feature_dim=7, 
        op_machine_pair_dim=2,
        hidden_size=TRANSFORMER_HIDDEN_SIZE,
        n_heads=TRANSFORMER_N_HEADS,
        n_layers=TRANSFORMER_N_LAYERS
    )
    
    sac_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)
    d5qn_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)

    current_states = [env.reset() for env in envs]
    dones = [False] * BATCH_SIZE
    
    episode_makespans = []
    log_data = {
        'sac_critic': [], 'sac_actor': [], 'sac_alpha': [], 
        'd5qn': [], 'q_values': [], 'td_errors': [], 
        'alpha_values': [], 'policy_entropies': []
    }
    
    best_val_makespan = float('inf')
    patience = 10
    steps_since_last_improvement = 0
    
    logger.info("훈련을 시작합니다...")
    for t in tqdm(range(1, 200001)):
        for i in range(BATCH_SIZE):
            if dones[i]:
                if envs[i]._calculate_makespan() > 0:
                    episode_makespans.append(envs[i]._calculate_makespan())
                current_states[i] = envs[i].reset()
                agent2.q_network.reset_noise()
                agent2.target_network.reset_noise()

            current_state_dict = current_states[i]
            
            state_data = Data(
                x=current_state_dict['x'],
                edge_index=current_state_dict['edge_index'],
                eligible_ops=current_state_dict['eligible_ops'],
                eligible_ops_details=current_state_dict.get('eligible_ops_details', {}), # .get()으로 안정성 확보
                m_features=current_state_dict['m_features'],
                om_features=current_state_dict['om_features'],
                num_nodes=current_state_dict['num_nodes'],
                op_histories=current_state_dict.get('op_histories', []) # .get()으로 안정성 확보
            )

            op_action, _, _ = agent1.select_action(state_data)
            machine_action = agent2.select_action(state_data, op_action)
            action = (op_action, machine_action)
            
            prev_state_data = state_data
            next_state_dict, reward, done = envs[i].step(action)
            
            next_state_data = Data(
                x=next_state_dict['x'],
                edge_index=next_state_dict['edge_index'],
                eligible_ops=next_state_dict['eligible_ops'],
                eligible_ops_details=next_state_dict.get('eligible_ops_details', {}),
                m_features=next_state_dict['m_features'],
                om_features=next_state_dict['om_features'],
                num_nodes=next_state_dict['num_nodes'],
                op_histories=next_state_dict.get('op_histories', [])
            )

            sac_buffer.push(prev_state_data, action, reward, next_state_data, done)
            d5qn_buffer.push(prev_state_data, action, reward, next_state_data, done)
            
            current_states[i] = next_state_dict
            dones[i] = done
            
        if len(d5qn_buffer) > BATCH_SIZE * 10:
            sac_metrics = agent1.update(sac_buffer, BATCH_SIZE)
            d5qn_metrics = agent2.update(d5qn_buffer, BATCH_SIZE)
            
            if sac_metrics:
                log_data['sac_critic'].append(sac_metrics['critic_loss'])
                log_data['sac_actor'].append(sac_metrics['actor_loss'])
                log_data['sac_alpha'].append(sac_metrics['alpha_loss'])
                log_data['alpha_values'].append(sac_metrics['alpha_value'])
                log_data['policy_entropies'].append(sac_metrics['policy_entropy'])
            if d5qn_metrics:
                log_data['d5qn'].append(d5qn_metrics['d5qn_loss'])
                log_data['q_values'].append(d5qn_metrics['avg_q_value'])
                log_data['td_errors'].append(d5qn_metrics['avg_td_error'])

        if t % 500 == 0 and t > 0:
            avg_val_makespan = validate(agent1, agent2, validation_instances, max_ops_in_dataset)
            
            avg_train_makespan = np.mean(episode_makespans) if episode_makespans else 0
            
            avg_d5qn_loss = np.mean(log_data['d5qn']) if log_data['d5qn'] else 0
            avg_q_value = np.mean(log_data['q_values']) if log_data['q_values'] else 0
            avg_td_error = np.mean(log_data['td_errors']) if log_data['td_errors'] else 0
            
            avg_critic_loss = np.mean(log_data['sac_critic']) if log_data['sac_critic'] else 0
            avg_actor_loss = np.mean(log_data['sac_actor']) if log_data['sac_actor'] else 0
            avg_alpha_value = np.mean(log_data['alpha_values']) if log_data['alpha_values'] else 0
            avg_policy_entropy = np.mean(log_data['policy_entropies']) if log_data['policy_entropies'] else 0

            logger.info(f"\n스텝 {t}: 평균 훈련 Makespan: {avg_train_makespan:.2f}, 평균 검증 Makespan: {avg_val_makespan:.2f}")
            logger.info(f"    D5QN (Loss: {avg_d5qn_loss:.4f}, Q-Value: {avg_q_value:.2f}, TD-Error: {avg_td_error:.2f})")
            logger.info(f"    SAC (Critic Loss: {avg_critic_loss:.4f}, Actor Loss: {avg_actor_loss:.4f}, Alpha: {avg_alpha_value:.4f}, Entropy: {avg_policy_entropy:.4f})")
            
            writer.add_scalar('Makespan/train', avg_train_makespan, t)
            writer.add_scalar('Makespan/validation', avg_val_makespan, t)
            writer.add_scalar('Loss/D5QN', avg_d5qn_loss, t)
            writer.add_scalar('Loss/SAC_Critic', avg_critic_loss, t)
            writer.add_scalar('Loss/SAC_Actor', avg_actor_loss, t)
            writer.add_scalar('Q-Values/D5QN_Avg_Q', avg_q_value, t)
            writer.add_scalar('TD-Error/D5QN_Avg_TD', avg_td_error, t)
            writer.add_scalar('SAC/Alpha', avg_alpha_value, t)
            writer.add_scalar('SAC/Policy_Entropy', avg_policy_entropy, t)
            
            for key in log_data: log_data[key].clear()
            episode_makespans.clear()

            if avg_val_makespan < best_val_makespan:
                best_val_makespan = avg_val_makespan
                steps_since_last_improvement = 0
                torch.save(agent1.actor.state_dict(), 'agent1_actor_best.pth')
                torch.save(agent2.q_network.state_dict(), 'agent2_d5qn_best.pth')
                logger.info(f"*** 새로운 최고 성능 달성! 모델 가중치를 저장합니다. (Makespan: {best_val_makespan:.2f}) ***")
            else:
                steps_since_last_improvement += 1
                logger.info(f"성능 개선 없음. (카운터: {steps_since_last_improvement}/{patience})")

            if steps_since_last_improvement >= patience:
                logger.info(f"\n!!!!! {patience}번의 검증 주기 동안 성능 개선이 없어 훈련을 조기 종료합니다. !!!!!")
                break
    
    writer.close()

if __name__ == "__main__":
    main()