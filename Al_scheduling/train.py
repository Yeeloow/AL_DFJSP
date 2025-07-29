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

warnings.filterwarnings("ignore", message="gemm_and_bias error:.*")

def validate(agent1, agent2, validation_instances, max_ops):
    total_makespan = 0
    agent1.actor.eval(); agent2.q_network.eval()
    with torch.no_grad():
        for instance_data_list in validation_instances:
            env = FJSPEnv(); env.instance_jobs_data = instance_data_list
            state_dict = env.reset(); done = False
            while not done:
                state_data = Data(**state_dict)
                op_action, _, _ = agent1.select_action(state_data, deterministic=True)
                machine_action = agent2.select_action(state_data, op_action)
                state_dict, _, done = env.step((op_action, machine_action))
            total_makespan += env._calculate_makespan()
    agent1.actor.train(); agent2.q_network.train()
    return total_makespan / len(validation_instances)

def main():
    writer = SummaryWriter()
    envs = [FJSPEnv(n_jobs=N_JOBS, n_machines=N_MACHINES) for _ in range(BATCH_SIZE)]
    validation_instances = [FJSPEnv(n_jobs=N_JOBS, n_machines=N_MACHINES).instance_jobs_data for _ in range(100)]
    max_ops_in_dataset = max(sum(len(job_data) for env in envs for job_data in env.instance_jobs_data), MAX_OPS_PER_JOB * N_JOBS)

    agent1 = Agent1_SAC(op_feature_dim=10, max_ops=max_ops_in_dataset)
    agent2 = Agent2_D5QN(
        machine_feature_dim=7, op_machine_pair_dim=2,
        hidden_size=TRANSFORMER_HIDDEN_SIZE, n_heads=TRANSFORMER_N_HEADS, n_layers=TRANSFORMER_N_LAYERS
    )
    
    sac_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)
    d5qn_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)
    current_states = [env.reset() for env in envs]
    dones = [False] * BATCH_SIZE
    log_data = {'sac': [], 'd5qn': []}; episode_makespans = []
    best_val_makespan = float('inf')
    
    print("훈련을 시작합니다 (단일 프로세스, 안정화 모드)...")
    for t in tqdm(range(1, 200001)):
        for i in range(BATCH_SIZE):
            if dones[i]:
                if envs[i]._calculate_makespan() > 0: episode_makespans.append(envs[i]._calculate_makespan())
                current_states[i] = envs[i].reset()
                agent2.q_network.reset_noise(); agent2.target_network.reset_noise()

            current_state_dict = current_states[i]
            state_data = Data(**current_state_dict)

            op_action, _, _ = agent1.select_action(state_data)
            machine_action = agent2.select_action(state_data, op_action)
            action = (op_action, machine_action)
            
            next_state_dict, reward, done = envs[i].step(action)
            next_state_data = Data(**next_state_dict)

            sac_buffer.push(state_data, action, reward, next_state_data, done)
            d5qn_buffer.push(state_data, action, reward, next_state_data, done)
            
            current_states[i] = next_state_dict
            dones[i] = done
            
        if len(d5qn_buffer) > BATCH_SIZE * 10:
            sac_loss_dict = agent1.update(sac_buffer, BATCH_SIZE)
            d5qn_loss_dict = agent2.update(d5qn_buffer, BATCH_SIZE)
            if sac_loss_dict: log_data['sac'].append(sac_loss_dict)
            if d5qn_loss_dict: log_data['d5qn'].append(d5qn_loss_dict)

        if t % 5000 == 0 and t > 0:
            avg_val_makespan = validate(agent1, agent2, validation_instances, max_ops_in_dataset)
            avg_train_makespan = np.mean(episode_makespans) if episode_makespans else 0
            # 1. 평균 손실 및 Q-가치 계산
            avg_sac_losses = {k: np.mean([d[k] for d in log_data['sac']]) for k in log_data['sac'][0]} if log_data['sac'] else {}
            avg_d5qn_losses = {k: np.mean([d[k] for d in log_data['d5qn']]) for k in log_data['d5qn'][0]} if log_data['d5qn'] else {}

            # 2. 텐서보드에 로그 기록
            writer.add_scalar('Makespan/Train', avg_train_makespan, t)
            writer.add_scalar('Makespan/Validation', avg_val_makespan, t)
            for k, v in avg_sac_losses.items():
                writer.add_scalar(f'SAC/{k}', v, t)
            for k, v in avg_d5qn_losses.items():
                writer.add_scalar(f'D5QN/{k}', v, t)

            # 3. 콘솔에 진행 상황 출력
            print(f"\n[Step {t}] Train Makespan: {avg_train_makespan:.2f} | Validation Makespan: {avg_val_makespan:.2f}")
            print(f"  SAC Losses: {avg_sac_losses}")
            print(f"  D5QN Losses: {avg_d5qn_losses}")

            # 4. 최고의 모델 저장
            if avg_val_makespan < best_val_makespan:
                best_val_makespan = avg_val_makespan
                torch.save(agent1.actor.state_dict(), 'best_actor_model.pth')
                torch.save(agent2.q_network.state_dict(), 'best_q_network_model.pth')
                print(f"*** 새로운 최고 성능 모델 저장! (Validation Makespan: {best_val_makespan:.2f}) ***")

            # 5. 다음 로깅을 위해 로그 데이터 초기화
            log_data = {'sac': [], 'd5qn': []}
            episode_makespans = []
            
    writer.close()

if __name__ == "__main__":
    main()