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
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_gantt_chart(schedule_data, n_jobs, n_machines, filename):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # --- 이 부분이 수정되었습니다 ---
    cmap = plt.colormaps.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, n_jobs)]

    for m_id, tasks in schedule_data.items():
        for task in tasks:
            job_id, op_id, start, end = task
            duration = end - start
            
            # job_id에 맞는 색상을 리스트에서 선택합니다.
            color = colors[job_id % n_jobs] if job_id != -1 else 'gray'
            ax.broken_barh([(start, duration)], (m_id * 10, 9), facecolors=color, edgecolor='black')
            text = f'J{job_id}-O{op_id}' if job_id != -1 else 'Pre'
            ax.text(start + duration / 2, m_id * 10 + 4.5, text, ha='center', va='center', color='white', fontweight='bold')

    ax.set_xlabel('Time'); ax.set_ylabel('Machines'); ax.set_title('Gantt Chart for Validation Instance')
    ax.set_yticks([i * 10 + 4.5 for i in range(n_machines)])
    ax.set_yticklabels([f'Machine {i}' for i in range(n_machines)])
    ax.grid(True, axis='x', linestyle=':')
    legend_elements = [Patch(facecolor=colors[i % n_jobs], label=f'Job {i}') for i in range(n_jobs)]
    legend_elements.append(Patch(facecolor='gray', label='Pre-occupied'))
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.savefig(filename); plt.close(fig)
    print(f"간트 차트가 '{filename}' 파일로 저장되었습니다.")

def plot_eligibility_table(jobs, filename):
    row_labels = []; cell_text = []
    for job in jobs:
        for op in job.ops:
            row_labels.append(f'Job {job.id} - Op {op.id}')
            machines_str = ', '.join(map(str, op.eligible_machines))
            cell_text.append([machines_str])
    if not cell_text: return
    fig_height = max(4, len(row_labels) * 0.4)
    fig, ax = plt.subplots(figsize=(6, fig_height))
    ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=['Eligible Machines'], loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
    ax.set_title('Job-Machine Eligibility Table', fontweight="bold", y=0.9)
    fig.tight_layout(); plt.savefig(filename, bbox_inches='tight'); plt.close(fig)
    print(f"장비 할당 테이블이 '{filename}' 파일로 저장되었습니다.")

VALIDATION_LOG_FILE = 'validation_log.csv'
warnings.filterwarnings("ignore", message="gemm_and_bias error:.*")

def generate_pre_occupied_schedule(n_machines):
    schedule = {}
    for m_id in range(n_machines):
        if np.random.rand() < 0.5:
            machine_intervals = []
            num_intervals = np.random.randint(1, 4)
            last_end_time = 0
            for _ in range(num_intervals):
                gap = np.random.randint(0, 51)
                start_time = last_end_time + gap
                duration = np.random.randint(10, 30)
                end_time = start_time + duration
                machine_intervals.append((-1, -1, start_time, end_time))
                last_end_time = end_time
            schedule[m_id] = machine_intervals
    return schedule

def validate(agent1, agent2, validation_instances, max_ops, current_step):
    print(f"\n--- [Validation Step {current_step}] Analyzing First Instance ONLY ---")

    # --- 1. 오직 첫 번째 검증 인스턴스만 사용 ---
    first_instance_data = validation_instances[0]
    
    env = FJSPEnv()
    env.instance_jobs_data = first_instance_data
    
    # --- 2. 해당 인스턴스의 자격 테이블을 먼저 생성 ---
    # reset을 먼저 호출하여 env.jobs 객체를 생성해야 함
    pre_occupied_schedule = generate_pre_occupied_schedule(env.n_machines)
    state_dict = env.reset(initial_schedule=pre_occupied_schedule)
    table_filename = f'validation_eligibility_table_step_{current_step}.png'
    plot_eligibility_table(env.jobs, table_filename)

    # --- 3. 해당 인스턴스에 대한 시뮬레이션 실행 ---
    step_in_episode = 0
    done = False
    
    agent1.actor.eval()
    agent2.q_network.eval()

    with torch.no_grad():
        while not done:
            state_data = Data(**state_dict)
            op_action, _, _ = agent1.select_action(state_data, deterministic=True)
            
            # Q-value 및 자격 기계 로그 항상 출력
            machine_action = agent2.select_action(state_data, op_action, log_q_values=True)
            
            # is_validation=True로 설정하여 보상 로그도 reward_log.log에 기록
            _, _, state_dict, done = env.step((op_action, machine_action), is_validation=True)
            step_in_episode += 1

    final_makespan = env._calculate_makespan()
    print(f"--- Analysis Complete. Makespan: {final_makespan:.2f} ---")
    
    # --- 4. 해당 인스턴스에 대한 간트 차트 생성 ---
    schedule_data_for_plot = {}
    for m in env.machines:
        # machine 객체마다 빈 리스트를 먼저 할당합니다.
        schedule_data_for_plot[m.id] = []
        # busy_intervals에 모든 정보가 함께 있으므로, 이 리스트만 순회합니다.
        for interval_data in m.busy_intervals:
            # 튜플에서 모든 정보를 한 번에 언패킹합니다.
            start, end, job_id, op_id = interval_data
            # 최종 플로팅용 데이터 리스트에 추가합니다.
            schedule_data_for_plot[m.id].append((job_id, op_id, start, end))
    gantt_filename = f'validation_gantt_step_{current_step}.png'
    plot_gantt_chart(schedule_data_for_plot, env.n_jobs, env.n_machines, gantt_filename)

    agent1.actor.train()
    agent2.q_network.train()
    
    # 5. 디버깅 중이므로, 평균이 아닌 단일 인스턴스의 makespan을 반환
    return final_makespan

def main():
    writer = SummaryWriter()
    envs = [FJSPEnv(n_jobs=N_JOBS, n_machines=N_MACHINES) for _ in range(BATCH_SIZE)]
    validation_instances = [FJSPEnv(n_jobs=N_JOBS, n_machines=N_MACHINES).instance_jobs_data for _ in range(100)]
    max_ops_in_dataset = max(sum(len(job_data) for env in envs for job_data in env.instance_jobs_data), MAX_OPS_PER_JOB * N_JOBS)

    agent1 = Agent1_SAC(op_feature_dim=10, max_ops=max_ops_in_dataset)
    agent2 = Agent2_D5QN(
        machine_feature_dim=5, # 5차원으로 변경
        op_machine_pair_dim=2,
        hidden_size=TRANSFORMER_HIDDEN_SIZE, n_heads=TRANSFORMER_N_HEADS, n_layers=TRANSFORMER_N_LAYERS
    )
    
    sac_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)
    d5qn_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)
    current_states = [env.reset() for env in envs]
    dones = [False] * BATCH_SIZE
    log_data = {'sac': [], 'd5qn': []}; episode_makespans = []
    best_val_makespan = float('inf')
    
    file_exists = os.path.exists(VALIDATION_LOG_FILE)
    if not file_exists:
        with open(VALIDATION_LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['step', 'validation_makespan', 'timestamp'])

    print("훈련을 시작합니다 (단일 프로세스, 안정화 모드)...")
    for t in tqdm(range(1, 200001)):
        for i in range(BATCH_SIZE):
            if dones[i]:
                if envs[i]._calculate_makespan() > 0: episode_makespans.append(envs[i]._calculate_makespan())
                pre_occupied = generate_pre_occupied_schedule(envs[i].n_machines)
                current_states[i] = envs[i].reset(initial_schedule=pre_occupied)
                agent2.q_network.reset_noise(); agent2.target_network.reset_noise()

            current_state_dict = current_states[i]
            state_data = Data(**current_state_dict)

            op_action, _, _ = agent1.select_action(state_data)
            machine_action = agent2.select_action(state_data, op_action)
            action = (op_action, machine_action)
            
            # --- 이 부분이 수정되었습니다 ---
            # step 함수의 반환 값을 4개의 명확한 변수로 받습니다.
            prev_state_dict, reward, next_state_dict, done = envs[i].step(action)
            
            # 반환받은 딕셔너리를 사용해 Data 객체 생성
            prev_state_data = Data(**prev_state_dict)
            next_state_data = Data(**next_state_dict)

            # 버퍼에 정확한 데이터를 저장
            sac_buffer.push(prev_state_data, action, reward, next_state_data, done)
            d5qn_buffer.push(prev_state_data, action, reward, next_state_data, done)
            
            current_states[i] = next_state_dict
            dones[i] = done
            
        if len(d5qn_buffer) > BATCH_SIZE * 10:
            sac_loss_dict = agent1.update(sac_buffer, BATCH_SIZE)
            d5qn_loss_dict = agent2.update(d5qn_buffer, BATCH_SIZE)
            if sac_loss_dict: log_data['sac'].append(sac_loss_dict)
            if d5qn_loss_dict: log_data['d5qn'].append(d5qn_loss_dict)

        if t % 30 == 0 and t > 0:
            avg_val_makespan = validate(agent1, agent2, validation_instances, max_ops_in_dataset, t)
            avg_train_makespan = np.mean(episode_makespans) if episode_makespans else 0
            
            with open(VALIDATION_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer_csv = csv.writer(f)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                writer_csv.writerow([t, f"{avg_val_makespan:.2f}", timestamp])

            avg_sac_losses = {k: np.mean([d[k] for d in log_data['sac']]) for k in log_data['sac'][0]} if log_data['sac'] else {}
            avg_d5qn_losses = {k: np.mean([d[k] for d in log_data['d5qn']]) for k in log_data['d5qn'][0]} if log_data['d5qn'] else {}

            writer.add_scalar('Makespan/Train', avg_train_makespan, t)
            writer.add_scalar('Makespan/Validation', avg_val_makespan, t)
            for k, v in avg_sac_losses.items():
                writer.add_scalar(f'SAC/{k}', v, t)
            for k, v in avg_d5qn_losses.items():
                writer.add_scalar(f'D5QN/{k}', v, t)

            print(f"\n[Step {t}] Train Makespan: {avg_train_makespan:.2f} | Validation Makespan: {avg_val_makespan:.2f}")
            print(f"  SAC Losses: {avg_sac_losses}")
            print(f"  D5QN Losses: {avg_d5qn_losses}")

            if avg_val_makespan < best_val_makespan:
                best_val_makespan = avg_val_makespan
                torch.save(agent1.actor.state_dict(), 'best_actor_model.pth')
                torch.save(agent2.q_network.state_dict(), 'best_q_network_model.pth')
                print(f"*** 새로운 최고 성능 모델 저장! (Validation Makespan: {best_val_makespan:.2f}) ***")

            log_data = {'sac': [], 'd5qn': []}
            episode_makespans = []
            
    writer.close()

if __name__ == "__main__":
    main()