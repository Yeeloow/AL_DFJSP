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
import logging

def setup_loss_logger():
    # 'loss_logger' 라는 고유한 이름으로 로거를 생성
    logger = logging.getLogger('loss_logger')
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # 핸들러가 이미 설정되어 있는지 확인하여 중복 방지
    if not logger.handlers:
        handler = logging.FileHandler('loss.log', mode='w', encoding='utf-8')
        # step과 loss 값만 기록하므로, 메시지만 출력하는 간단한 포맷터 사용
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
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
    print(f"\n--- [Validation Step {current_step}] ---")
    
    agent1.actor.eval()
    agent2.q_network.eval()

    # --- Part 1: 첫 번째 인스턴스에 대한 상세 분석 및 시각화 ---
    print("Analyzing the first instance for detailed logs and Gantt chart...")
    first_instance_data = validation_instances[0]
    env_detail = FJSPEnv()
    env_detail.instance_jobs_data = first_instance_data
    
    pre_occupied_schedule = generate_pre_occupied_schedule(env_detail.n_machines)
    state_dict_detail = env_detail.reset(initial_schedule=pre_occupied_schedule)
    
    done_detail = False
    with torch.no_grad():
        while not done_detail:
            state_data = Data(**state_dict_detail)
            op_action, _, _ = agent1.select_action(state_data, deterministic=True)
            
            # --- ▼▼▼ 수정된 부분 ▼▼▼ ---
            # current_step=current_step 인자를 명시적으로 전달합니다.
            machine_action = agent2.select_action(state_data, op_action, log_q_values=True, current_step=current_step)
            _, _, state_dict_detail, done_detail = env_detail.step((op_action, machine_action), is_validation=True, current_step=current_step)
            # --- ▲▲▲ 수정 끝 ▲▲▲ ---

    # 첫 번째 인스턴스에 대한 간트 차트 저장
    schedule_data_for_plot = {}
    for m in env_detail.machines:
        schedule_data_for_plot[m.id] = []
        for interval_data in m.busy_intervals:
            start, end, job_id, op_id = interval_data
            schedule_data_for_plot[m.id].append((job_id, op_id, start, end))
    gantt_filename = f'validation_gantt_step_{current_step}.png'
    plot_gantt_chart(schedule_data_for_plot, env_detail.n_jobs, env_detail.n_machines, gantt_filename)

    # --- Part 2: 전체 검증 인스턴스에 대한 평균 성능 측정 ---
    print(f"Running on all {len(validation_instances)} instances for average makespan...")
    all_makespans = []
    for instance_data in validation_instances:
        env_avg = FJSPEnv()
        env_avg.instance_jobs_data = instance_data
        pre_occupied_schedule = generate_pre_occupied_schedule(env_avg.n_machines)
        state_dict_avg = env_avg.reset(initial_schedule=pre_occupied_schedule)
        
        done_avg = False
        with torch.no_grad():
            while not done_avg:
                state_data = Data(**state_dict_avg)
                op_action, _, _ = agent1.select_action(state_data, deterministic=True)
                # Part 2는 로그를 남기지 않으므로 current_step을 전달할 필요가 없습니다.
                machine_action = agent2.select_action(state_data, op_action, log_q_values=False)
                _, _, state_dict_avg, done_avg = env_avg.step((op_action, machine_action), is_validation=False)

        final_makespan = env_avg._calculate_makespan()
        all_makespans.append(final_makespan)

    average_makespan = np.mean(all_makespans)
    print(f"--- Validation Complete. Average Makespan: {average_makespan:.2f} ---")

    agent1.actor.train()
    agent2.q_network.train()
    
    return average_makespan

def main():
    writer = SummaryWriter()
    loss_logger = setup_loss_logger()
    loss_logger.info("step,sac_critic_loss,sac_actor_loss,sac_alpha_loss,d5qn_loss,d5qn_avg_q_value,d5qn_avg_td_error")
    # --- ▲▲▲ 추가 끝 ▲▲▲ ---
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

        if t % 500 == 0 and t > 0:
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

           # --- ▼▼▼ [추가됨] 계산된 Loss들을 loss.log 파일에 기록 ▼▼▼ ---
            if avg_sac_losses and avg_d5qn_losses:
                sac_critic = avg_sac_losses.get('critic_loss', 'N/A')
                sac_actor = avg_sac_losses.get('actor_loss', 'N/A')
                sac_alpha = avg_sac_losses.get('alpha_loss', 'N/A')
                d5qn_loss = avg_d5qn_losses.get('d5qn_loss', 'N/A')
                d5qn_q = avg_d5qn_losses.get('avg_q_value', 'N/A')
                d5qn_td = avg_d5qn_losses.get('avg_td_error', 'N/A')
                
                log_msg = (f"{t},{sac_critic:.4f},{sac_actor:.4f},{sac_alpha:.4f},"
                           f"{d5qn_loss:.4f},{d5qn_q:.4f},{d5qn_td:.4f}")
                loss_logger.info(log_msg)
            # --- ▲▲▲ 추가 끝 ▲▲▲ ---

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