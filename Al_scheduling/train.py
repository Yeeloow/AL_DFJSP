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
    """
    주어진 스케줄 데이터로 간트 차트를 그리고 파일로 저장하는 함수
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 각 Job에 대한 고유한 색상 설정
    colors = plt.cm.get_cmap('tab20', n_jobs)

    for m_id, tasks in schedule_data.items():
        for task in tasks:
            job_id, op_id, start, end = task
            duration = end - start
            
            # 사전 점유 스케줄은 회색으로, 에이전트가 스케줄한 작업은 Job 색상으로 표시
            color = colors(job_id % 20) if job_id != -1 else 'gray'
            
            # 막대그래프 추가 (시작시간, 기간)
            ax.broken_barh([(start, duration)], (m_id * 10, 9), facecolors=color, edgecolor='black')
            
            # 막대 안에 텍스트 추가
            text = f'J{job_id}-O{op_id}' if job_id != -1 else 'Pre'
            ax.text(start + duration / 2, m_id * 10 + 4.5, text, ha='center', va='center', color='white', fontweight='bold')

    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Gantt Chart for Validation Instance')
    ax.set_yticks([i * 10 + 4.5 for i in range(n_machines)])
    ax.set_yticklabels([f'Machine {i}' for i in range(n_machines)])
    ax.grid(True, axis='x', linestyle=':')
    
    # 범례(Legend) 추가
    legend_elements = [Patch(facecolor=colors(i % 20), label=f'Job {i}') for i in range(n_jobs)]
    legend_elements.append(Patch(facecolor='gray', label='Pre-occupied'))
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)  # 메모리 누수를 방지하기 위해 그림 닫기
    print(f"간트 차트가 '{filename}' 파일로 저장되었습니다.")
def plot_eligibility_table(jobs, filename):
    """
    Job/Operation별로 할당 가능한 기계 목록을 테이블 형태의 이미지로 저장하는 함수
    """
    row_labels = []
    cell_text = []
    
    # 테이블에 들어갈 데이터 준비
    for job in jobs:
        for op in job.ops:
            row_labels.append(f'Job {job.id} - Op {op.id}')
            # 기계 ID 리스트를 문자열로 변환 (예: [1, 2, 5] -> "1, 2, 5")
            machines_str = ', '.join(map(str, op.eligible_machines))
            cell_text.append([machines_str])

    if not cell_text:
        print("[Table Plot] No data to plot for eligibility table.")
        return

    # 테이블 크기 동적 조절
    fig_height = max(4, len(row_labels) * 0.4)
    fig, ax = plt.subplots(figsize=(6, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # 테이블 생성
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=['Eligible Machines'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_title('Job-Machine Eligibility Table', fontweight="bold", y=0.9)
    fig.tight_layout()
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"장비 할당 테이블이 '{filename}' 파일로 저장되었습니다.")

VALIDATION_LOG_FILE = 'validation_log.csv'

warnings.filterwarnings("ignore", message="gemm_and_bias error:.*")

def generate_pre_occupied_schedule(n_machines):
    """
    기계별로 랜덤한 개수의, 겹치지 않는 사전 점유 스케줄을 생성하는 함수
    """
    schedule = {}
    # 약 50%의 기계에만 사전 점유 스케줄을 할당
    for m_id in range(n_machines):
        if np.random.rand() < 0.5:
            
            # --- 이 부분이 수정되었습니다 ---
            machine_intervals = []
            # 해당 기계에 1~3개의 랜덤한 개수의 스케줄을 할당
            num_intervals = np.random.randint(1, 4)
            last_end_time = 0

            for _ in range(num_intervals):
                # 이전 스케줄이 끝난 후, 50의 랜덤한 간격을 둠
                gap = np.random.randint(0, 51)
                start_time = last_end_time + gap

                duration = np.random.randint(10, 30)
                end_time = start_time + duration
                
                machine_intervals.append((-1, -1, start_time, end_time))
                
                # 다음 스케줄이 겹치지 않도록 마지막 종료 시간 업데이트
                last_end_time = end_time
            
            schedule[m_id] = machine_intervals
            # --- 여기까지 수정 ---
            
    return schedule

def validate(agent1, agent2, validation_instances, max_ops, current_step):
    total_makespan = 0
    validation_results = []

    agent1.actor.eval()
    agent2.q_network.eval()
    with torch.no_grad():
        for i, instance_data_list in enumerate(validation_instances):
            env = FJSPEnv()
            env.instance_jobs_data = instance_data_list
            
            pre_occupied_schedule = generate_pre_occupied_schedule(env.n_machines)
            state_dict = env.reset(initial_schedule=pre_occupied_schedule)
            
            step_in_episode = 0
            done = False
            while not done:
                state_data = Data(**state_dict)
                op_action, _, _ = agent1.select_action(state_data, deterministic=True)
                machine_action = agent2.select_action(state_data, op_action)
                state_dict, _, done = env.step((op_action, machine_action), is_validation=True)
                step_in_episode += 1
            
            total_makespan += env._calculate_makespan()
            
            schedule_data_for_plot = {}
            for m in env.machines:
                for op_info, interval in zip(m.processed_op_sequence, m.busy_intervals):
                    job_id, op_id = op_info
                    start, end = interval
                    if m.id not in schedule_data_for_plot:
                        schedule_data_for_plot[m.id] = []
                    schedule_data_for_plot[m.id].append((job_id, op_id, start, end))
            
            validation_results.append({
                'steps': step_in_episode,
                'schedule_data': schedule_data_for_plot,
                'n_jobs': env.n_jobs,
                'n_machines': env.n_machines,
                'jobs': env.jobs  # --- 이 부분을 추가하여 Job 데이터 저장 ---
            })

    if validation_results:
        longest_instance = max(validation_results, key=lambda x: x['steps'])
        
        print("\n" + "="*50)
        print(f"[Gantt Chart & Table] Plotting longest instance from validation at step {current_step}.")
        print(f"  -> Instance ran for {longest_instance['steps']} steps.")
        print("="*50 + "\n")
        
        # 간트 차트 생성
        gantt_filename = f'validation_gantt_step_{current_step}_longest.png'
        plot_gantt_chart(
            longest_instance['schedule_data'],
            longest_instance['n_jobs'],
            longest_instance['n_machines'],
            gantt_filename
        )
        
        # --- 테이블 생성 함수 호출 추가 ---
        table_filename = f'validation_eligibility_table_step_{current_step}.png'
        plot_eligibility_table(longest_instance['jobs'], table_filename)
        # --- 여기까지 추가 ---

    agent1.actor.train()
    agent2.q_network.train()
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

        if t % 100 == 0 and t > 0:
            avg_val_makespan = validate(agent1, agent2, validation_instances, max_ops_in_dataset, t)
            avg_train_makespan = np.mean(episode_makespans) if episode_makespans else 0
            
            with open(VALIDATION_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer_csv = csv.writer(f)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                writer_csv.writerow([t, f"{avg_val_makespan:.2f}", timestamp])

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