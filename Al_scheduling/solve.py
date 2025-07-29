# solve.py

import torch
from fjsp_environment import FJSPEnv
from agents import Agent1_SAC, Agent2_D5QN
from config import *
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data

def plot_gantt_chart(env):
    fig, ax = plt.subplots(figsize=(15, 10))
    colors = plt.colormaps.get_cmap('viridis', env.n_jobs)
    for machine_idx, machine in enumerate(env.machines):
        for op, start, end in machine.op_history:
            job_color = colors(op.job_id)
            ax.barh(y=f"Machine {machine_idx}", width=end - start, left=start, height=0.7, 
                    align='center', color=job_color, edgecolor='black')
            ax.text(start + (end - start)/2, machine_idx, f'J{op.job_id}-O{op.id}', 
                    ha='center', va='center', color='white', fontweight='bold')
    ax.set_xlabel('Time'); ax.set_title('Flexible Job Shop Scheduling Gantt Chart')
    ax.invert_yaxis(); plt.grid(axis='x', linestyle='--', alpha=0.6); plt.show()

def solve_new_problem():
    new_problem_instance = [
        [([0, 1], [10, 12]), ([2, 3], [8, 10])],
        [([0, 2], [5, 7]), ([1, 3], [15, 13]), ([0, 3], [8, 8])],
        [([1, 2], [9, 11]), ([0, 3], [6, 9])],
    ]
    n_jobs = len(new_problem_instance); n_machines = 4
    max_ops = sum(len(job_data) for job_data in new_problem_instance)

    print("학습된 에이전트 모델을 생성하고 가중치를 불러옵니다...")
    agent1 = Agent1_SAC(op_feature_dim=10, max_ops=max_ops)
    agent2 = Agent2_D5QN(
        machine_feature_dim=8, op_machine_pair_dim=2,
        hidden_size=TRANSFORMER_HIDDEN_SIZE, n_heads=TRANSFORMER_N_HEADS, n_layers=TRANSFORMER_N_LAYERS
    )
    try:
        agent1.actor.load_state_dict(torch.load('agent1_actor_best.pth', map_location=DEVICE))
        agent2.q_network.load_state_dict(torch.load('agent2_d5qn_best.pth', map_location=DEVICE))
        print("가중치 로딩 성공!")
    except FileNotFoundError:
        print("오류: 저장된 모델 가중치 파일(.pth)을 찾을 수 없습니다."); return

    agent1.actor.eval(); agent2.q_network.eval()
    print("\n새로운 문제에 대한 스케줄링을 시작합니다...")
    env = FJSPEnv(n_jobs=n_jobs, n_machines=n_machines)
    env.instance_jobs_data = new_problem_instance
    state_dict = env.reset(); done = False
    
    while not done:
        state_data = Data(**state_dict)
        op_action, _, _ = agent1.select_action(state_data, deterministic=True)
        machine_action = agent2.select_action(state_data, op_action)
        action = (op_action, machine_action)
        state_dict, _, done = env.step(action)

    final_makespan = env._calculate_makespan()
    print(f"\n스케줄링 완료! 최종 Makespan: {final_makespan:.2f}")
    print("결과를 간트 차트로 시각화합니다.")
    plot_gantt_chart(env)

if __name__ == '__main__':
    solve_new_problem()