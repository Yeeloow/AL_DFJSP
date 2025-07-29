# debug_step_by_step.py

import torch
import numpy as np
from torch_geometric.data import Data
from fjsp_environment import FJSPEnv
from agents import Agent1_SAC, Agent2_D5QN
from config import *

def run_single_step_debug():
    print("--- 스텝별 디버깅 시작 ---")
    new_jobs_instance = [
        [([0, 1], [10, 15])],
        [([0, 1], [8, 12])]
    ]
    pre_occupied_schedule = {
        0: [(-1, -1, 10, 20)],
        1: [(-1, -1, 15, 30)]
    }
    n_jobs = len(new_jobs_instance)
    n_machines = len(pre_occupied_schedule)
    max_ops = sum(len(job_data) for job_data in new_jobs_instance)
    
    agent1 = Agent1_SAC(op_feature_dim=10, max_ops=max_ops)
    agent2 = Agent2_D5QN(
        machine_feature_dim=7, op_machine_pair_dim=2,
        hidden_size=TRANSFORMER_HIDDEN_SIZE, n_heads=TRANSFORMER_N_HEADS, n_layers=TRANSFORMER_N_LAYERS
    )
    agent1.actor.eval(); agent2.q_network.eval()

    env = FJSPEnv(n_jobs=n_jobs, n_machines=n_machines)
    env.instance_jobs_data = new_jobs_instance
    
    state_dict = env.reset(initial_schedule=pre_occupied_schedule)
    
    assert 'x' in state_dict and state_dict['x'] is not None, "FATAL: reset() did not return 'x' feature."
    state_data = Data(**state_dict)

    print("\n--- 초기 환경 상태 ---")
    print(f"Machine 0의 초기 available_time: {env.machines[0].available_time}")
    print(f"Machine 1의 초기 available_time: {env.machines[1].available_time}")
    print(f"처리 가능한 공정 목록: {state_data.eligible_ops}")
    
    print("\n--- 첫 번째 스텝 결정 과정 ---")
    
    op_action, _, _ = agent1.select_action(state_data, deterministic=True)
    print(f"Agent 1이 선택한 공정(op_action): {op_action}")
    
    print("\nAgent 2의 내부 계산 과정:")
    eft_tensor = agent2._calculate_statistical_features(state_data, op_action)
    for m_idx in range(n_machines):
        print(f"  - Machine {m_idx}: Earliest Fit Time = {eft_tensor[m_idx, 0].item()}, Num Slots = {eft_tensor[m_idx, 1].item()}")
        
    machine_action = agent2.select_action(state_data, op_action)
    print(f"Agent 2가 선택한 기계(machine_action): {machine_action}")
    
    action = (op_action, machine_action)
    _, _, _, _ = env.step(action)
    
    print("\n--- 스텝 실행 후 결과 ---")
    job_id, op_id = env.op_map_rev[op_action]
    scheduled_op = env.jobs[job_id].ops[op_id]
    
    print(f"선택된 공정 J{job_id}-O{op_id}가 Machine {machine_action}에 할당됨")
    print(f"  - 시작 시간: {scheduled_op.start_time:.2f}")
    print(f"  - 종료 시간: {scheduled_op.completion_time:.2f}")

    print(f"\nMachine 0의 최종 available_time: {env.machines[0].available_time}")
    print(f"Machine 1의 최종 available_time: {env.machines[1].available_time}")

    print("\n--- 디버깅 완료 ---")
    print("시작 시간이 pre-occupied schedule (10~20)과 겹치지 않는지 확인하세요.")

if __name__ == '__main__':
    run_single_step_debug()