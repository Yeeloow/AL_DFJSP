# fjsp_environment.py

import numpy as np
import torch
from config import *
import copy
import logging


# --- 아래 로깅 설정 코드 추가 ---

# 1. 로거 생성
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 2. 파일 핸들러 생성 (로그를 파일에 저장)
# mode='w'는 실행할 때마다 로그 파일을 새로 씀 (이전 내용은 삭제)
file_handler = logging.FileHandler('environment.log', mode='w')

# 3. 로그 포맷 설정
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 4. 로거에 핸들러 추가
if not logger.handlers:
    logger.addHandler(file_handler)


class Operation:
    def __init__(self, op_id, job_id, data):
        self.id = op_id; self.job_id = job_id
        self.eligible_machines, self.processing_times = data
        self.is_scheduled = False; self.completion_time = 0.0; self.start_time = 0.0
        self.min_proc_time = np.min(self.processing_times) if len(self.processing_times) > 0 else 0
        self.avg_proc_time = np.mean(self.processing_times) if len(self.processing_times) > 0 else 0
        self.machine_map = {m: t for m, t in zip(self.eligible_machines, self.processing_times)}
        self.scheduled_machine = None

class Job:
    def __init__(self, job_id, ops_data):
        self.id = job_id
        self.ops = [Operation(i, self.id, data) for i, data in enumerate(ops_data)]
    @property
    def remaining_workload(self): return sum(op.avg_proc_time for op in self.ops if not op.is_scheduled)
    @property
    def completed_ops_count(self): return sum(1 for op in self.ops if op.is_scheduled)
    @property
    def completion_rate(self): return self.completed_ops_count / len(self.ops) if len(self.ops) > 0 else 1.0

class Machine:
    def __init__(self, machine_id):
        self.id = machine_id; self.available_time = 0.0
        # op_history를 두 개의 별도 리스트로 분리하여 역할 명확화
        self.busy_intervals = [] # (start_time, end_time) 튜플 저장용
        self.processed_op_sequence = [] # (job_id, op_id) 튜플 저장용

    @property
    def utilization_rate(self):
        if self.available_time == 0: return 0.0
        work_time = sum(end - start for start, end in self.busy_intervals)
        return work_time / (self.available_time + 1e-8)

class FJSPEnv:
    def __init__(self, n_jobs=N_JOBS, n_machines=N_MACHINES):
        self.n_jobs, self.n_machines = n_jobs, n_machines
        self._generate_random_instance()
        self.travel_times = np.random.randint(5, 16, size=(n_machines, n_machines))
        np.fill_diagonal(self.travel_times, 0)

    def _generate_random_instance(self):
        self.instance_jobs_data = []
        for i in range(self.n_jobs):
            ops = []; num_ops = np.random.randint(1, MAX_OPS_PER_JOB + 1)
            for j in range(num_ops):
                num_eligible_machines = np.random.randint(1, self.n_machines + 1)
                machines = np.random.choice(self.n_machines, num_eligible_machines, replace=False)
                times = np.random.randint(10, 50, size=num_eligible_machines)
                if len(times) == 0: times = [np.random.randint(10, 50)]
                ops.append((list(machines), list(times)))
            self.instance_jobs_data.append(ops)

    def reset(self, initial_schedule=None):
        self.jobs = [Job(i, copy.deepcopy(ops_data)) for i, ops_data in enumerate(self.instance_jobs_data)]
        self.machines = [Machine(i) for i in range(self.n_machines)]
        self.current_time = 0; self.op_map = {}; self.op_map_rev = {}
        idx = 0
        for job in self.jobs:
            for op in job.ops:
                self.op_map[(op.job_id, op.id)] = idx
                self.op_map_rev[idx] = (op.job_id, op.id)
                idx += 1
        self.total_ops = len(self.op_map)
        if initial_schedule:
            max_time = 0
            for machine_id, schedule_list in initial_schedule.items():
                if machine_id < len(self.machines):
                    machine = self.machines[machine_id]
                    for job_id, op_id, start_time, end_time in schedule_list:
                        machine.busy_intervals.append((start_time, end_time))
                        machine.processed_op_sequence.append((job_id, op_id))
                        machine.available_time = max(machine.available_time, end_time)
                        max_time = max(max_time, end_time)
                    machine.busy_intervals.sort(key=lambda x: x[0])
            self.current_time = max_time
        return self._get_state()

    def _get_eligible_ops_details(self):
        eligible_ops_details = {}
        for job in self.jobs:
            next_op_idx = job.completed_ops_count
            if next_op_idx < len(job.ops):
                op = job.ops[next_op_idx]
                prev_op_done = (op.id == 0) or (job.ops[op.id - 1].is_scheduled)
                if prev_op_done:
                    global_op_idx = self.op_map[(job.id, op.id)]
                    prev_op_comp_time = 0 if op.id == 0 else job.ops[op.id - 1].completion_time
                    eligible_ops_details[global_op_idx] = prev_op_comp_time
        return eligible_ops_details

    def _get_state(self):
        op_features_list = []
        raw_details = self._get_eligible_ops_details()
        eligible_ops_indices = list(raw_details.keys())
        detail_list = [raw_details.get(i, 0) for i in range(self.total_ops)]
        eligible_ops_details = torch.tensor(detail_list, dtype=torch.long)
        
        total_completion_rate_job = sum(j.completion_rate for j in self.jobs) / self.n_jobs if self.n_jobs > 0 else 0
        total_completion_rate_op = sum(j.completed_ops_count for j in self.jobs) / self.total_ops if self.total_ops > 0 else 0
        
        if self.total_ops > 0:
            for g_idx in range(self.total_ops):
                job_id, op_id = self.op_map_rev[g_idx]
                job, op = self.jobs[job_id], self.jobs[job_id].ops[op_id]
                prev_op_comp_time = 0 if op.id == 0 else job.ops[op.id - 1].completion_time
                est_start_time = np.mean([max(self.machines[m_idx].available_time, prev_op_comp_time) for m_idx in op.eligible_machines]) if op.eligible_machines else 0
                est_comp_time = est_start_time + op.avg_proc_time
                op_features_list.append([self.n_jobs, 1 if op.is_scheduled else 0, est_comp_time, op.min_proc_time, op.avg_proc_time, job.remaining_workload, job.completed_ops_count, job.completion_rate, total_completion_rate_op, total_completion_rate_job])
        machine_features_list = []
        avg_utilization = sum(m.utilization_rate for m in self.machines) / self.n_machines if self.n_machines > 0 else 0
        
        for m_idx in range(self.n_machines):
            # 현재 인덱스에 해당하는 기계 객체를 가져옵니다.
            machine = self.machines[m_idx]
            # 현재 기계(m_idx)에서 처리 가능한 '유효 작업(eligible ops)'의 총 개수를 계산합니다.
            num_candidates = sum(1 for op_g_idx in eligible_ops_indices if m_idx in self.jobs[self.op_map_rev[op_g_idx][0]].ops[self.op_map_rev[op_g_idx][1]].eligible_machines)
            # 현재 기계에 대한 특징들을 리스트 형태로 'machine_features_list'에 추가합니다.
            machine_features_list.append([self.n_machines, 1 if machine.available_time > self.current_time else 0, machine.available_time - self.current_time, machine.available_time, num_candidates, machine.utilization_rate, avg_utilization])

        # '작업(Operation)-기계(Machine)' 쌍의 특징을 저장할 2차원 리스트를 초기화합니다. (모든 값을 [0.0, 0.0]으로 설정)
        op_machine_pairs_features_list = [[([0.0, 0.0]) for _ in range(self.n_machines)] for _ in range(self.total_ops)]

        # 처리할 작업이 하나라도 있는 경우에만 아래 코드를 실행합니다.
        if self.total_ops > 0:
            # 전체 작업 수만큼 반복문을 실행합니다. (op_g_idx: 작업의 전역 인덱스)
            for op_g_idx in range(self.total_ops):
                # 작업의 전역 인덱스(op_g_idx)를 사용하여 실제 Job ID와 Operation ID를 찾습니다. (주: 원본 코드의 g_idx는 op_g_idx의 오타로 보입니다)
                job_id, op_id = self.op_map_rev[op_g_idx]
                # 찾은 ID를 이용해 실제 Job 객체와 Operation 객체를 가져옵니다.
                job, op = self.jobs[job_id], job.ops[op_id]
                # 현재 작업(op)을 처리할 수 있는 기계와 해당 처리 시간을 가져와 반복문을 실행합니다.
                for m_idx, proc_time in op.machine_map.items():
                    # '작업 부하 비율'을 계산합니다: (처리 시간) / (해당 Job의 남은 총 작업 시간)
                    ratio = proc_time / (job.remaining_workload + 1e-8)
                    # '작업-기계' 쌍 특징 리스트의 해당 위치에 [처리 시간, 작업 부하 비율]을 업데이트합니다.
                    op_machine_pairs_features_list[op_g_idx][m_idx] = [proc_time, ratio]
        
        edge_index_list = []
        for job in self.jobs:
            for i in range(len(job.ops) - 1): edge_index_list.append([self.op_map[(job.id, i)], self.op_map[(job.id, i + 1)]])
        
        for machine in self.machines:
            for i in range(len(machine.processed_op_sequence) - 1):
                u_op_ids, v_op_ids = machine.processed_op_sequence[i], machine.processed_op_sequence[i+1]
                u_global_idx = self.op_map.get(u_op_ids)
                v_global_idx = self.op_map.get(v_op_ids)
                if u_global_idx is not None and v_global_idx is not None:
                    edge_index_list.append([u_global_idx, v_global_idx])
        
        return {'x': torch.tensor(op_features_list, dtype=torch.float),
                'edge_index': torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() if edge_index_list else torch.empty((2,0), dtype=torch.long),
                'eligible_ops': eligible_ops_indices,
                'eligible_ops_details': eligible_ops_details,
                'm_features': torch.tensor(machine_features_list, dtype=torch.float),
                'om_features': torch.tensor(op_machine_pairs_features_list, dtype=torch.float),
                'num_nodes': self.total_ops,
                'op_histories': [m.busy_intervals for m in self.machines]}

    def step(self, action, is_validation=False):
        op_global_idx, machine_idx = action

        if op_global_idx not in self._get_eligible_ops_details():
            return self._get_state(), -500.0, True

        job_id, op_id = self.op_map_rev[op_global_idx]
        op = self.jobs[job_id].ops[op_id]

        if machine_idx not in op.eligible_machines:
            return self._get_state(), -500.0, True

        # 행동 실행 전 Makespan 기록
        prev_makespan = self._calculate_makespan()
        
        machine = self.machines[machine_idx]
        processing_time = op.machine_map[machine_idx]
        prev_op_comp_time = self._get_eligible_ops_details().get(op_global_idx, 0)
        
        travel_time = 0
        if op.id > 0:
            prev_op = self.jobs[job_id].ops[op.id - 1]
            prev_machine_idx = prev_op.scheduled_machine
            if prev_machine_idx is not None and prev_machine_idx != machine_idx:
                travel_time = self.travel_times[prev_machine_idx][machine_idx]
        
        ready_time_after_travel = prev_op_comp_time + travel_time
        start_time = max(machine.available_time, ready_time_after_travel)
        completion_time = start_time + processing_time
        
        op.is_scheduled, op.start_time, op.completion_time = True, start_time, completion_time
        op.scheduled_machine = machine_idx
        
        machine.available_time = completion_time
        machine.busy_intervals.append((start_time, completion_time))
        machine.busy_intervals.sort(key=lambda x: x[0])
        machine.processed_op_sequence.append((op.job_id, op.id))
        
        if any(op.is_scheduled for job in self.jobs for op in job.ops):
            if any(m.busy_intervals for m in self.machines):
                self.current_time = min(m.available_time for m in self.machines if m.busy_intervals)
        
        # 행동 실행 후 Makespan 기록 및 보상 계산
        new_makespan = self._calculate_makespan()
        reward = (prev_makespan - new_makespan) / processing_time
        
        # --- is_validation일 때만 보상 로그 기록 ---
        if is_validation:
            logger.debug(f"--- Reward Log for (Op: {op_global_idx}, Machine: {machine_idx}) ---")
            logger.debug(f"  - Makespan Change: {prev_makespan:.2f} -> {new_makespan:.2f} | Reward = {reward:.2f}")
        
        done = all(op.is_scheduled for job in self.jobs for op in job.ops)
            
        return self._get_state(), reward, done

    def _calculate_makespan(self):
        max_completion_time = 0
        for job in self.jobs:
            for op in job.ops:
                if op.is_scheduled: max_completion_time = max(max_completion_time, op.completion_time)
        return max_completion_time