# fjsp_environment.py

import torch
import numpy as np
from config import *
import copy
import logging

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # 로그 레벨 설정

# 로그 파일 핸들러 생성 ('w' 모드는 실행 시마다 파일을 새로 씀)
file_handler = logging.FileHandler('reward_log.log', mode='w')

# 로그 포맷터 생성 (시간 - 로그레벨 - 메시지 형식)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 로거에 핸들러 추가 (중복 추가 방지)
if not logger.handlers:
    logger.addHandler(file_handler)
# --- 로그 설정 끝 ---

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
        self.busy_intervals = []
        # self.processed_op_sequence = []  <-- 이 줄을 삭제하거나 주석 처리합니다.

    @property
    def utilization_rate(self):
        if self.available_time == 0: return 0.0
        # ▼▼▼ 데이터 구조 변경에 따라 이 부분을 수정합니다 ▼▼▼
        work_time = sum(end - start for start, end, _, _ in self.busy_intervals)
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
            for machine_id, schedule_list in initial_schedule.items():
                if machine_id < len(self.machines):
                    machine = self.machines[machine_id]
                    for job_id, op_id, start_time, end_time in schedule_list:
                        # ▼▼▼ 데이터를 하나로 통합하여 저장합니다 ▼▼▼
                        machine.busy_intervals.append((start_time, end_time, job_id, op_id))
                        # machine.processed_op_sequence.append((job_id, op_id)) <-- 이 줄 삭제
                        machine.available_time = max(machine.available_time, end_time)
                    machine.busy_intervals.sort(key=lambda x: x[0])
            self.current_time = 0
        return self._get_state()

    def _get_eligible_ops_details(self):
        """
        현재 스케줄링 가능한 후보 공정들의 목록과 상세 정보를 반환합니다.
        가장 핵심적인 선행 관계 제약 조건을 이 함수에서 올바르게 처리합니다.
        """
        eligible_ops_details = {}

        # 1. 모든 Job을 하나씩 순회합니다.
        for job in self.jobs:
            
            # 2. 해당 Job의 공정 리스트(job.ops)를 처음부터 순서대로 확인하여,
            #    아직 스케줄링되지 않은('is_scheduled'가 False인) 가장 첫 번째 공정을 찾습니다.
            first_unscheduled_op = None
            for op in job.ops:
                if not op.is_scheduled:
                    first_unscheduled_op = op
                    break  # 찾았으면 더 이상 탐색할 필요가 없으므로 중단합니다.

            # 3. 만약 그런 미완료 공정을 찾았다면,
            if first_unscheduled_op:
                op_to_check = first_unscheduled_op
                
                # 4. 이 공정이 스케줄링 가능한지 '선행 조건'을 확인합니다.
                #   - 이 공정이 작업의 첫 번째 공정(op.id == 0)이거나,
                #   - 혹은 바로 이전 공정(job.ops[op.id - 1])이 이미 완료된 경우에만 가능합니다.
                is_predecessor_done = (op_to_check.id == 0) or (job.ops[op_to_check.id - 1].is_scheduled)

                if is_predecessor_done:
                    # 5. 모든 조건을 만족하면, 이 공정을 최종 후보 목록에 추가합니다.
                    global_op_idx = self.op_map[(job.id, op_to_check.id)]
                    prev_op_comp_time = 0 if op_to_check.id == 0 else job.ops[op_to_check.id - 1].completion_time
                    eligible_ops_details[global_op_idx] = prev_op_comp_time

        return eligible_ops_details

    def _get_state(self):
        # (1) 현재 스텝에서 스케줄링 가능한 작업(operation)들의 세부정보 조회
        raw_details = self._get_eligible_ops_details()
        # (2) eligible operation 들의 글로벌 인덱스 리스트
        eligible_ops_indices = list(raw_details.keys())
        # (3) op별 세부정보를 인덱스 순서에 맞춰 채운 뒤 텐서로 변환 (없는 op는 0으로 채움)
        detail_list = [raw_details.get(i, 0) for i in range(self.total_ops)]
        eligible_ops_details = torch.tensor(detail_list, dtype=torch.long)

        # ---------- Operation feature 계산 ----------
        op_features_list = []
        # (4) 모든 job에 대한 평균 완료율(job 단위)
        total_completion_rate_job = sum(j.completion_rate for j in self.jobs) / self.n_jobs if self.n_jobs > 0 else 0
        # (5) 전체 operation 기준 평균 완료율(operation 단위)
        total_completion_rate_op  = sum(j.completed_ops_count for j in self.jobs) / self.total_ops if self.total_ops > 0 else 0

        if self.total_ops > 0:
            # (6) 각 operation(global index)별 feature 생성
            for g_idx in range(self.total_ops):
                job_id, op_id = self.op_map_rev[g_idx]                      # 글로벌 → (job, op) 매핑
                job, op = self.jobs[job_id], self.jobs[job_id].ops[op_id]   # 해당 job·op 객체
                # (6-1) 바로 이전 공정 완료 시각(첫 공정이면 0)
                prev_op_comp_time = 0 if op.id == 0 else job.ops[op.id - 1].completion_time
                # (6-2) 각 후보 장비의 earliest start time 평균
                est_start_time = np.mean([
                    self._find_first_fit_for_machine(
                        self.machines[m_idx],          # 검사할 기계
                        prev_op_comp_time,              # 선행 공정 완료 시간 (공정 준비 시간)
                        op.avg_proc_time                # 이 기계에서의 평균 처리 시간
                    )
                    for m_idx in op.eligible_machines
                ]) if op.eligible_machines else 0
                # (6-3) 예상 completion time = 예상 시작 + 평균 가공시간
                est_comp_time = est_start_time + op.avg_proc_time

                # (6-4) feature vector 구성
                # [작업 수, 스케줄 여부, 예상 완료 시각, 최소/평균 가공시간,
                #  job 남은 작업량, job 완료된 공정수, job 완료율,
                #  시스템 op 평균 완료율, 시스템 job 평균 완료율]
                op_features_list.append([
                    self.n_jobs,
                    1 if op.is_scheduled else 0,
                    est_comp_time,
                    op.min_proc_time,
                    op.avg_proc_time,
                    job.remaining_workload,
                    job.completed_ops_count,
                    job.completion_rate,
                    total_completion_rate_op,
                    total_completion_rate_job
                ])

         # ---------- Machine feature 계산 (수정된 부분) ----------
        # 1) t_ready_min 계산 (가장 빨리 준비되는 공정의 시작 가능 시각)
        t_ready_min = min((eligible_ops_details[i].item() for i in eligible_ops_indices), default=0)

        # 2) max_delay 계산 (정규화용)
        max_delay = max((max(0, m.available_time - t_ready_min) for m in self.machines), default=0) + 1e-8

        # 3) Machine feature 재구성
        machine_features_list = []
        avg_utilization = sum(m.utilization_rate for m in self.machines) / self.n_machines if self.n_machines > 0 else 0

        for m_idx, machine in enumerate(self.machines):
            num_candidates = sum(
                1 for op_g_idx in eligible_ops_indices
                if m_idx in self.jobs[self.op_map_rev[op_g_idx][0]].ops[self.op_map_rev[op_g_idx][1]].eligible_machines
            )
            # 제안하신 새로운 피처 계산
            can_start_now = 1 if machine.available_time <= t_ready_min else 0
            start_delay = max(0, machine.available_time - t_ready_min)
            start_delay_norm = start_delay / max_delay

            # 최종 Machine Feature Vector (총 5-dim)
            machine_features_list.append([
                num_candidates,         # 0: 이 기계를 쓸 수 있는 후보 공정 수
                can_start_now,          # 1: 공정이 준비됐을 때, 기계가 즉시 투입 가능한지 여부
                start_delay_norm,       # 2: 공정 준비 완료 후 기계 대기 시간 (정규화)
                machine.utilization_rate, # 3: 기계의 누적 가동률
                avg_utilization         # 4: 시스템 전체의 평균 가동률
            ])

        # ---------- Operation–Machine pair feature 초기화 ----------
        # (8) [총 op] x [총 장비] 크기의 2차원 리스트 -> [proc_time, proc_time / 남은 작업량] 형태
        op_machine_pairs_features_list = [
    [[0.0, 0.0] for _ in range(self.n_machines)]
    for _ in range(self.total_ops)
]

        if self.total_ops > 0:
            # (9) 실제 가능한 op–machine 조합에 대해 값 채우기
            for op_g_idx in range(self.total_ops):
                job_id, op_id = self.op_map_rev[op_g_idx]
                job, op = self.jobs[job_id], job.ops[op_id]
                for m_idx, proc_time in op.machine_map.items():
                    ratio = proc_time / (job.remaining_workload + 1e-8)     # 0 div 방지 epsilon
                    op_machine_pairs_features_list[op_g_idx][m_idx] = [proc_time, ratio]

        # ---------- 그래프 edge 구성 ----------
        edge_index_list = []
        # (10) 각 job 내부의 공정 순서 edge
        for job in self.jobs:
            for i in range(len(job.ops) - 1):
                edge_index_list.append([
                    self.op_map[(job.id, i)],
                    self.op_map[(job.id, i + 1)]
                ])

        # (11) 각 장비에서 실제 완료된 공정 순서 edge
        for machine in self.machines:
    # busy_intervals가 시간순으로 정렬되어 있으므로, 이 리스트가 곧 작업 순서입니다.
            for i in range(len(machine.busy_intervals) - 1):
                # i번째와 i+1번째 작업 구간 데이터에서 작업 ID 정보를 추출합니다.
                # 튜플의 3번째, 4번째 요소가 각각 job_id, op_id 입니다.
                u_op_ids = (machine.busy_intervals[i][2], machine.busy_intervals[i][3])
                v_op_ids = (machine.busy_intervals[i+1][2], machine.busy_intervals[i+1][3])
                
                # op_id가 -1인 경우는 사전 점유 스케줄이므로 엣지에서 제외합니다.
                if u_op_ids[1] == -1 or v_op_ids[1] == -1:
                    continue

                u_global_idx = self.op_map.get(u_op_ids)
                v_global_idx = self.op_map.get(v_op_ids)
                
                if u_global_idx is not None and v_global_idx is not None:
                    edge_index_list.append([u_global_idx, v_global_idx])

        # ---------- 상태 딕셔너리 반환 ----------
        return {
            'x'             : torch.tensor(op_features_list, dtype=torch.float),
            'edge_index'    : torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
                                if edge_index_list else torch.empty((2, 0), dtype=torch.long),
            'eligible_ops'  : eligible_ops_indices,
            'eligible_ops_details': eligible_ops_details,
            # 수정된 machine_features
            'm_features': torch.tensor(machine_features_list, dtype=torch.float),
            'om_features'   : torch.tensor(op_machine_pairs_features_list, dtype=torch.float),
            'num_nodes'     : self.total_ops,
            # op_histories는 Agent2에서 더 이상 사용하지 않지만, 다른 로직을 위해 유지
            'op_histories'  : [m.busy_intervals for m in self.machines]
        }

    def step(self, action, is_validation=False):
        current_state_dict = self._get_state()
        prev_makespan = self._calculate_makespan()
        op_global_idx, machine_idx = action

        job_id, op_id = self.op_map_rev[op_global_idx]
        op = self.jobs[job_id].ops[op_id]

        if machine_idx not in op.machine_map:
            return current_state_dict, -1000.0, self._get_state(), True
        
        machine = self.machines[machine_idx]
        processing_time = op.machine_map[machine_idx]

        if op.id > 0:
            prev_op = self.jobs[job_id].ops[op.id - 1]
            if not prev_op.is_scheduled:
                return current_state_dict, -1000.0, self._get_state(), True
            prev_op_comp_time = prev_op.completion_time
            travel_time = 0
            prev_machine_idx = prev_op.scheduled_machine
            if prev_machine_idx is not None and prev_machine_idx != machine_idx:
                travel_time = self.travel_times[prev_machine_idx][machine_idx]
            ready_time = prev_op_comp_time + travel_time
        else:
            ready_time = 0
        
        start_time = self._find_first_fit_for_machine(machine, ready_time, processing_time)
        completion_time = start_time + processing_time
        
        op.is_scheduled, op.start_time, op.completion_time = True, start_time, completion_time
        op.scheduled_machine = machine_idx
        if is_validation:
            if op.id >0:
                print(f"이전 공정 종료 시각: {prev_op_comp_time}")
                print(f"이동 시각: {travel_time}")
            print(f"선행 공정 준비 시각: {ready_time}") # 자격 목록 함께 출력
            print(f"장비 스케줄: {machine.busy_intervals}") # 자격 목록 함께 출력

        machine.available_time = max(machine.available_time, completion_time)
        machine.busy_intervals.append((start_time, completion_time, op.job_id, op.id))
        machine.busy_intervals.sort(key=lambda x: x[0])
        
        placement_efficiency = prev_makespan - start_time
        if placement_efficiency < 0: placement_efficiency = 0
        placement_reward = placement_efficiency / (processing_time + 1e-8)
        new_makespan = self._calculate_makespan()
        penalty = new_makespan - prev_makespan
        reward = placement_reward - penalty

        # Validation에서만 수행
        if is_validation:
            log_message = (f"Action(J{job_id}-O{op_id}, M{machine_idx}) | "
                        f"Placement Reward: {placement_reward:6.2f}, Penalty: {penalty:6.2f}, Final Reward: {reward:6.2f}")
            print(f"Job id: {job_id}, op_id: {op_id}") # 자격 목록 함께 출력
            print(f"start_time: {start_time}") # 자격 목록 함께 출력
            print(f"Completion time: {completion_time}") # 자격 목록 함께 출력
            print(f"Machine index: {machine_idx}") # 자격 목록 함께 출력
            print(f"프로세싱 시간: {processing_time}") # 자격 목록 함께 출력
            
            logger.debug(log_message)

        # --- 수정된 부분 ---
        # self.ops -> job.ops 로 수정하여 올바른 객체에 접근하도록 변경
        done = all(op.is_scheduled for job in self.jobs for op in job.ops)
        # --- 수정 끝 ---
        
        next_state_dict = self._get_state()
        return current_state_dict, reward, next_state_dict, done

    def _find_first_fit_for_machine(self, machine, ready_time, proc_time):
        # self: FJSPEnv 환경 객체
        # machine: 작업을 할당할 Machine 객체
        # ready_time: 이 작업이 시작될 수 있는 가장 이른 시간 (선행 작업 종료 시간 + 이동 시간)
        # proc_time: 이 기계에서 이 작업을 처리하는 데 걸리는 시간

        # 기계의 작업 이력(busy_intervals)이 비어 있는지 확인합니다.
        if not machine.busy_intervals:
            # 기계가 비어있다면, 작업이 준비되는 즉시(ready_time) 시작할 수 있습니다.
            return ready_time

        # 첫 번째 예정된 작업 시작 시간 전에 현재 작업을 완료할 수 있는지 확인합니다.
        if ready_time + proc_time <= machine.busy_intervals[0][0]:
            # 가능하다면, 작업이 준비된 시간(ready_time)에 시작합니다.
            return ready_time

        # 기존에 예약된 작업들 사이의 유휴 시간(idle time)을 탐색하기 위해 루프를 실행합니다.
        for i in range(len(machine.busy_intervals) - 1):
            # i번째 작업의 종료 시간이 유휴 시간의 시작점(idle_start)입니다.
            idle_start = machine.busy_intervals[i][1]
            # i+1번째 작업의 시작 시간이 유휴 시간의 종료점(idle_end)입니다.
            idle_end = machine.busy_intervals[i+1][0]

            # 실제 시작 후보 시간은 '유휴 시간 시작'과 '작업 준비 완료 시간' 중 더 늦은 시간입니다.
            potential_start = max(idle_start, ready_time)
            
            # 후보 시간에 시작해도 유휴 시간이 끝나기 전에 완료되는지 확인합니다.
            if potential_start + proc_time <= idle_end:
                # 가능하다면, 찾은 시작 시간을 즉시 반환합니다.
                return potential_start

        # 중간에 끼워넣을 공간이 없다면, 기존 모든 작업이 끝난 후에 작업을 시작합니다.
        # '기계의 마지막 작업 종료 시간'과 '작업 준비 완료 시간' 중 더 늦은 시간을 반환합니다.
        return max(machine.available_time, ready_time)

    def _calculate_makespan(self):
        max_completion_time = 0
        for job in self.jobs:
            for op in job.ops:
                if op.is_scheduled:
                    max_completion_time = max(max_completion_time, op.completion_time)
        return max_completion_time