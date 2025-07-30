# replay_buffer.py

import numpy as np
import random
import torch

# PER을 효율적으로 구현하기 위한 SumTree 자료구조
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # 우선순위 강도 (0: uniform, 1: full priority)
        self.beta = beta    # 중요도 샘플링(IS) 가중치 보정 강도
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 0.01 # TD-error가 0이 되는 것을 방지하기 위한 작은 값
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        # 새로운 경험은 최대 우선순위를 부여하여 최소 한 번은 샘플링되도록 보장
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            if data is not 0:
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)
        if not batch:
            return None  # 유효한 배치가 없으면 None을 반환
        sampling_probabilities = np.array(priorities) / self.tree.total()
        # 중요도 샘플링(IS) 가중치 계산
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones, idxs, torch.from_numpy(is_weights).float()

    def update_priorities(self, batch_indices, td_errors):
        priorities = np.abs(td_errors) + self.epsilon
        clipped_priorities = np.minimum(priorities, self.max_priority)
        
        for idx, p in zip(batch_indices, clipped_priorities):
            self.tree.update(idx, p ** self.alpha)
            
    def __len__(self):
        return self.tree.n_entries