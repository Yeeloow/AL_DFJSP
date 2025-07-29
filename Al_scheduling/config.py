# config.py

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FJSP 환경 설정
N_JOBS = 15
N_MACHINES = 10
MAX_OPS_PER_JOB = 10 # 작업당 최대 공정 수

# N_JOBS = 2
# N_MACHINES = 2
# MAX_OPS_PER_JOB = 10 # 작업당 최대 공정 수

# 에이전트 1 (SAC) 하이퍼파라미터
GAT_N_HEADS = 4
GAT_OUT_DIM = 8
GAT_DROPOUT = 0.6
SAC_LR = 3e-5
SAC_TAU = 0.005
SAC_DECAY_RATIO = 0.9

# 에이전트 2 (D5QN) 하이퍼파라미터
TRANSFORMER_N_HEADS = 4
TRANSFORMER_N_LAYERS = 2
TRANSFORMER_HIDDEN_SIZE = 32
D5QN_LR = 3e-4
D5QN_EPSILON = 0.8
D5QN_GAMMA = 1.0

# 훈련 설정
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000
TRAINING_EPISODES = 10000
VALIDATION_EPISODES = 100