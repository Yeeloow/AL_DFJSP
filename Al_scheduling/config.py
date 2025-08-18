# config.py (최종 수정본)

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FJSP 환경 설정
N_JOBS = 5
N_MACHINES = 7
MAX_OPS_PER_JOB = 5

# 에이전트 1 (SAC) 하이퍼파라미터
GAT_N_HEADS = 4
GAT_OUT_DIM = 8
GAT_DROPOUT = 0.6
SAC_LR = 3e-6
SAC_TAU = 0.001
SAC_GAMMA = 0.99 # SAC의 감마도 일관성을 위해 조정할 수 있습니다.

# 에이전트 2 (D5QN) 하이퍼파라미터
TRANSFORMER_N_HEADS = 4
TRANSFORMER_N_LAYERS = 2
TRANSFORMER_HIDDEN_SIZE = 32
D5QN_LR = 3e-3 # 학습 안정성을 위해 학습률을 조금 낮추는 것을 고려해볼 수 있습니다.
D5QN_EPSILON = 0.8
# --- ▼▼▼ 수정된 부분 ▼▼▼ ---
D5QN_GAMMA = 0.99 # 1.0에서 0.99로 조정하여 타겟 값 안정화
# --- ▲▲▲ 수정 끝 ▲▲▲ ---

# 훈련 설정
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000
TRAINING_EPISODES = 10000
VALIDATION_EPISODES = 100