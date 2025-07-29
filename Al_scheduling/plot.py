import matplotlib.pyplot as plt
import re
import numpy as np

def create_graphs_from_log(log_file_path, output_image_path):
    """
    훈련 로그 파일을 읽어 4개의 비교 그래프를 생성하고 이미지 파일로 저장합니다.
    """
    log_content = None
    # 여러 인코딩 방식을 시도하여 파일 읽기
    encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
    
    for encoding in encodings_to_try:
        try:
            with open(log_file_path, 'r', encoding=encoding) as f:
                log_content = f.read()
            print(f"'{log_file_path}' 파일을 '{encoding}' 인코딩으로 성공적으로 읽었습니다.")
            break 
        except UnicodeDecodeError:
            continue
    
    if log_content is None:
        print(f"오류: 모든 인코딩 방식으로 '{log_file_path}' 파일을 읽는 데 실패했습니다.")
        return

    # 데이터를 저장할 딕셔너리 초기화
    metrics = {
        'steps': [], 'train_makespans': [], 'val_makespans': [],
        'd5qn_losses': [], 'q_values': [], 'td_errors': [],
        'sac_critic_losses': [], 'sac_actor_losses': [], 'alpha_values': [], 'entropies': []
    }

    # 로그 블록을 한 번에 파싱하는 정규 표현식
    pattern = re.compile(
        r"스텝\s*(\d+):\s*평균 훈련 Makespan:\s*([\d.-]+),\s*평균 검증 Makespan:\s*([\d.-]+)\s*"
        r"D5QN \(Loss:\s*([\d.-]+),\s*Q-Value:\s*([\d.-]+),\s*TD-Error:\s*([\d.-]+)\)\s*"
        r"SAC \(Critic Loss:\s*([\d.-]+),\s*Actor Loss:\s*([\d.-]+),\s*Alpha:\s*([\d.-]+),\s*Entropy:\s*([\d.-]+)\)",
        re.DOTALL
    )
    
    for match in pattern.finditer(log_content):
        metrics['steps'].append(int(match.group(1)))
        metrics['train_makespans'].append(float(match.group(2)))
        metrics['val_makespans'].append(float(match.group(3)))
        metrics['d5qn_losses'].append(float(match.group(4)))
        metrics['q_values'].append(float(match.group(5)))
        metrics['td_errors'].append(float(match.group(6)))
        metrics['sac_critic_losses'].append(float(match.group(7)))
        metrics['sac_actor_losses'].append(float(match.group(8)))
        metrics['alpha_values'].append(float(match.group(9)))
        metrics['entropies'].append(float(match.group(10)))
            
    print(f"총 {len(metrics['steps'])}개의 로그 포인트를 파싱했습니다.")

    if not metrics['steps']:
        print("경고: 로그에서 유효한 데이터를 찾지 못했습니다. 그래프를 생성할 수 없습니다.")
        return

    # --- 2x2 그리드에 그래프 생성 ---
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Training Progress Analysis', fontsize=24)

    # 1. Makespan 비교
    ax1 = axs[0, 0]
    ax1.plot(metrics['steps'], metrics['train_makespans'], marker='.', linestyle='-', color='g', label='Train Makespan')
    ax1.plot(metrics['steps'], metrics['val_makespans'], marker='.', linestyle='-', color='r', label='Validation Makespan')
    ax1.set_title('Train vs Validation Makespan', fontsize=14)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Makespan', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. SAC Losses 비교
    ax2 = axs[0, 1]
    ax2.plot(metrics['steps'], metrics['sac_critic_losses'], marker='.', linestyle='-', color='purple', label='SAC Critic Loss')
    ax2.plot(metrics['steps'], metrics['sac_actor_losses'], marker='.', linestyle='-', color='orange', label='SAC Actor Loss')
    ax2.set_title('SAC Agent Losses', fontsize=14)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. D5QN Loss (단독 그래프)
    ax3 = axs[1, 0]
    ax3.plot(metrics['steps'], metrics['d5qn_losses'], marker='.', linestyle='-', color='b', label='D5QN Loss')
    ax3.set_title('D5QN Loss', fontsize=14)
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    # 4. SAC 엔트로피 관련 지표 비교 (이중 Y축)
    ax4 = axs[1, 1]
    ax4_twin = ax4.twinx()
    p3, = ax4.plot(metrics['steps'], metrics['alpha_values'], marker='.', linestyle='-', color='brown', label='Alpha Value')
    p4, = ax4_twin.plot(metrics['steps'], metrics['entropies'], marker='.', linestyle='--', color='pink', label='Policy Entropy')
    ax4.set_title('SAC Alpha & Policy Entropy', fontsize=14)
    ax4.set_xlabel('Step', fontsize=12)
    ax4.set_ylabel('Alpha (Entropy Coeff.)', fontsize=12, color='brown')
    ax4_twin.set_ylabel('Policy Entropy', fontsize=12, color='pink')
    ax4.tick_params(axis='y', labelcolor='brown')
    ax4_twin.tick_params(axis='y', labelcolor='pink')
    ax4.legend(handles=[p3, p4], loc='upper left')
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_image_path)
    print(f"그래프가 '{output_image_path}' 파일로 저장되었습니다.")


if __name__ == '__main__':
    # 분석할 로그 파일 이름을 지정하세요.
    log_filename = 'training.txt'
    output_filename = 'training_graphs_revised.png'
    create_graphs_from_log(log_filename, output_filename)