import matplotlib.pyplot as plt
import re
import numpy as np

def create_graphs_from_log(log_file_path, output_image_path):
    """
    훈련 로그 파일을 읽어 4개의 분석 그래프를 생성하고 이미지 파일로 저장합니다.
    """
    try:
        # cp949 인코딩으로 파일 읽기
        with open(log_file_path, 'r', encoding='cp949') as f:
            log_content = f.read()
        print(f"'{log_file_path}' 파일을 성공적으로 읽었습니다.")
    except FileNotFoundError:
        print(f"오류: '{log_file_path}' 파일을 찾을 수 없습니다. 파일이 스크립트와 같은 위치에 있는지 확인해주세요.")
        return
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    steps = []
    train_makespans = []
    val_makespans = []
    d5qn_losses = []
    sac_critic_losses = []
    sac_actor_losses = []

    # --- 정규 표현식 수정 ---
    # '스텝' 블록 전체를 찾는 패턴
    block_pattern = re.compile(r"스텝 (\d+):(.+?)(?=스텝 \d+:|\Z)", re.DOTALL)
    
    # 각 블록 내부에서 개별 항목을 찾는 패턴
    train_makespan_pattern = re.compile(r"평균 훈련 Makespan: ([\d.]+)")
    val_makespan_pattern = re.compile(r"평균 검증 Makespan: ([\d.]+)")
    d5qn_loss_pattern = re.compile(r"평균 D5QN Loss: ([\d.]+)")
    sac_critic_loss_pattern = re.compile(r"평균 SAC Critic Loss: ([\d.]+)")
    sac_actor_loss_pattern = re.compile(r"평균 SAC Actor Loss: ([\d.]+)")
    
    for match in block_pattern.finditer(log_content):
        step = int(match.group(1))
        block_text = match.group(2)
        
        train_match = train_makespan_pattern.search(block_text)
        val_match = val_makespan_pattern.search(block_text)
        d5qn_match = d5qn_loss_pattern.search(block_text)
        sac_critic_match = sac_critic_loss_pattern.search(block_text)
        sac_actor_match = sac_actor_loss_pattern.search(block_text)
        
        # 모든 데이터가 존재할 때만 리스트에 추가
        if all((train_match, val_match, d5qn_match, sac_critic_match, sac_actor_match)):
            steps.append(step)
            train_makespans.append(float(train_match.group(1)))
            val_makespans.append(float(val_match.group(1)))
            d5qn_losses.append(float(d5qn_match.group(1)))
            sac_critic_losses.append(float(sac_critic_match.group(1)))
            sac_actor_losses.append(float(sac_actor_match.group(1)))
    
    print(f"총 {len(steps)}개의 로그 포인트를 파싱했습니다.")

    if not steps:
        print("경고: 로그에서 유효한 데이터를 찾지 못했습니다. 그래프를 생성할 수 없습니다.")
        return

    # --- 그래프 생성 ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Progress Analysis', fontsize=20)

    # 1. D5QN Loss
    axs[0, 0].plot(steps, d5qn_losses, marker='.', linestyle='-', color='blue')
    axs[0, 0].set_title('D5QN Loss per Iteration', fontsize=14)
    axs[0, 0].set_xlabel('Step', fontsize=12)
    axs[0, 0].set_ylabel('Loss', fontsize=12)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 2. Makespan
    axs[0, 1].plot(steps, train_makespans, marker='.', linestyle='-', color='green', label='Train Makespan (avg)')
    axs[0, 1].plot(steps, val_makespans, marker='.', linestyle='-', color='red', label='Validation Makespan')
    axs[0, 1].set_title('Makespan per Iteration', fontsize=14)
    axs[0, 1].set_xlabel('Step', fontsize=12)
    axs[0, 1].set_ylabel('Makespan', fontsize=12)
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. SAC Critic Loss
    axs[1, 0].plot(steps, sac_critic_losses, marker='.', linestyle='-', color='purple')
    axs[1, 0].set_title('SAC Critic Loss per Iteration', fontsize=14)
    axs[1, 0].set_xlabel('Step', fontsize=12)
    axs[1, 0].set_ylabel('Loss', fontsize=12)
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # 4. SAC Actor Loss
    axs[1, 1].plot(steps, sac_actor_losses, marker='.', linestyle='-', color='orange')
    axs[1, 1].set_title('SAC Actor Loss per Iteration', fontsize=14)
    axs[1, 1].set_xlabel('Step', fontsize=12)
    axs[1, 1].set_ylabel('Loss', fontsize=12)
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_image_path)
    print(f"그래프가 '{output_image_path}' 파일로 저장되었습니다.")


if __name__ == '__main__':
    log_filename = 'train_20250620_202009.log'
    output_filename = 'training_graphs.png'
    create_graphs_from_log(log_filename, output_filename)