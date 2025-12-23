import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    random.seed(seed)  # Python 내장 random 모듈
    np.random.seed(seed)  # NumPy 난수 생성기
    torch.manual_seed(seed)  # PyTorch CPU 연산의 난수 생성기
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # GPU 연산 난수 고정 (단일 GPU)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU 환경을 위한 seed 고정
    
    # CUDNN 관련 옵션 (아래 설정은 속도를 낮출 수 있지만 재현성을 보장함)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m}m {s}s"

def compute_energy(logits, T=1.0):
    """Energy calculation for Code 1 style"""
    return T * torch.logsumexp(logits / T, dim=1)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step/total_steps*np.pi))

def save_stage1_checkpoint(model, optimizer, scheduler, epoch, save_path):
    """Stage1 체크포인트 저장"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.base_optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, save_path)
    print(f"Stage1 checkpoint saved at epoch {epoch}: {save_path}")

def load_stage1_checkpoint(device, model, optimizer, scheduler, checkpoint_path):
    """Stage1 체크포인트 로드"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_from_epoch = checkpoint['epoch'] + 1
        print(f"Stage1 checkpoint loaded. Resuming from epoch {start_from_epoch}")
        return start_from_epoch
    else:
        print("No Stage1 checkpoint found. Starting from epoch 1")
        return 1
    

# utils/tools.py

# utils/tools.py

import math
from typing import Optional, Dict, Tuple

def get_rho_ood(
    mode: str,
    epoch: int,
    total_epochs: int,
    start_epoch: int = 1,
    rho_min: float = 0.0,
    rho_max: float = 1.0,
    val_metric: Optional[float] = None,
    state: Optional[Dict] = None,
    ema_beta: float = 0.9,
    auroc_min: float = 0.5,
    auroc_max: float = 1.0,
) -> Tuple[float, Dict]:
    """
    Unified rho_ood scheduler (0~1 범위).

    Parameters
    ----------
    mode : str
        스케줄 타입:
          - 'const'
          - 'linear_inc'   : rho_min → rho_max (증가)
          - 'linear_dec'   : rho_max → rho_min (감소)
          - 'cosine'
          - 'auroc_adapt'  : val AUROC 기반 적응 스케줄
    epoch : int
        현재 epoch (1-based).
    total_epochs : int
        총 epoch 수.
    start_epoch : int, default=1
        Stage2 시작 epoch.
    rho_min, rho_max : float
        rho 하한/상한 (자동으로 [0,1]으로 클램프됨).
    last_loss : float, optional
        'ema_loss' 모드에서 사용할 최근 loss 값.
    val_metric : float, optional
        'auroc_adapt' 모드에서 사용할 validation AUROC (ID vs vOOD).
    state : dict, optional
        EMA 상태 등을 저장하는 딕셔너리. 호출자 쪽에서 유지하면서 넘겨주면 됨.
        사용되는 키:
          - 'auroc_ema'
    ema_beta : float
        EMA decay 계수 (0~1).
    loss_min, loss_max : float
        'ema_loss' 모드에서 loss 정규화 범위.
    auroc_min, auroc_max : float
        'auroc_adapt' 모드에서 AUROC 정규화 범위.

    Returns
    -------
    rho : float
        현재 epoch에서 사용할 rho 값 (0~1).
    state : dict
        업데이트된 상태 (EMA 등).
    """
    if state is None:
        state = {}

    # ---- rho 범위 강제 [0, 1] ----
    rho_min = max(0.0, min(1.0, rho_min))
    rho_max = max(0.0, min(1.0, rho_max))
    if rho_max < rho_min:
        rho_min, rho_max = rho_max, rho_min  # swap

    # ---- Stage1: warm-up 구간에서는 dynamic scheduling 최소화 ----
    if epoch <= start_epoch:
        if mode == "const":
            rho = (rho_min + rho_max) / 2.0
        else:
            rho = rho_min
        return float(rho), state

    # ---- Stage2 진행률 t ∈ [0, 1] ----
    denom = max(1, total_epochs - start_epoch)
    t = (epoch - start_epoch) / denom
    t = max(0.0, min(1.0, t))

    # --------------------------
    # 1) Constant
    # --------------------------
    if mode == "const":
        rho = (rho_min + rho_max) / 2.0

    # --------------------------
    # 2) Linear Increase: rho_min → rho_max
    # --------------------------
    elif mode == "linear_inc":
        rho = rho_min + (rho_max - rho_min) * t

    # --------------------------
    # 3) Linear Decrease: rho_max → rho_min
    # --------------------------
    elif mode == "linear_dec":
        rho = rho_max - (rho_max - rho_min) * t

    # --------------------------
    # 4) Cosine annealing (대략 max→min)
    # --------------------------
    elif mode == "cosine":
        cos_term = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 → 0
        rho = rho_min + (rho_max - rho_min) * cos_term
    
    elif mode == "auroc_ma":
        if val_metric is None:
            rho = (rho_min + rho_max) / 2.0
        else:
            auroc_cur = float(val_metric)

            if auroc_max <= auroc_min:
                auroc_max = auroc_min + 1e-3
            z = (auroc_cur - auroc_min) / (auroc_max - auroc_min)
            z = max(0.0, min(1.0, z))

            rho = rho_max - (rho_max - rho_min) * z

    # --------------------------
    # 7) Energy Metric (SNR/Gap) 기반 [NEW]
    #    val_metric 인자에 SNR 값이 들어온다고 가정
    # --------------------------
    elif mode == "energy_metric":
        """
        Energy Metric (SNR) 기반 adaptive rho 스케줄 (target 값 없음 버전)

        - val_metric: 현재 epoch에서 계산된 SNR 값
            SNR_t = (mu_ID - mu_vOOD) / (std_ID + std_vOOD + eps)

        - state 안에 저장되는 값:
            state["snr_mean"] : SNR EMA 평균 μ_t
            state["snr_var"]  : SNR EMA 분산 σ_t^2 (대략적인 scale 용)
        
        로직:
            1) 현재 SNR을 이용해 EMA mean/var 업데이트
            2) z_t = (SNR_t - μ_t) / (σ_t + eps)
            3) u_t = 0.5 - 0.5 * tanh(alpha * z_t)   # [0,1] 범위
            4) rho_t = rho_min + (rho_max - rho_min) * u_t
        """

        # 1) 현재 metric (SNR). 없으면 그냥 이전 rho 유지
        if val_metric is None:
            prev_rho = state.get("rho", (rho_min + rho_max) / 2.0)
            rho = prev_rho
        else:
            current_snr = float(val_metric)

            # 2) EMA mean / var 업데이트
            # 초기화: 첫 호출이면 mean을 현재 값으로 시작
            mu_prev = state.get("snr_mean", current_snr)
            var_prev = state.get("snr_var", 0.0)

            beta = ema_beta  # 0.9 정도
            # EMA mean
            mu = beta * mu_prev + (1.0 - beta) * current_snr
            # EMA variance (간단한 EMA 기반 분산 추정)
            delta = current_snr - mu_prev
            var = beta * var_prev + (1.0 - beta) * delta * delta

            state["snr_mean"] = mu
            state["snr_var"] = var

            # 3) z-score 계산
            snr_std = math.sqrt(var) + 1e-6
            z = (current_snr - mu) / snr_std   # 표준화된 SNR

            # 4) z를 [0,1]로 squash
            alpha = 1.0  # tanh 감도 (크게 할수록 민감)
            u = 0.5 - 0.5 * math.tanh(alpha * z)
            #   z >> 0 (easy)  -> u ~ 0
            #   z << 0 (hard)  -> u ~ 1
            # 5) 최종 rho
            rho = rho_min + (rho_max - rho_min) * u

        rho = max(rho_min, min(rho_max, rho))
        state["rho"] = rho

    else:
        # 알 수 없는 mode → 중간값으로 fallback
        rho = (rho_min + rho_max) / 2.0

    rho = max(0.0, min(1.0, float(rho)))
    return rho, state

from sklearn.metrics import roc_auc_score

def compute_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    sklearn 기반 AUROC 계산
    - tie 처리 정확
    - 논문/리포팅용과 100% 일치
    """
    scores = scores.detach().cpu().numpy().reshape(-1)
    labels = labels.detach().cpu().numpy().reshape(-1)

    # 한 클래스만 있을 경우 AUROC 정의 불가 → 0.5 반환
    if len(np.unique(labels)) < 2:
        return 0.5

    return float(roc_auc_score(labels, scores))

@torch.no_grad()
def compute_binary_metrics_on_loader(model, id_loader, vood_loader, args, device, max_batches=None):
    """
    ID: id_loader(images, labels)
    vOOD: vood_loader(feat_pert)
    returns: (snr, auroc)
    """
    model.eval()

    id_energy_list = []
    for b, (images, labels) in enumerate(id_loader):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        e = compute_energy(logits, T=args.temperature)
        id_energy_list.append(e.detach().cpu())
        if max_batches is not None and (b + 1) >= max_batches:
            break

    vood_energy_list = []
    for b, (feat_pert,) in enumerate(vood_loader):
        feat_pert = feat_pert.to(device, non_blocking=True)
        v_logits = model.fc(feat_pert)
        e = compute_energy(v_logits, T=args.temperature)
        vood_energy_list.append(e.detach().cpu())
        if max_batches is not None and (b + 1) >= max_batches:
            break

    id_energies = torch.cat(id_energy_list, dim=0).float()
    vood_energies = torch.cat(vood_energy_list, dim=0).float()

    mu_id, std_id = id_energies.mean(), id_energies.std()
    mu_vood, std_vood = vood_energies.mean(), vood_energies.std()
    eps = 1e-6
    snr = ((mu_id - mu_vood) / (std_id + std_vood + eps)).item()

    scores = torch.cat([id_energies, vood_energies], dim=0)
    labels = torch.cat([
        torch.ones_like(id_energies, dtype=torch.long),
        torch.zeros_like(vood_energies, dtype=torch.long)
    ], dim=0)
    auroc = compute_auroc(scores, labels)

    model.train()
    return snr, auroc, mu_id.item(), mu_vood.item()

import random
import matplotlib.pyplot as plt
# ---- 실행 및 시각화 코드 ----
if __name__ == "__main__":
    # 공통 설정
    total_epochs = 100
    start_epoch = 40
    rho_min = 0.0
    rho_max = 1.0
    auroc_min = 0.5
    auroc_max = 1.0
    ema_beta = 0.9

    # ---------------------------------------------------------
    # 1. Time-based Schedulers (Const, Linear, Cosine)
    # ---------------------------------------------------------
    time_modes = ["const", "linear_inc", "linear_dec", "cosine"]
    history_time = {m: {"epochs": [], "rho": []} for m in time_modes}

    for epoch in range(1, total_epochs + 1):
        for mode in time_modes:
            rho, _ = get_rho_ood(mode, epoch, total_epochs, start_epoch, rho_min, rho_max)
            history_time[mode]["epochs"].append(epoch)
            history_time[mode]["rho"].append(rho)

    plt.figure(figsize=(10, 6))
    for mode in time_modes:
        plt.plot(history_time[mode]["epochs"], history_time[mode]["rho"], label=mode, linewidth=2)
    
    plt.axvline(x=start_epoch, color='gray', linestyle='--', label="Start Epoch (40)")
    plt.title("[1] Time-based Schedulers (Epoch Dependent)")
    plt.xlabel("Epoch")
    plt.ylabel("Rho")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show() # 첫 번째 그래프 출력

    # ---------------------------------------------------------
    # 2. AUROC Adaptive Scheduler
    # ---------------------------------------------------------
    mode_auroc = "auroc_ma"
    hist_auroc = {"epochs": [], "rho": [], "metric": []}
    state_auroc = {}

    for epoch in range(1, total_epochs + 1):
        # Simulation: AUROC가 0.5에서 시작해 0.95까지 점진적 상승 + 노이즈
        fake_auroc = 0.3 + 0.45 * (epoch / total_epochs) + random.uniform(-0.02, 0.02)
        fake_auroc = max(0.0, min(1.0, fake_auroc))
        
        rho, state_auroc = get_rho_ood(
            mode_auroc, epoch, total_epochs, start_epoch, rho_min, rho_max,
            val_metric=fake_auroc, state=state_auroc, auroc_min=auroc_min, auroc_max=auroc_max
        )
        
        hist_auroc["epochs"].append(epoch)
        hist_auroc["rho"].append(rho)
        hist_auroc["metric"].append(fake_auroc)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(hist_auroc["epochs"], hist_auroc["rho"], 'b-', label="Rho (Calculated)", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Rho", color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(hist_auroc["epochs"], hist_auroc["metric"], 'r--', label="Val AUROC (Input)", alpha=0.7)
    ax2.set_ylabel("Validation AUROC", color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Start Epoch 표시
    ax1.axvline(x=start_epoch, color='gray', linestyle='--', alpha=0.5, label="Start Epoch")
    
    plt.title("[2] AUROC Adaptive (Inverse Relation)")
    # 레전드 합치기
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left')
    
    plt.tight_layout()
    plt.show() # 두 번째 그래프 출력

    # ---------------------------------------------------------
    # 3. Energy Metric (SNR) Scheduler (Updated Logic)
    # ---------------------------------------------------------
    mode_energy = "energy_metric"
    hist_energy = {"epochs": [], "rho": [], "metric": [], "ema_mean": []}
    state_energy = {}

    for epoch in range(1, total_epochs + 1):
        # Simulation: SNR이 평균 1.0 주변에서 사인파 형태로 진동 + 약간의 상승 트렌드
        # 진동을 주어야 Z-score가 변화하며 rho가 바뀝니다.
        trend = 0.005 * epoch
        noise = random.uniform(-0.1, 0.1)
        oscillation = 0.5 * math.sin(epoch / 5.0) # 주기적 변동
        fake_snr = 0.1 + trend + oscillation + noise
        
        rho, state_energy = get_rho_ood(
            mode_energy, epoch, total_epochs, start_epoch, rho_min, rho_max,
            val_metric=fake_snr, state=state_energy, ema_beta=ema_beta
        )
        
        hist_energy["epochs"].append(epoch)
        hist_energy["rho"].append(rho)
        hist_energy["metric"].append(fake_snr)
        # 내부 상태 확인용 (EMA Mean)
        hist_energy["ema_mean"].append(state_energy.get("snr_mean", fake_snr))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Rho Plot
    ax1.plot(hist_energy["epochs"], hist_energy["rho"], 'g-', label="Rho (Adaptive)", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Rho", color='g', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True, alpha=0.3)

    # Metric Plot
    ax2 = ax1.twinx()
    ax2.plot(hist_energy["epochs"], hist_energy["metric"], color='orange', linestyle='--', label="SNR (Input)", alpha=0.6)
    ax2.plot(hist_energy["epochs"], hist_energy["ema_mean"], color='black', linestyle=':', label="SNR EMA (Mean)", alpha=0.4)
    ax2.set_ylabel("Energy Metric (SNR)", color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')

    # Start Epoch
    ax1.axvline(x=start_epoch, color='gray', linestyle='--', alpha=0.5, label="Start Epoch")

    plt.title("[3] Energy Metric (Relative Z-score Adaptation)")
    
    # 레전드 합치기
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.show() # 세 번째 그래프 출력