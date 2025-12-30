"""Sharpness-Aware Minimization based Outlier Synthesis (SOS) Refactored"""
"""Rho scheduling 추가"""
"""validation 추가"""
import time
import os
import numpy as np
import torch
import torch.nn as nn
import wandb
import json
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import TensorDataset, DataLoader, Subset

# Import Models
from model.cifar_resnet import *
from model.WideResNet import WideResNet
from model.sephead import SeparationHead
from model.cifar_densenet import DenseNet3
# Import Utils
from utils.dataloader import get_dataloader
from utils.tools import (
    set_seed,
    cosine_annealing,
    save_stage1_checkpoint,
    load_stage1_checkpoint,
    format_elapsed,
    get_rho_ood,
    compute_auroc,
    compute_binary_metrics_on_loader
)
from utils.sam import SAM
from utils.trainer import evaluate

# --- Refactored Modules ---
from utils.config import parse_args
from utils.generator import extract_all_sam_features_for_epoch
from utils.trainer import train_epoch

def main():
    run_start_time = time.time()
    args = parse_args()

    # ---------------------------------------------------------
    # Experiment Name & Setup
    # ---------------------------------------------------------
    if args.model_name == "WideResNet":
        model_aka = "wrn"
    elif args.model_name == "ResNet":
        model_aka = f"resnet{18 if args.base_data=='cifar10' else 34}"
    elif args.model_name == "DenseNet":
        model_aka = "densenet"

    if args.exp_name is None:
        # 자동 이름 생성: GenMode_TrainMode_Details 형태
        args.exp_name =(f"{args.ood_gen_mode}_{args.ood_train_mode}_"
            f"s{args.seed}_{model_aka}_mode_{args.rho_ood_mode}_"
            f"rho{args.rho_ood_min}-{args.rho_ood_max}")
        
        if args.ood_train_mode == 'binary':
            args.exp_name += f"_E{args.lambda_energy}"
        else:
            if args.use_sep_loss: 
                args.exp_name += f"_sep{args.lambda_sep}"
            if args.use_kl_loss: 
                args.exp_name += f"_kl{args.lambda_kl}"
            if args.use_rank_loss: 
                args.exp_name += f"_rk{args.lambda_rank}"

    if args.use_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", f"SOS_rho_schd_{args.base_data}_{args.ood_gen_mode}\
                                   _{args.ood_train_mode}_S{args.seed}"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=args.exp_name,
            config=vars(args),
        )
        wandb.alert(
            title=f"Run started Dataset={args.base_data}",
            text=f"Experiment {args.exp_name} has started. | CUDA: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}"
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories
    stage1_base_dir = f"/home/thoon1999/SOS/{args.ood_gen_mode}_{args.ood_train_mode}_E{args.epochs}_seed{args.seed}/stage1_checkpoints"
    stage1_model_path = f"{args.ood_gen_mode}_{args.ood_train_mode}_Stage1_{args.base_data}_{args.model_name}_ep{args.start_epoch}.pth"
    stage1_ckpt_path = os.path.join(stage1_base_dir, stage1_model_path)
    os.makedirs(stage1_base_dir, exist_ok=True)

    data_specific_dir = f"{args.save_dir_base}/{args.base_data}"
    save_dir = f"{data_specific_dir}/feat_{args.exp_name}"
    ckpt_dir = f"{data_specific_dir}/models_{args.exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Data Setup
    # ---------------------------------------------------------
    train_loader = get_dataloader(device, args.base_data, args.base_data, args.batch_size, phase="train")
    val_loader = None

    if args.use_val:
        dataset = train_loader.dataset
        targets = np.array(dataset.targets)
        # Stratified split
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.1,   # 나머지 10%
            random_state=args.seed
        )
        train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=True,
            prefetch_factor=4
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=False
        )

    test_loader = get_dataloader(device, args.base_data, args.base_data, args.batch_size, phase="test")
    eval_loader = val_loader if (args.use_val and val_loader is not None) else test_loader

    num_classes = {"cifar10": 10, "cifar100": 100}[args.base_data]

    # ---------------------------------------------------------
    # Model & Modules Init
    # ---------------------------------------------------------
    if args.model_name == "ResNet":
        model = resnet18(num_classes=num_classes).to(device) if args.base_data == 'cifar10' else resnet34(num_classes=num_classes).to(device)

    elif args.model_name == "WideResNet":
        model = WideResNet(
            depth=args.depth, 
            widen_factor=args.widen_factor, 
            dropRate=args.drop_rate, 
            num_classes=num_classes
        ).to(device)
        
    elif args.model_name == "DenseNet":
        model = DenseNet3(
            depth=100,
            num_classes=num_classes,
            reduction=0.5,
            bottleneck=True
        ).to(device)

    energy_head = None
    sep_head = None

    if args.ood_train_mode == 'binary':
        energy_head = nn.Linear(1, 2).to(device)

    if args.ood_train_mode == 'regularization' and args.use_rank_loss:
        with torch.no_grad():
            _, df = model.forward_virtual(torch.zeros(1, 3, 32, 32).to(device))
        feat_dim = df.shape[1]
        sep_head = SeparationHead(feat_dim).to(device)

    # Optimizer
    params_to_opt = list(model.parameters())
    if energy_head is not None:
        params_to_opt += list(energy_head.parameters())
    if sep_head is not None:
        params_to_opt += list(sep_head.parameters())

    sam_optimizer = SAM(
        params_to_opt,
        base_optimizer=torch.optim.SGD,
        rho=args.rho, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        sam_optimizer.base_optimizer,
        lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader), 1.0, 1e-6 / args.lr)
    )

    criterion = nn.CrossEntropyLoss()
    resume_epoch = load_stage1_checkpoint(device, model, sam_optimizer, scheduler, stage1_ckpt_path)
    actual_start_epoch = resume_epoch

    if args.epochs <= 100:
        save_epochs = {41, 61, 81, 100}
    elif args.epochs <= 200:
        save_epochs = {81, 100, 140, 170, 200}
    else:
        save_epochs = {201, 250, 300, 400, args.epochs}

    # ---------------------------------------------------------
    # Main Loop
    # ---------------------------------------------------------
    crashed = False
    last_val_acc = 0
    rho_state = {}
    val_metric = 0.0

    try:
        for ep in range(actual_start_epoch, args.epochs + 1):
            
            # [수정됨] 1. 로그 데이터를 모을 빈 딕셔너리 생성 (Epoch 시작 시점)
            log_payload = {}
            log_payload["epoch"] = ep

            # ---------------------------------------------------------
            # 1. Rho Scheduling & Feature Extraction & Training
            # ---------------------------------------------------------
            if ep > args.start_epoch:
                print(f"\n[Epoch {ep}] Extracting vOOD ({args.ood_gen_mode})...")
                
                # Rho 스케줄링 계산
                rho_ood_current, rho_state = get_rho_ood(
                    mode=args.rho_ood_mode, 
                    epoch=ep,
                    total_epochs=args.epochs,
                    start_epoch=args.start_epoch,
                    rho_min=args.rho_ood_min,
                    rho_max=args.rho_ood_max,
                    val_metric=val_metric,      # 이전 epoch에서 업데이트된 val_metric 사용
                    state=rho_state,
                    ema_beta=0.9,               
                    auroc_min=0.5,            
                    auroc_max=1.0,
                )
                print(f" -> rho_ood_current = {rho_ood_current:.4f}")

                # Feature Extraction
                _, feat_pert, _ = extract_all_sam_features_for_epoch(
                    model, train_loader, device, args, criterion, rho_ood_current
                )

                # vOOD Split (Train/Val)
                N_total = feat_pert.size(0)
                if args.use_val:
                    # assert N_total >= 45000, ... (필요 시 주석 해제)
                    n_vood_val = 5000
                    n_vood_train = 40000

                    g = torch.Generator()
                    g.manual_seed(args.seed + ep) # 매 epoch마다 다른 셔플 적용

                    perm = torch.randperm(N_total, generator=g)
                    feat_pert_train = feat_pert[perm[:n_vood_train]]
                    feat_pert_val   = feat_pert[perm[n_vood_train:n_vood_train + n_vood_val]]
                else:
                    feat_pert_train = feat_pert
                    feat_pert_val = None

                # vOOD Train Loader 구성
                id_steps = len(train_loader)
                vood_train_bs = int(np.ceil(feat_pert_train.size(0) / id_steps))
                
                vood_train_dataset = TensorDataset(feat_pert_train)
                vood_loader = DataLoader(
                    vood_train_dataset,
                    batch_size=vood_train_bs,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True,
                    persistent_workers=True,
                    drop_last=False
                )

                # vOOD Val Loader 구성
                if args.use_val:
                    vood_val_loader = DataLoader(
                        TensorDataset(feat_pert_val),
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        persistent_workers=False,
                        drop_last=False
                    )
                else:
                    vood_val_loader = None
                
                # Train Execution
                loss, id_energy_all, vood_energy_all = train_epoch(
                    args, model, train_loader, ep, device, criterion, sam_optimizer, 
                    scheduler, save_epochs, save_dir, energy_head, sep_head, vood_loader=vood_loader
                )
                
            else:
                # Stage 1 (Start Epoch 이전)
                rho_ood_current = 0.0 
                loss, _, _ = train_epoch(
                    args, model, train_loader, ep, device, criterion, sam_optimizer, 
                    scheduler, save_epochs, save_dir, energy_head, sep_head
                )
            
            print(f"[Epoch {ep}] Loss: {loss:.4f} | Mode: {args.ood_gen_mode}/{args.ood_train_mode}")

            # [수정됨] 2. 기본 학습 정보를 log_payload에 담기 (즉시 전송 X)
            log_payload["train_loss"] = loss
            log_payload["rho_ood"] = rho_ood_current

            # ---------------------------------------------------------
            # 2. Metric Calculation (Binary Mode) & Scheduling Update
            # ---------------------------------------------------------
            if ep > args.start_epoch and args.ood_train_mode == 'binary':
                if args.use_val and (val_loader is not None):
                    # Validation Set 사용 시
                    current_snr, current_auroc, mu_id, mu_vood = compute_binary_metrics_on_loader(
                        model=model,
                        id_loader=val_loader,
                        vood_loader=vood_val_loader,
                        args=args,
                        device=device,
                        max_batches=None
                    )
                else:
                    # Train Set 통계 사용 시 (기존 방식)
                    if id_energy_all is None:
                        current_snr, current_auroc, mu_id, mu_vood = 0.0, 0.5, 0.0, 0.0
                    else:
                        id_energies = id_energy_all.detach().cpu().float()
                        vood_energies = vood_energy_all.detach().cpu().float()

                        mu_id, std_id = id_energies.mean(), id_energies.std()
                        mu_vood, std_vood = vood_energies.mean(), vood_energies.std()

                        eps = 1e-6
                        current_snr = ((mu_id - mu_vood) / (std_id + std_vood + eps)).item()

                        scores = torch.cat([id_energies, vood_energies], dim=0)
                        labels = torch.cat([
                            torch.ones_like(id_energies, dtype=torch.long),
                            torch.zeros_like(vood_energies, dtype=torch.long)
                        ], dim=0)
                        current_auroc = compute_auroc(scores, labels)

                        mu_id, mu_vood = mu_id.item(), mu_vood.item()

                print(f" -> [Metrics] SNR: {current_snr:.4f} | AUROC: {current_auroc:.4f}")

                # [수정됨] 3. 상세 Metric 정보를 log_payload에 통합 (즉시 전송 X)
                log_payload.update({
                    "monitor_snr": current_snr,
                    "monitor_auroc": current_auroc,
                    "mean_energy_id": mu_id,
                    "mean_energy_vood": mu_vood,
                })

                # [중요] 다음 epoch의 rho 계산을 위해 val_metric 업데이트
                if args.rho_ood_mode == "energy_metric":
                    val_metric = current_snr
                else:
                    val_metric = current_auroc

            # [수정됨] 4. 최종 스케줄링 기준 Metric 기록
            if args.rho_ood_mode == "energy_metric":
                log_payload["val_energy"] = val_metric
            else:
                log_payload["val_auroc"] = val_metric

            # ---------------------------------------------------------
            # 3. Checkpointing & Evaluation
            # ---------------------------------------------------------
            if ep == args.start_epoch:
                save_stage1_checkpoint(model, sam_optimizer, scheduler, ep, stage1_ckpt_path)
            
            if ep == args.epochs or ep in save_epochs:
                v_loss, acc = evaluate(model, test_loader, criterion, device)
                print(f" -> Val Acc: {acc:.2f}%")
                last_val_acc = float(acc)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_ep{ep}.pth"))
                
                # [수정됨] 5. Val Acc도 로그에 추가
                log_payload["test_acc"] = acc
            
            if ep == args.epochs:
                cfg = dict(args) if hasattr(args, "keys") else vars(args)
                args_path = os.path.join(ckpt_dir, f"args_{args.exp_name}.json")
                with open(args_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                print(f"[args saved] {args_path}")

            # ---------------------------------------------------------
            # [수정됨] 6. WandB Log 최종 전송 (Epoch 당 1회)
            # ---------------------------------------------------------
            if args.use_wandb:
                wandb.log(log_payload)

        # ---------------------------------------------------------
        # [Loop 종료 후] 학습 완료 처리
        # ---------------------------------------------------------
        if args.use_wandb:
            elapsed = time.time() - run_start_time
            elapsed_hms = format_elapsed(elapsed)
            if not crashed:
                acc_text = f"{last_val_acc:.2f}%" if last_val_acc is not None else "N/A"
                wandb.alert(
                    title=f"Run finished Dataset={args.base_data}",
                    text=f"Elapsed: {elapsed_hms} | Final Acc: {acc_text} | exp_name: {args.exp_name}"
                )
                wandb.finish()

    except Exception as e:
        crashed = True
        if args.use_wandb:
            try:
                wandb.alert(
                    title=f"Run crashed",
                    text=f"exp_name: {args.exp_name} | {type(e).__name__}: {e}"
                )
                print("wandb alert sent successfully")
            except Exception as alert_err:
                print(f"wandb.alert() failed: {alert_err}")
        raise

if __name__ == "__main__":
    main()