"""Sharpness-Aware Minimization based Outlier Synthesis (SOS)"""
"""Rho scheduling 추가"""
"""validation 추가"""
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import TensorDataset, DataLoader, Subset

from model.cifar_resnet import *
from model.WideResNet import WideResNet
from model.sephead import SeparationHead, LossPredLoss
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
from utils.method_train import evaluate
from utils.sam import SAM, sam_restore

run_start_time = time.time()

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def compute_energy(logits, T=1.0):
    """Energy calculation for Code 1 style"""
    return T * torch.logsumexp(logits / T, dim=1)

def extract_all_sam_features_for_epoch(model, dataloader, device, args, criterion, rho_ood):
    """
    Unified Feature Extraction
    - args.ood_gen_mode == 'energy': Perturb to increase Energy (Code 1)
    - args.ood_gen_mode == 'ce': Perturb to increase CE Loss (Code 2)
    """
    model.eval()
    model.zero_grad()
    temp_sam = SAM(
        model.parameters(),
        base_optimizer=torch.optim.SGD,
        rho=rho_ood,
        lr=0.1,
        momentum=0.9,
    )

    all_feat_clean = []
    all_feat_pert = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 1) Gradient Calculation Step
        temp_sam.zero_grad()
        with torch.enable_grad():
            logits_clean, feat_clean = model.forward_virtual(images)
            
            # --- Branching based on Generation Mode ---
            if args.ood_gen_mode == 'energy':
                # Code 1: Maximize Energy
                energy_clean = compute_energy(logits_clean, T=args.temperature)
                loss = energy_clean.mean()
            else: # 'ce'
                # Code 2: Maximize CE Loss (Hard samples)
                loss = criterion(logits_clean, labels)
            # ------------------------------------------
            
            loss.backward()

        # 2) SAM Perturbation Step
        temp_sam.first_step(zero_grad=True)

        # 3) Get Perturbed Features
        with torch.no_grad():
            _, feat_pert = model.forward_virtual(images)

        sam_restore(temp_sam)

        all_feat_clean.append(feat_clean.cpu())
        all_feat_pert.append(feat_pert.cpu())
        all_labels.append(labels.cpu())

        # Clear gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    model.train()
    return (
        torch.cat(all_feat_clean),
        torch.cat(all_feat_pert),
        torch.cat(all_labels),
    )

# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Unified Virtual-OOD Training")
    
    # --- NEW: Core Mode Selectors ---
    parser.add_argument("--ood_gen_mode", type=str, default="energy", choices=["energy", "ce"],
                        help="Method to generate vOOD: 'energy' (ascent) or 'ce' (loss ascent)")
    parser.add_argument("--ood_train_mode", type=str, default="binary", choices=["binary", "regularization"],
                        help="Training objective: 'binary' (Energy Head) or 'regularization' (KL/Sep/Rank)")

    # WandB
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument('--use_val', action='store_true', default=False,
                        help='Use validation split from training data')
    # Mode 1 Specific (Energy Head / Binary)
    parser.add_argument("--lambda_energy", type=float, default=0.1,
        help="Weight for energy discriminator (Only used if ood_train_mode='binary')")

    # Mode 2 Specific (Regularization / SepHead)
    parser.add_argument('--use_sep_loss', action='store_true', default=False)
    parser.add_argument('--use_kl_loss', action='store_true', default=False)
    parser.add_argument('--use_rank_loss', action='store_true', default=False)
    parser.add_argument('--lambda_sep', type=float, default=0.1)
    parser.add_argument('--lambda_kl', type=float, default=0.1)
    parser.add_argument('--lambda_rank', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=1.0)
    
    # Selection Mechanism (from Code 2)
    parser.add_argument('--use_select', action='store_true',
                        help='Select top-k entropy samples for vOOD (usually with ce generation)')

    # Common Hyperparams
    parser.add_argument("--temperature", type=float, default=1.0, help="Temp for energy/softmax")
    parser.add_argument("--base_data", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=40, help="Stage2 start epoch")
    parser.add_argument("--rho", type=float, default=0.1, help="Train rho")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--model_name", type=str, default="WideResNet", choices=["WideResNet", "ResNet"])
    
    # Model Arch
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--drop_rate", type=float, default=0.3)

    # System
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_dir_base", type=str, default="./unified_exp")
    
    # vOOD Generation Param
    parser.add_argument("--rho_ood_mode", type=str, default="linear_dec", 
                        choices=["energy_metric", "auroc_ma", "linear_inc", "linear_dec", "cosine", "const"])
    parser.add_argument("--rho_ood_min", type=float, default=0.1)
    parser.add_argument("--rho_ood_max", type=float, default=0.5)

    return parser.parse_args()

args = parse_args()

# ---------------------------------------------------------
# Experiment Name & Setup
# ---------------------------------------------------------
if args.model_name == "WideResNet":
    model_aka = "wrn"
elif args.model_name == "ResNet":
    model_aka = f"resnet{18 if args.base_data=='cifar10' else 34}"

if args.exp_name is None:
    # 자동 이름 생성: GenMode_TrainMode_Details 형태
    args.exp_name = (
        f"{args.ood_gen_mode}_{args.ood_train_mode}_"
        f"s{args.seed}_{model_aka}_mode_{args.rho_ood_mode}_"
        f"rho{args.rho_ood_min}-{args.rho_ood_max}"
    )
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
stage1_base_dir = f"/home/thoon1999/act/{args.ood_gen_mode}_{args.ood_train_mode}_E{args.epochs}/stage1_checkpoints"
stage1_model_path = f"{args.ood_gen_mode}_{args.ood_train_mode}_Stage1_{args.base_data}_{args.model_name}_ep{args.start_epoch}.pth"
stage1_ckpt_path = os.path.join(stage1_base_dir, stage1_model_path)
os.makedirs(stage1_base_dir, exist_ok=True)

data_specific_dir = f"{args.save_dir_base}/{args.base_data}"
save_dir = f"{data_specific_dir}/feat_{args.exp_name}"
ckpt_dir = f"{data_specific_dir}/models_{args.exp_name}"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Data
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
# 1. Main Backbone
if args.model_name == "ResNet":
    model = resnet18(num_classes=num_classes).to(device) if args.base_data == 'cifar10' else resnet34(num_classes=num_classes).to(device)

elif args.model_name == "WideResNet":
    model = WideResNet(
        depth=args.depth, 
        widen_factor=args.widen_factor, 
        dropRate=args.drop_rate, 
        num_classes=num_classes
    ).to(device)

# 2. Auxiliary Modules (Conditional)
energy_head = None
sep_head = None

# If using Binary Mode (Code 1)
if args.ood_train_mode == 'binary':
    energy_head = nn.Linear(1, 2).to(device)

# If using Rank Loss (Code 2)
if args.ood_train_mode == 'regularization' and args.use_rank_loss:
    with torch.no_grad():
        _, df = model.forward_virtual(torch.zeros(1, 3, 32, 32).to(device))
    feat_dim = df.shape[1]
    sep_head = SeparationHead(feat_dim).to(device)

# 3. Optimizer Setup
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
uniform_dist = torch.ones(num_classes, device=device) / num_classes if args.use_kl_loss else None

# Load Stage 1
resume_epoch = load_stage1_checkpoint(device, model, sam_optimizer, scheduler, stage1_ckpt_path)
actual_start_epoch = resume_epoch

if args.epochs > 100:
    save_epochs = {81, 100, 140, 170, 200}
else:
    save_epochs = {41, 61, 81, 100}

# ---------------------------------------------------------
# Training Function
# ---------------------------------------------------------
def train_epoch(args, model, loader, epoch, vood_loader=None):
    model.train()
    if energy_head: energy_head.train()
    if sep_head: sep_head.train()

    collect = epoch in save_epochs
    if collect: feat_orig_list, feat_pert_list = [], []
    
    avg_loss = 0.0
    n_seen = 0
    stage2 = (epoch > args.start_epoch)

    # ★ Stage2 & binary 모드일 때 AUROC용 energy 저장 리스트
    id_energy_list = []
    vood_energy_list = []

    if stage2:
        if vood_loader is None:
            raise ValueError("Need vood_loader for Stage 2")
        # ✅ step 수가 같아야 함 (중복/누락 방지)
        assert len(vood_loader) == len(loader), f"ID steps={len(loader)} != vOOD steps={len(vood_loader)}"
        vood_iter = iter(vood_loader)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
# --- Stage 1: Standard SGD Training ---
        if not stage2:

            sam_optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            sam_optimizer.first_step(zero_grad=True)
            
            logits_p = model(images)
            loss_p = criterion(logits_p, labels)
            loss_p.backward()
            sam_optimizer.second_step(zero_grad=True)
            current = loss_p.item()

        # --- Stage 2: vOOD Training ---
        else:
            # ✅ StopIteration 재시작 금지: 1 epoch 동안 vOOD 40,000 전부 정확히 한 번 사용
            feat_pert = next(vood_iter)[0].to(device)


            # Common: Extract ID features
            with torch.no_grad():
                _, feat_orig = model.forward_virtual(images)
            
            if collect:
                feat_orig_list.append(feat_orig.detach().cpu())
                feat_pert_list.append(feat_pert.detach().cpu())

            # =================================================
            # Step 1 of SAM
            # =================================================
            sam_optimizer.rho = args.rho
            sam_optimizer.zero_grad()
            
            logits_id1, feat_id1 = model.forward_virtual(images)
            L_cls1 = criterion(logits_id1, labels)
            total_loss1 = L_cls1
            loss_dict = {"L_cls": L_cls1.item()}
            
            # Process vOOD for Step 1
            feat_p1 = feat_pert.detach().requires_grad_(True)
            v_logit_p1 = model.fc(feat_p1)
            
            # [Branch: Binary Mode]
            if args.ood_train_mode == 'binary':
                energy_id1 = compute_energy(logits_id1, T=args.temperature).unsqueeze(1)
                energy_vood1 = compute_energy(v_logit_p1, T=args.temperature).unsqueeze(1)
                energy_all1 = torch.cat([energy_id1, energy_vood1], dim=0)
                
                # Label: ID=1, vOOD=0
                bin_labels = torch.cat([
                    torch.ones(energy_id1.size(0), dtype=torch.long, device=device),
                    torch.zeros(energy_vood1.size(0), dtype=torch.long, device=device)
                ], dim=0)
                
                bin_logits1 = energy_head(energy_all1)
                L_energy1 = criterion(bin_logits1, bin_labels)
                total_loss1 += args.lambda_energy * L_energy1
                loss_dict['L_energy'] = L_energy1.item()
                
            # [Branch: Regularization Mode]
            elif args.ood_train_mode == 'regularization':
                # Pseudo labels (needed for Sep/Rank)
                with torch.no_grad():
                    v_pseudo_labels = v_logit_p1.argmax(dim=1) # using current logit for pseudo

                if args.use_sep_loss:
                    L_sep1 = F.cross_entropy(v_logit_p1, v_pseudo_labels)
                    total_loss1 -= args.lambda_sep * L_sep1
                    loss_dict['L_sep'] = L_sep1.item()
                
                if args.use_kl_loss:
                    L_kl1 = F.kl_div(F.log_softmax(v_logit_p1, dim=-1),
                                     uniform_dist.expand(v_logit_p1.size(0), -1),
                                     reduction='batchmean')
                    total_loss1 += args.lambda_kl * L_kl1
                    loss_dict['L_kl'] = L_kl1.item()
                    
                if args.use_rank_loss and sep_head is not None:
                    id_pseudo = logits_id1.argmax(dim=1)
                    id_conf = F.cross_entropy(logits_id1, id_pseudo, reduction='none')
                    v_conf = F.cross_entropy(v_logit_p1, v_pseudo_labels, reduction='none')
                    
                    combined_target = torch.cat([id_conf, v_conf], dim=0)
                    combined_feat = torch.cat([feat_id1.detach(), feat_p1], dim=0)
                    combined_pred = sep_head(combined_feat).squeeze()
                    
                    L_rank1 = LossPredLoss(combined_pred, combined_target, margin=args.margin)
                    total_loss1 += args.lambda_rank * L_rank1
                    loss_dict['L_rank'] = L_rank1.item()

            total_loss1.backward()
            sam_optimizer.first_step(zero_grad=True)

            # =================================================
            # Step 2 of SAM
            # =================================================
            logits_id2, feat_id2 = model.forward_virtual(images)
            L_cls2 = criterion(logits_id2, labels)
            total_loss2 = L_cls2
            
            feat_p2 = feat_pert.detach().requires_grad_(True)
            v_logit_p2 = model.fc(feat_p2)

            # [Branch: Binary Mode]
            if args.ood_train_mode == 'binary':
                energy_id2 = compute_energy(logits_id2, T=args.temperature).unsqueeze(1)
                energy_vood2 = compute_energy(v_logit_p2, T=args.temperature).unsqueeze(1)
                energy_all2 = torch.cat([energy_id2, energy_vood2], dim=0)
                
                # Re-create labels (same shape)
                bin_labels = torch.cat([
                    torch.ones(energy_id2.size(0), dtype=torch.long, device=device),
                    torch.zeros(energy_vood2.size(0), dtype=torch.long, device=device)
                ], dim=0)
                
                bin_logits2 = energy_head(energy_all2)
                L_energy2 = criterion(bin_logits2, bin_labels)
                total_loss2 += args.lambda_energy * L_energy2

                # ★ AUROC 계산을 위해 에너지 저장 (detach + CPU)
                id_energy_list.append(energy_id2.detach().cpu())       # [B, 1]
                vood_energy_list.append(energy_vood2.detach().cpu())   # [B, 1]

            # [Branch: Regularization Mode]
            elif args.ood_train_mode == 'regularization':
                # Re-calc pseudo labels for step 2 logic (or maintain? usually recalc)
                with torch.no_grad(): v_pseudo_labels = v_logit_p2.argmax(dim=1)

                if args.use_sep_loss:
                    L_sep2 = F.cross_entropy(v_logit_p2, v_pseudo_labels)
                    total_loss2 -= args.lambda_sep * L_sep2
                
                if args.use_kl_loss:
                    L_kl2 = F.kl_div(F.log_softmax(v_logit_p2, dim=-1),
                                     uniform_dist.expand(v_logit_p2.size(0), -1),
                                     reduction='batchmean')
                    total_loss2 += args.lambda_kl * L_kl2
                
                if args.use_rank_loss and sep_head is not None:
                    id_pseudo = logits_id2.argmax(dim=1)
                    id_conf = F.cross_entropy(logits_id2, id_pseudo, reduction='none')
                    v_conf = F.cross_entropy(v_logit_p2, v_pseudo_labels, reduction='none')
                    
                    combined_target = torch.cat([id_conf.detach(), v_conf.detach()], dim=0)
                    combined_feat = torch.cat([feat_id2.detach(), feat_p2], dim=0)
                    combined_pred = sep_head(combined_feat).squeeze()
                    
                    L_rank2 = LossPredLoss(combined_pred, combined_target, margin=args.margin)
                    total_loss2 += args.lambda_rank * L_rank2

            total_loss2.backward()
            sam_optimizer.second_step(zero_grad=True)
            
            current = total_loss2.item()
            n_seen += images.size(0)

        scheduler.step()
        avg_loss = 0.8 * avg_loss + 0.2 * current
        

    if collect and n_seen > 0:
        np.save(os.path.join(save_dir, f"features_ep{epoch:03d}_orig.npy"), torch.cat(feat_orig_list).numpy())
        np.save(os.path.join(save_dir, f"features_ep{epoch:03d}_pert.npy"), torch.cat(feat_pert_list).numpy())
    
    # ★ Stage2 & binary 모드면 에너지도 돌려주기
    if stage2 and args.ood_train_mode == 'binary' and len(id_energy_list) > 0:
        id_energy_all = torch.cat(id_energy_list, dim=0).view(-1)       # [N]
        vood_energy_all = torch.cat(vood_energy_list, dim=0).view(-1)   # [N]
        return avg_loss, id_energy_all, vood_energy_all
    else:
        return avg_loss, None, None

# ---------------------------------------------------------
# Main Loop
# ---------------------------------------------------------
crashed = False
last_val_acc = 0
rho_state = {}
val_metric = 0.0
try:
    for ep in range(actual_start_epoch, args.epochs + 1):
        
        if ep > args.start_epoch:
            print(f"\n[Epoch {ep}] Extracting vOOD ({args.ood_gen_mode})...")
            # ★ 1. 이 epoch에서 사용할 rho_ood 계산
            rho_ood_current, rho_state = get_rho_ood(
            mode=args.rho_ood_mode, 
            epoch=ep,
            total_epochs=args.epochs,
            start_epoch=args.start_epoch,
            rho_min=args.rho_ood_min,
            rho_max=args.rho_ood_max,
            val_metric=val_metric,      # 여기다 네가 계산한 AUROC 넣기
            state=rho_state,
            ema_beta=0.9,               # EMA용 베타값
            auroc_min=0.5,            # 데이터셋 특성에 맞게 조정 가능
            auroc_max=1.0,
            )
            print(f" -> rho_ood_current = {rho_ood_current:.4f}")

            # 1. Extract Features (train_loader 기반: use_val=True면 45,000개)
            _, feat_pert, _ = extract_all_sam_features_for_epoch(
                model, train_loader, device, args, criterion, rho_ood_current
            )

            # 2. vOOD를 40,000(train) / 5,000(val_metric)로 split (누락/중복 없이)
            N_total = feat_pert.size(0)  # use_val=True면 45000
            if args.use_val:
                assert N_total >= 45000, f"Expected vOOD >= 45000 when use_val=True, got {N_total}"
                n_vood_val = 5000
                n_vood_train = 40000

                # 재현성을 위해 epoch+seed로 고정 셔플 (원하면 ep 빼고 seed만 써서 고정 split도 가능)
                g = torch.Generator()
                g.manual_seed(args.seed + ep)

                perm = torch.randperm(N_total, generator=g)
                idx_train = perm[:n_vood_train]
                idx_val   = perm[n_vood_train:n_vood_train + n_vood_val]

                feat_pert_train = feat_pert[idx_train]  # [40000, D]
                feat_pert_val   = feat_pert[idx_val]    # [5000, D]
            else:
                # use_val=False면 기존처럼 전체를 학습용으로 사용(=기존 동작 유지)
                feat_pert_train = feat_pert
                feat_pert_val = None

            # 3. vOOD_train loader는 ID train_loader step 수에 맞게 batch_size 자동 설정
            id_steps = len(train_loader)  # ID는 45,000 / bs=128 기준 step 수
            vood_train_bs = int(np.ceil(feat_pert_train.size(0) / id_steps))
            # 핵심: drop_last=False로 40,000 전부 사용 + len(vood_loader)=len(train_loader)가 되도록 설계
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

            # (선택) 안전 체크: vOOD step 수가 ID step 수와 정확히 맞는지 확인
            assert len(vood_loader) == id_steps, f"Step mismatch: ID steps={id_steps}, vOOD steps={len(vood_loader)}"

            # 4. val_metric 계산용 vOOD_val loader (5k): batch_size는 args.batch_size(128) 그대로 사용
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

            
            # 4. Train (Uses ood_train_mode internally)
            loss, id_energy_all, vood_energy_all = train_epoch(args, model, train_loader, ep, vood_loader=vood_loader)
            
        else:
            rho_ood_current = 0.0 # Stage 1에서는 0
            loss, _, _ = train_epoch(args, model, train_loader, ep)
        
        print(f"[Epoch {ep}] Loss: {loss:.4f} | Mode: {args.ood_gen_mode}/{args.ood_train_mode}")
        if args.use_wandb:
            # [추가] rho_ood 값을 함께 로깅하여 스케줄링 확인
            log_dict = ({
                "epoch": ep, 
                "train_loss": loss, 
                "rho_ood": rho_ood_current
            })
            # metric 종류에 따라 로깅
            if args.rho_ood_mode == "energy_metric":
                log_dict["val_energy"] = val_metric
            else:
                log_dict["val_auroc"] = val_metric
            wandb.log(log_dict)

        # Checkpointing
        if ep == args.start_epoch:
            save_stage1_checkpoint(model, sam_optimizer, scheduler, ep, stage1_ckpt_path)
        
        if ep % 10 == 0 or ep == args.epochs or ep in save_epochs:
            v_loss, acc = evaluate(model, test_loader, criterion, device)
            print(f" -> Val Acc: {acc:.2f}%")
            last_val_acc = float(acc)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_ep{ep}.pth"))
            if energy_head: 
                torch.save(energy_head.state_dict(), os.path.join(ckpt_dir, f"head_ep{ep}.pth"))
            if sep_head:
                torch.save(sep_head.state_dict(), os.path.join(ckpt_dir, f"sep_ep{ep}.pth"))
        
        if ep == args.epochs:
            import json
            cfg = dict(args) if hasattr(args, "keys") else vars(args)
            args_path = os.path.join(ckpt_dir, f"args_{args.exp_name}.json")
            with open(args_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"[args saved] {args_path}")
                
        # ====== Stage2 & binary 모드에서 Metric 업데이트 ======
        if ep > args.start_epoch and args.ood_train_mode == 'binary':
            #  1) use_val=True면 val_loader 기준으로 metric 계산
            if args.use_val and (val_loader is not None):
                current_snr, current_auroc, mu_id, mu_vood = compute_binary_metrics_on_loader(
                    model=model,
                    id_loader=val_loader,
                    vood_loader=vood_val_loader,
                    args=args,
                    device=device,
                    max_batches=None  # 필요하면 속도 위해 int로 제한 가능
                )
            #  2) use_val=False면 기존 train_epoch에서 나온 에너지로 metric 계산(기존 동작 유지)
            else:
                if id_energy_all is None:
                    # use_val=False인데도 안전하게 넘어가도록(예: stage2 직후 예외 상황)
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
            print(f"    (Debug) Mean ID: {mu_id:.2f}, Mean vOOD: {mu_vood:.2f}")

            #  스케줄링 metric 업데이트
            if args.rho_ood_mode == "energy_metric":
                val_metric = current_snr
            else:
                val_metric = current_auroc

            if args.use_wandb:
                wandb.log({
                    "monitor_snr": current_snr,
                    "monitor_auroc": current_auroc,
                    "mean_energy_id": mu_id,
                    "mean_energy_vood": mu_vood,
                })

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