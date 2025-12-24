import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np # 경로 확인용 등 필요 시 사용

from utils.dataloader import get_dataloader
from utils.tools import *
from utils.ood_metric import *
from metrics.vim import ViM
from utils.trainer import evaluate
from model.cifar_resnet import resnet18
from model.cifar_densenet import DenseNet3
from model.WideResNet import WideResNet
from model.ResNet import ResNet50

# 시드 설정
seed = 0
set_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ================= 사용자 설정 (두 번째 코드의 로직 반영) =================
base_data = "cifar10"      # "cifar10" or "cifar100"
model_name = "WideResNet"  # 현재 경로 로직은 WRN에 맞춰져 있음
epoch = 100
metric = "auroc_ma"        # "energy_metric" etc.

# 경로 구성을 위한 설정
# root_dir 예: ./sos_rho_schedule/ce_binary_E100/seed0/auroc_ma/cifar10
root_dir = f"./sos_rho_schedule/ce_binary_E100/seed{seed}/{metric}/{base_data}"

# 모델 폴더명 구성 (두 번째 코드의 feat_ -> models_ 로직 반영)
# target_feat_folder = f"feat_ce_binary_s{seed}_wrn_mode_{metric}_rho0.0-0.5_E0.1"
# model_folder = target_feat_folder.replace("feat_", "models_")
# 위 로직을 그대로 쓰거나, 아래처럼 직접 구성:
model_folder = f"models_ce_binary_s{seed}_wrn_mode_{metric}_rho0.0-0.5_E0.1"

ckpt_path = os.path.join(root_dir, model_folder, f"model_ep{epoch}.pth")

# CSV 저장 시 구분을 위해 model_type을 파일명이나 설정에서 유추하여 지정
model_type = f"s{seed}_{metric}_schedule" 
# =======================================================================


# for score_type in ["MSP", "odin", "energy", "M", "react", "vim"]:
for score_type in ["energy"]:
    print("="*70)
    print(f"ID DATA : {base_data}, model : {model_name}, score_type : {score_type}, model_type : {model_type}")
    print(f"Loading from: {ckpt_path}")
    print("="*70)

    # 1. 모델 초기화 및 로드
    if base_data.lower().startswith("cifar"):
        num_classes_map = {"cifar10": 10, "cifar100": 100}
        num_classes = num_classes_map.get(base_data.lower(), 10)
        
        if model_name == "WideResNet":
            # 두 번째 코드의 WRN 설정 (depth=40, widen_factor=2, dropRate=0.3)
            model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.3)
        elif model_name == "ResNet":
            model = resnet18(num_classes=num_classes)
        elif model_name == "DenseNet":
            model = DenseNet3(100, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # 체크포인트 로드
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # state_dict 키 처리 (DataParallel 등으로 저장된 경우 'module.' 제거 등 필요할 수 있음)
            # 여기서는 두 코드의 일반적인 저장 방식인 model_state_dict가 없으면 전체를 로드한다고 가정
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            print(f"✅ Loaded {model_name} for {base_data} from checkpoint: {ckpt_path}")
        else:
            raise FileNotFoundError(f"❌ Checkpoint not found at: {ckpt_path}")
            
    else:  # ImageNet 등 다른 데이터셋인 경우 (기존 로직 유지)
        if model_name == "ResNet50":
            model = ResNet50(num_classes=1000)
            model_path = "/home/thoon1999/act/model/checkpoint/imagenet_res50_7610.pth" # 기존 경로 유지
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise ValueError("ImageNet support only for ResNet50 in this snippet.")
        print(f"Loaded pretrained {model_name} for {base_data}.")

    # 2. 모델 설정 완료 후 평가 준비
    model.to(device)
    model.eval()

    # 데이터 로더 설정
    train_loader = get_dataloader(device=device, base_data=base_data, dataname=base_data, batch_size=128, phase='train')
    id_dataloader = get_dataloader(device=device, base_data=base_data, dataname=base_data, batch_size=200, phase='test')

    # Argument Parser 설정 (함수 호출에 필요)
    parser = argparse.ArgumentParser(description="Evaluate OOD detection")
    parser.add_argument('--percentile', type=float, default=99, help='Percentile value')
    parser.add_argument('--feature_list', type=float, default=[128.], help='Layer channel list for sample_estimator')
    args = parser.parse_args([])
    
    args.num_classes = num_classes
    args.train_loader = train_loader

    # Score Type별 사전 계산 (Mahalanobis, ViM 등)
    if score_type == 'M':
        args.sample_mean, args.precision = sample_estimator(device, model, num_classes, [128.], train_loader)
    else:
        args.sample_mean, args.precision = None, None
        
    if score_type == 'vim':
        w = model.fc.weight
        b = model.fc.bias
        vim_detector = ViM(model=model, d=64, w=w, b=b)
        vim_detector.fit(id_dataloader, device=device)
        args.vim_detector = vim_detector
    else:
        args.vim_detector = None

    # 3. ID Score 계산 및 평가
    id_score = get_score(args, device, id_dataloader, model, temperature=1.0, mode='ID', score_type=score_type)

    criterion = nn.CrossEntropyLoss() 
    val_loss, val_acc = evaluate(model, id_dataloader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    def compute_ood_result(args, id_score, base_data, ood_data, model, score_type, device):
        # OOD 데이터에 대해서만 score 계산
        ood_dataloader = get_dataloader(device=device, base_data=base_data, dataname=ood_data, batch_size=200, phase='ood')
        ood_score = get_score(args, device, ood_dataloader, model, temperature=1.0, mode='OOD', score_type=score_type)
        result = compute_metrics(id_score, ood_score)
        return result

    # 평가할 OOD 데이터셋 리스트
    ood_datas = ['svhn', 'LSUN-R', 'texture', 'iSUN', 'LSUN-C', 'places365']
    results_list = []
    
    for ood_data in ood_datas:
        print(f"Processing OOD: {ood_data}...")
        result = compute_ood_result(args, id_score, base_data, ood_data, model, score_type, device)
        result['ood_data'] = ood_data
        results_list.append(result)

    # 결과 집계 및 출력
    dfs = pd.DataFrame(results_list).set_index('ood_data')
    dfs.loc['Average'] = dfs.mean()  # 전체 평균 계산
    print(dfs)

    print("Thresholds:")
    print(dfs.threshold)

    # threshold 등 제외하고 퍼센트로 변환
    dfs_metrics = round(dfs.loc[:, ~dfs.columns.isin(['threshold', 'aupr'])] * 100, 2)
    print("Results (in %):")
    print(dfs_metrics)

    # 4. 결과 저장 (CSV)
    df_result = make_result_table(args, base_data, model_name, model_type, score_type, dfs_metrics, include_average=True)
    
    # 파일명에도 metric이나 seed 정보를 반영하고 싶다면 아래 경로 수정 가능
    save_path = f'./master_result/{base_data}_result.csv'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path, header=[0,1])
        combined_df = pd.concat([existing_df, df_result], ignore_index=True)
        combined_df.to_csv(save_path, index=False)
    else:
        df_result.to_csv(save_path, index=False)
    
    print(f"Saved results to {save_path}")