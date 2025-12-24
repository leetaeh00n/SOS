import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from utils.dataloader import get_dataloader
from utils.tools import *
from utils.ood_metric import *
from metrics.vim import ViM
from utils.trainer import evaluate
from model.cifar_resnet import resnet18
from model.cifar_densenet import DenseNet3
from model.WideResNet import WideResNet
from model.ResNet import ResNet50

# 출력 포맷 설정 (소수점 2자리)
pd.options.display.float_format = '{:.2f}'.format

# GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ================= 사용자 탐색 설정 =================
base_datas = ["cifar10", "cifar100"]          # ID 데이터셋 리스트
seeds = [0, 1, 2, 3, 4]                       # 시드 리스트
train_metrics = ["auroc_ma", "energy_metric"] # 모델 폴더 구분용 Metric
epochs_list = [100, 200, 500]                 # 탐색할 Epoch 리스트 추가
model_name = "WideResNet"

# 평가에 사용할 Score (Energy 고정)
eval_score_type = "energy" 
# =================================================

# 모델 별칭 매핑
m_aka = {"WideResNet": "wrn", "ResNet": "resnet", "DenseNet": "densenet", "ResNet50": "resnet50"}
model_aka = m_aka.get(model_name, model_name.lower())

# 최종 Best 모델 저장을 위한 딕셔너리
final_best_summary = {}

# ----------------- 전체 루프 시작 -----------------
for base_data in base_datas:
    print(f"\n{'='*40}")
    print(f" Start Processing Base Data: {base_data}")
    print(f"{'='*40}")

    # 1. 데이터 로드 (Base Data가 바뀔 때만 수행하여 속도 최적화)
    num_classes = 100 if base_data == 'cifar100' else 10
    
    # 통계 계산용 train_loader / 평가용 id_dataloader
    train_loader = get_dataloader(device=device, base_data=base_data, dataname=base_data, batch_size=128, phase='train')
    id_dataloader = get_dataloader(device=device, base_data=base_data, dataname=base_data, batch_size=200, phase='test')
    
    # 모델 생성 함수
    def get_model():
        if model_name == "WideResNet":
            return WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.3)
        elif model_name == "ResNet":
            return resnet18(num_classes=num_classes)
        elif model_name == "DenseNet":
            return DenseNet3(100, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    # 해당 base_data의 최고 기록 초기화 (모든 Epoch, Seed, Metric을 통틀어 최고 기록)
    best_auroc_record = {"auroc": -1.0, "fpr95": 1.0, "path": None, "info": None}
    best_fpr_record   = {"auroc": -1.0, "fpr95": 2.0, "path": None, "info": None}
    
    # --- Epoch 루프 추가 ---
    for epoch in epochs_list:
        
        # --- Seed & Metric 루프 ---
        for seed in seeds:
            set_seed(seed) # 재현성 확보
            
            for t_metric in train_metrics:
                # -------------------------------------------------
                # 2. 경로 구성 (Epoch 반영)
                # -------------------------------------------------
                # 폴더명에도 Epoch가 들어간다고 가정 (예: ce_binary_E100 -> ce_binary_E200)
                rho_range = "0.0-0.5" if base_data == "cifar10" else "0.0-1.0"
                
                # root_dir에 Epoch 반영: ce_binary_E{epoch}
                root_dir = f"./sos_rho_schedule/ce_binary_E{epoch}/seed{seed}/{t_metric}/{base_data}"
                
                # model_folder 이름 구성
                model_folder = f"models_ce_binary_s{seed}_{model_aka}_mode_{t_metric}_rho{rho_range}_E0.1"
                
                # 파일명에 Epoch 반영: model_ep{epoch}.pth
                ckpt_path = os.path.join(root_dir, model_folder, f"model_ep{epoch}.pth")
                
                # Info 문자열에 Epoch 정보 추가
                model_info_str = f"[Epoch: {epoch} | Seed: {seed} | Metric: {t_metric}]"

                # 체크포인트 확인
                if not os.path.exists(ckpt_path):
                    # print(f"Skipping: {ckpt_path} (Not Found)")
                    continue

                # -------------------------------------------------
                # 3. 모델 로드 및 평가 준비
                # -------------------------------------------------
                try:
                    model = get_model()
                    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    model.to(device)
                    model.eval()
                except Exception as e:
                    print(f"Error loading {ckpt_path}: {e}")
                    continue

                # Argparse 대용 객체 설정
                parser = argparse.ArgumentParser()
                parser.add_argument('--percentile', type=float, default=99)
                parser.add_argument('--feature_list', type=float, default=[128.])
                args = parser.parse_args([])
                args.num_classes = num_classes
                args.train_loader = train_loader
                args.sample_mean, args.precision = None, None
                args.vim_detector = None 

                # -------------------------------------------------
                # 4. Score 계산 및 Metric 산출
                # -------------------------------------------------
                # ID Score
                id_score = get_score(args, device, id_dataloader, model, temperature=1.0, mode='ID', score_type=eval_score_type)

                # OOD Evaluation Loop
                ood_datas = ['svhn', 'LSUN-R', 'texture', 'iSUN', 'LSUN-C', 'places365']
                results_list = []
                
                for ood_data in ood_datas:
                    ood_dataloader = get_dataloader(device=device, base_data=base_data, dataname=ood_data, batch_size=200, phase='ood')
                    ood_score = get_score(args, device, ood_dataloader, model, temperature=1.0, mode='OOD', score_type=eval_score_type)
                    
                    # Metric 계산 (fpr95, AUROC 등)
                    res = compute_metrics(id_score, ood_score)
                    res['ood_data'] = ood_data
                    results_list.append(res)

                # DataFrame 생성 및 평균 계산
                df = pd.DataFrame(results_list).set_index('ood_data')
                df.loc['Average'] = df.mean() # 전체 평균 행 추가

                # -------------------------------------------------
                # 5. 결과 출력 (백분율 변환)
                # -------------------------------------------------
                # 화면 표시용
                display_cols = ['fpr95', 'auroc']
                df_display = df[display_cols] * 100 # % 단위 변환

                print(f"\n>>> Results for {model_info_str} on {base_data}")
                print("-" * 50)
                print(df_display)
                print("-" * 50)

                # -------------------------------------------------
                # 6. Best Model 추적 (AUROC 최대, FPR 최소 각각 추적)
                # -------------------------------------------------
                avg_auroc = df.loc['Average', 'auroc'] 
                avg_fpr   = df.loc['Average', 'fpr95']   
                
                # 1) AUROC 기준 1등 갱신
                if avg_auroc > best_auroc_record['auroc']:
                    best_auroc_record['auroc'] = avg_auroc
                    best_auroc_record['fpr95'] = avg_fpr
                    best_auroc_record['path']  = ckpt_path
                    best_auroc_record['info']  = model_info_str
                
                # 2) FPR95 기준 1등 갱신 (낮을수록 좋음)
                if avg_fpr < best_fpr_record['fpr95']:
                    best_fpr_record['auroc'] = avg_auroc
                    best_fpr_record['fpr95'] = avg_fpr
                    best_fpr_record['path']  = ckpt_path
                    best_fpr_record['info']  = model_info_str

    # 해당 Base Data의 루프가 끝난 후 두 기록을 모두 저장
    final_best_summary[base_data] = {
        "auroc_best": best_auroc_record,
        "fpr95_best": best_fpr_record
    }

# ----------------- 최종 결과 요약 출력 -----------------
print("\n\n")
print("="*80)
print("🏆 FINAL BEST MODELS SUMMARY 🏆")
print("="*80)

for base_data in base_datas:
    records = final_best_summary[base_data]
    auroc_best = records['auroc_best']
    fpr_best = records['fpr95_best']
    
    print(f"Dataset: {base_data}")
    
    if auroc_best['path'] is None:
        print("  -> No valid models found.")
        print("-" * 80)
        continue

    # Case 1: AUROC 1등과 FPR 1등이 같은 모델인 경우 (완벽한 1등)
    if auroc_best['path'] == fpr_best['path']:
        print(f"  👑 Absolute Best Model (Best in both AUROC & FPR)")
        print(f"    - Config     : {auroc_best['info']}")
        print(f"    - AUROC      : {auroc_best['auroc']*100:.2f}%")
        print(f"    - FPR95      : {auroc_best['fpr95']*100:.2f}%")
        print(f"    - Model Path : {auroc_best['path']}")
    
    # Case 2: 서로 다른 모델이 1등인 경우 (둘 다 출력)
    else:
        print(f"  🥇 Best AUROC Model")
        print(f"    - Config     : {auroc_best['info']}")
        print(f"    - AUROC      : {auroc_best['auroc']*100:.2f}%")
        print(f"    - FPR95      : {auroc_best['fpr95']*100:.2f}%")
        print(f"    - Model Path : {auroc_best['path']}")
        print("")
        print(f"  🥇 Best FPR95 Model")
        print(f"    - Config     : {fpr_best['info']}")
        print(f"    - AUROC      : {fpr_best['auroc']*100:.2f}%")
        print(f"    - FPR95      : {fpr_best['fpr95']*100:.2f}%")
        print(f"    - Model Path : {fpr_best['path']}")

    print("-" * 80)