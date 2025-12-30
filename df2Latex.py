import os
import argparse
import torch
import pandas as pd
import numpy as np

# 기존 코드의 의존성 (환경에 맞게 경로 확인 필요)
from utils.dataloader import get_dataloader
from utils.tools import *
from utils.ood_metric import *
from model.cifar_resnet import resnet18, resnet34
from model.cifar_densenet import DenseNet3
from model.WideResNet import WideResNet

# Pandas 출력 옵션
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

# ==============================================================
# 1. User Configuration (이 부분만 수정하세요)
# ==============================================================
base_data   = "cifar10"         # "cifar10" or "cifar100"
model_name  = "ResNet"      # "WideResNet", "ResNet", "DenseNet"
seed        = 0               # 0, 1, 2 ...
total_epoch = 100               # 100, 200 ...
metric_mode = "energy_metric"   # "energy_metric" or "auroc_ma"

# (기타 설정)
eval_score_type = "energy"
batch_size      = 200
device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ==============================================================

def get_generated_path(base_data, model_name, seed, total_epoch, metric_mode):
    """설정에 따른 Path 생성"""
    if model_name == "WideResNet":
        model_aka = "wrn"
    elif model_name == "DenseNet":
        model_aka = "densenet"
    elif model_name == "ResNet":
        model_aka = "resnet18" if base_data == "cifar10" else "resnet34"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    rho_range = "0.0-0.5" if base_data == "cifar10" else "0.0-1.0"
    
    root_dir = f"./sos_rho_schedule/ce_binary_E{total_epoch}/seed{seed}/{metric_mode}/{base_data}"
    folder_name = f"models_ce_binary_s{seed}_{model_aka}_mode_{metric_mode}_rho{rho_range}_E0.1"
    file_name = f"model_ep{total_epoch}.pth"

    return os.path.join(root_dir, folder_name, file_name)

def generate_latex_table(base_data, values_list):
    """계산된 값을 받아 전체 LaTeX 테이블 코드를 생성하는 함수"""
    
    # 데이터 행(Row) 생성
    row_str = "Ours"
    for val in values_list:
        row_str += f" & {val:.2f}"
    row_str += " \\\\"

    # 데이터셋 이름 대문자 변환 (Caption용)
    dataset_name = "CIFAR-10" if base_data == "cifar10" else "CIFAR-100"

    # 전체 LaTeX 템플릿
    latex_code = fr"""
    \begin{{table*}}[t]
    \centering
    \caption{{OOD detection performance on {dataset_name} as ID.}}
    \label{{tab:ood_{base_data}}}
    \resizebox{{\textwidth}}{{!}}{{
    \begin{{tabular}}{{lcccccccccccccc}}
    \toprule
    \multirow{{2}}{{*}}{{Method}} & \multicolumn{{2}}{{c}}{{SVHN}} & \multicolumn{{2}}{{c}}{{LSUN-R}} & \multicolumn{{2}}{{c}}{{texture}} & \multicolumn{{2}}{{c}}{{iSUN}} & \multicolumn{{2}}{{c}}{{LSUN-C}} & \multicolumn{{2}}{{c}}{{places365}} & \multicolumn{{2}}{{c}}{{Average}} \\
    \cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}} \cmidrule(lr){{8-9}} \cmidrule(lr){{10-11}} \cmidrule(lr){{12-13}} \cmidrule(lr){{14-15}}
    & FPR95$\downarrow$ & AUROC$\uparrow$ & FPR95$\downarrow$ & AUROC$\uparrow$ & FPR95$\downarrow$ & AUROC$\uparrow$ & FPR95$\downarrow$ & AUROC$\uparrow$ & FPR95$\downarrow$ & AUROC$\uparrow$ & FPR95$\downarrow$ & AUROC$\uparrow$ & FPR95$\downarrow$ & AUROC$\uparrow$ \\
    \midrule
    {row_str}
    \bottomrule
    \end{{tabular}}
    }}
    \end{{table*}}
"""
    return latex_code

def main():
    # 1. 경로 생성 및 확인
    target_ckpt_path = get_generated_path(base_data, model_name, seed, total_epoch, metric_mode)
    
    print(f"\n{'='*60}")
    print(f"Target Configuration:")
    print(f" - Data : {base_data}")
    print(f" - Model: {model_name}")
    print(f" - Seed : {seed}")
    print(f" - Epoch: {total_epoch}")
    print(f" - Mode : {metric_mode}")
    print(f"{'-'*60}")
    print(f"Generated Path: \n{target_ckpt_path}")
    print(f"{'='*60}\n")
    print(f"\n>> Loading: {target_ckpt_path}")
    
    if not os.path.exists(target_ckpt_path):
        print("❌ Error: Path does not exist!")
        return

    num_classes = 100 if base_data == 'cifar100' else 10
    
    # 2. 모델 로드
    def get_model_instance(m_name, num_cls):
        if m_name == "WideResNet":
            return WideResNet(depth=40, num_classes=num_cls, widen_factor=2, dropRate=0.3)
        elif m_name == "ResNet":
            return resnet18(num_classes=num_cls) if base_data == "cifar10" else resnet34(num_classes=num_cls)
        elif m_name == "DenseNet":
            return DenseNet3(100, num_cls)
        else:
            raise ValueError(f"Unknown model: {m_name}")

    try:
        model = get_model_instance(model_name, num_classes)
        checkpoint = torch.load(target_ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. ID 데이터 평가
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.num_classes = num_classes
    
    id_dataloader = get_dataloader(device=device, base_data=base_data, dataname=base_data, batch_size=batch_size, phase='test')
    id_score = get_score(args, device, id_dataloader, model, temperature=1.0, mode='ID', score_type=eval_score_type)

    # 4. OOD 데이터 평가 및 값 수집
    ood_datas = ['svhn', 'LSUN-R', 'texture', 'iSUN', 'LSUN-C', 'places365']
    
    # (Dataset, Metric) MultiIndex용 리스트
    columns_tuples = []
    values_list = []  # DataFrame 및 LaTeX용 값 저장

    for ood_data in ood_datas:
        ood_dataloader = get_dataloader(device=device, base_data=base_data, dataname=ood_data, batch_size=batch_size, phase='ood')
        ood_score = get_score(args, device, ood_dataloader, model, temperature=1.0, mode='OOD', score_type=eval_score_type)
        
        metrics = compute_metrics(id_score, ood_score)
        fpr = metrics['fpr95'] * 100
        auroc = metrics['auroc'] * 100
        
        columns_tuples.extend([(ood_data, 'FPR95'), (ood_data, 'AUROC')])
        values_list.extend([fpr, auroc])

    # 평균 계산
    avg_fpr = np.mean(values_list[0::2])   # 짝수 인덱스
    avg_auroc = np.mean(values_list[1::2]) # 홀수 인덱스
    
    columns_tuples.extend([('Average', 'FPR95'), ('Average', 'AUROC')])
    values_list.extend([avg_fpr, avg_auroc])

    # 5. DataFrame 출력 (터미널 확인용)
    multi_index = pd.MultiIndex.from_tuples(columns_tuples, names=['Dataset', 'Metric'])
    df = pd.DataFrame([values_list], columns=multi_index, index=['Ours'])
    
    print("\n" + "="*80)
    print(" 📊 Computed Results (DataFrame)")
    print("="*80)
    print(df)
    print("="*80)

    # 6. 최종 LaTeX 코드 생성 및 출력
    final_latex_code = generate_latex_table(base_data, values_list)

    print("\n\n" + "="*80)
    print(" 📋 Final LaTeX Table Code (Copy & Paste below)")
    print("="*80)
    print(final_latex_code)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()