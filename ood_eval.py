import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd

from utils.dataloader import get_dataloader
from utils.tools import *
from utils.ood_metric import *
from utils.vim import ViM
from utils.method_train import evaluate
from model.cifar_resnet import resnet18
from model.cifar_densenet import DenseNet3
from model.WideResNet import WideResNet
from model.ResNet import ResNet50

set_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


base_data = "cifar10"
model_name = "WideResNet"
# for score_type in ["MSP", "odin", "energy", "M", "react", "vim"]:
for score_type in ["energy"]:
# score_type = "M" # "MSP", "energy", "M", "react"
    model_type = "online" # "vanila", "sam", "online", "oe", "vanila2"
    print("="*70)
    print(f"ID DATA : {base_data}, model : {model_name}, score_type : {score_type}, model_type : {model_type}")
    print("="*70)

    # 모델 설정 (base_data에 따라 달라짐)
    if base_data.lower().startswith("cifar"):
        # CIFAR의 클래스 수 설정
        num_classes_map = {"cifar10": 10, "cifar100": 100}
        num_classes = num_classes_map.get(base_data.lower(), 10)
        
        if model_name == "DenseNet":
            model = DenseNet3(100, num_classes)
        elif model_name == "ResNet":
            model = resnet18(num_classes=num_classes)
        elif model_name == "WideResNet":
            model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.3)
        else:
            raise ValueError("CIFAR 데이터셋에 대해 지원하지 않는 모델 이름입니다. (DenseNet, ResNet, WideResNet 지원)")
        
        # CIFAR용 checkpoint 매핑 (필요 시 모델 가중치 로드)
        checkpoint_map = {
            "cifar10_WideResNet_vanila": "/home/thoon1999/act/ce_binary_E100/stage1_checkpoints/vanila_ce_binary_Stage1_cifar10_WideResNet_ep100.pth",
            "cifar10_WideResNet_vanila2": "/home/thoon1999/act/baseline/checkpoint/cifar10_wrn_pretrained_epoch_99.pt",
            "cifar10_WideResNet_sam": "/home/thoon1999/act/ce_binary_E100/stage1_checkpoints/ce_binary_Stage1_cifar10_WideResNet_ep100.pth",
            "cifar10_WideResNet_online": "/home/thoon1999/act/1121_unified/cifar10/models_ce_binary_s602_wrn_rho0.2_E0.1/model_ep100.pth",
            "cifar10_WideResNet_oe": "/home/thoon1999/act/baseline/checkpoint/cifar10_wrn_s1_oe_tune_epoch_9.pt",
            "cifar100_WideResNet_vanila": "/home/thoon1999/act/ce_binary_E100/stage1_checkpoints/vanila_ce_binary_Stage1_cifar100_WideResNet_ep100.pth",
            "cifar100_WideResNet_vanila2": "/home/thoon1999/act/baseline/checkpoint/cifar100_wrn_pretrained_epoch_99.pt",
            "cifar100_WideResNet_sam": "/home/thoon1999/act/ce_binary_E100/stage1_checkpoints/ce_binary_Stage1_cifar100_WideResNet_ep100.pth",
            "cifar100_WideResNet_online": "/home/thoon1999/act/1121_unified/cifar100/models_ce_binary_s601_wrn_rho0.5_E0.1/model_ep100.pth",
            "cifar100_WideResNet_oe": "/home/thoon1999/act/baseline/checkpoint/cifar100_wrn_s1_oe_tune_epoch_9.pt",
        }

        model_key = f"{base_data.lower()}_{model_name}_{model_type}"
        model_path = checkpoint_map.get(model_key, "")
        
        if model_path:
            if model_type in ["vanila", "sam"]:
                checkpoint = torch.load(model_path, weights_only=False, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded {model_name} for {base_data} from checkpoint: {model_path}")
            else:  # online
                model.load_state_dict(torch.load(model_path, weights_only=False,  map_location=device))
                print(f"Loaded {model_name} for {base_data} from checkpoint: {model_path}")
        else:
            print(f"Checkpoint for {model_key} not found. Using initialized model.")
            
    else:  # base_data가 "imagenet"인 경우
        if model_name == "ResNet50":
            model = ResNet50(num_classes=1000)
            model_path = "/home/thoon1999/act/model/checkpoint/imagenet_res50_7610.pth"
            model.load_state_dict(torch.load(model_path, weights_only=False,  map_location=device))

        elif model_name == "ResNet18":
            model = models.resnet18(pretrained=True)

        elif model_name == "DenseNet":
            model = models.densenet121(pretrained=True)
        else:
            raise ValueError("ImageNet 데이터셋에 대해 지원하지 않는 모델 이름입니다. (ResNet50, ResNet18, DenseNet 지원)")
        
        print(f"Loaded pretrained {model_name} for {base_data}.")

    # 모델을 device에 올리고 평가 모드로 전환
    model.to(device)
    model.eval()
    train_loader = get_dataloader(device=device,base_data=base_data, dataname=base_data, batch_size=128, phase='train')
    id_dataloader = get_dataloader(device=device, base_data=base_data, dataname=base_data, batch_size=200, phase='test')
    # 통계가 없으면 학습 세트로부터 계산

    parser = argparse.ArgumentParser(description="Evaluate OOD detection")
    parser.add_argument('--percentile', type=float, default=99, help='Percentile value (default: 0.9)')
    parser.add_argument('--feature_list', type=float, default=[128.], help='sample_estimator에 넘길 레이어별 채널 개수 리스트')
    args = parser.parse_args([])
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
    args.num_classes = num_classes
    args.train_loader = train_loader
    # ID 데이터셋의 score는 한 번만 계산
    
    id_score = get_score(args, device, id_dataloader, model, temperature=1.0, mode='ID', score_type=score_type)

    criterion = nn.CrossEntropyLoss() 
    val_loss, val_acc = evaluate(model, id_dataloader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    def compute_ood_result(args, id_score, base_data, ood_data, model, score_type, device):
        """
        precomputed id_score와 함께 OOD 데이터셋의 score를 계산하고 metric을 산출하는 함수.
        
        id_score: 이미 계산된 ID 데이터셋의 score
        base_data: ID 데이터셋 이름 (예: 'cifar10', 'cifar100', 'imagenet')
        ood_data: OOD 데이터셋 이름 (예: 'svhn', 'openimage_o' 등)
        model: 평가에 사용할 모델
        score_type: 사용할 score (예: 'MSP', 'energy', 'our' 등)
        device: 연산에 사용할 디바이스 (cpu 또는 cuda)
        
        return: 계산된 metric 결과 (딕셔너리)
        """
        # OOD 데이터에 대해서만 score 계산
        ood_dataloader = get_dataloader(device=device, base_data=base_data, dataname=ood_data, batch_size=200, phase='ood')
        ood_score = get_score(args, device, ood_dataloader, model, temperature=1.0, mode='OOD', score_type=score_type)

        result = compute_metrics(id_score, ood_score)

        return result


    ood_datas = ['svhn', 'LSUN-R', 'texture', 'iSUN', 'LSUN-C', 'places365']
    # ood_datas = ['texture', 'svhn', 'places365', 'LSUN-C', 'LSUN-R', 'iSUN']
        # ood_datas = ['texture']
    results_list = []
    for ood_data in ood_datas:
        result = compute_ood_result(args, id_score, base_data, ood_data, model, score_type, device)
        result['ood_data'] = ood_data
        results_list.append(result)

    dfs = pd.DataFrame(results_list).set_index('ood_data')
    dfs.loc['Average'] = dfs.mean()  # 전체 평균 계산
    print(dfs)

    print("Thresholds:")
    print(dfs.threshold)

    # threshold 컬럼은 출력만 하고, 계산 결과는 나머지 metric에 대해서만 반올림
    dfs_metrics = round(dfs.loc[:, ~dfs.columns.isin(['threshold', 'aupr'])] * 100, 2)
    print("Results (in %):")
    print(dfs_metrics)

    df_result = make_result_table(args, base_data, model_name, model_type, score_type, dfs_metrics, include_average=True)
    save_path = f'./master_result/{base_data}_result.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path, header=[0,1])
        combined_df = pd.concat([existing_df, df_result], ignore_index=True)
        combined_df.to_csv(save_path, index=False)
    else:
        df_result.to_csv(save_path, index=False)
