import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from utils.activation import get_react_score
from metrics.Mahalanobis import Mahalanobis_score, sample_estimator
from metrics.odin import get_odin_score

from sklearn.metrics import roc_auc_score, average_precision_score

# --------------------------------------------------------------------------
# 1. 정밀한 Metric 계산을 위한 Helper 함수들 (Reference Code에서 가져옴)
# --------------------------------------------------------------------------
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """정밀도를 위한 stable cumsum 구현"""
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1.):
    """Recall(TPR) 95% 지점에서의 FPR을 정밀하게 계산"""
    y_true = (y_true == pos_label)

    # 점수 내림차순 정렬
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    thresholds = y_score[threshold_idxs]
    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    
    # FPR 계산: FP / Total Negatives
    return fps[cutoff] / (np.sum(np.logical_not(y_true)))

# --------------------------------------------------------------------------
# 2. Score 계산 함수 수정 (Temperature 추가)
# --------------------------------------------------------------------------
def np2float(value):
    return float(value)

def compute_softmax_score(logits, temperature=1.0):
    softmax_scores = F.softmax(logits / temperature, dim=1)
    return torch.max(softmax_scores, dim=1).values.cpu().numpy()

def compute_energy_score(logits, temperature=1.0):
    '''
    Temperature Scaling이 적용된 Energy Score 계산
    Formula: T * logsumexp(logits / T)
    '''
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    
    # Reference 코드와 동일하게 T를 곱해주고, 안에서 T로 나눠줍니다.
    return temperature * torch.logsumexp(logits / temperature, dim=1).detach().cpu().numpy()

# --------------------------------------------------------------------------
# 3. get_score 및 compute_metrics 수정
# --------------------------------------------------------------------------
def get_score(args, device, dataloader, model, temperature=1.0, mode='ID', score_type='energy'):
    scores = []
    model.eval()
    
    print(f'Calculating {mode} {score_type.capitalize()} Scores')

    # Mahalanobis 분기
    if score_type == 'M':
        scores = Mahalanobis_score(device,
            model,
            dataloader,
            num_classes=args.num_classes,
            sample_mean=args.sample_mean,
            precision=args.precision,
            layer_index=0,
            magnitude=0.001,
            num_batches=50,
            in_dist=(mode=='ID')
        )
        return scores

    elif score_type == 'react':
        scores = get_react_score(args, model, dataloader, mode, device)

    else:
        for images, *_ in dataloader:
            model.to(device)
            images = images.to(device)

            # [ODIN 분기] Gradient 계산이 필요하므로 no_grad 없이 수행
            if score_type == 'odin':
                # args에 noise가 없으면 기본값(0.0014) 사용
                # ODIN 논문 권장 Temperature는 보통 1000
                odin_noise = getattr(args, 'noise', 0.0014)
                odin_temp = getattr(args, 'T', 1000.0) # args.T가 없으면 1000.0 사용
                
            #     scores_batch = get_odin_score(model, images, temperature=odin_temp, noise=odin_noise)
            # if score_type == 'odin':
                images = Variable(images, requires_grad=True)
                logits = model(images)
                odin_score = get_odin_score(device, images, logits, model, odin_temp, odin_noise)
                scores_batch = np.max(odin_score, axis=1).flatten()
            # [기타 Score] Gradient 불필요 -> no_grad 사용
            else:
                with torch.no_grad():
                    if score_type == 'energy':
                        logits = model(images)
                        scores_batch = compute_energy_score(logits, temperature)

                    elif score_type == 'MSP':
                        logits = model(images)
                        scores_batch = compute_softmax_score(logits, temperature)

                    elif score_type == 'scale':
                        logits = model.forward_scale(images, args.scale_p)
                        scores_batch = compute_energy_score(logits, temperature)
                    elif score_type == 'vim':
                        if args.vim_detector is None:
                            raise ValueError("vim_detector must be provided for 'vim' score_type")
                        with torch.no_grad():
                            scores_batch = args.vim_detector.predict(images) 
                    else:
                        raise ValueError('score_type must be "energy", "MSP", "scale", "react", "our", "odin","vim"')
            
            scores.extend(scores_batch)

    return np.array(scores)


def compute_metrics(id_score, ood_score):
    '''
    정밀한 FPR95 계산을 위해 stable_cumsum 방식 적용
    (lambda 기준) id = 1, ood = 0이 되도록 score 가정
    '''
    # 라벨 및 점수 통합
    y_true = np.concatenate([np.ones_like(id_score), np.zeros_like(ood_score)])
    y_scores = np.concatenate([id_score, ood_score])
    
    # [수정] 정밀한 FPR@95 계산 함수 사용
    fpr_at_95 = fpr_and_fdr_at_recall(y_true, y_scores, recall_level=0.95, pos_label=1)
    
    # Threshold는 참고용으로 percentile 사용 (단순 로깅용)
    threshold = np.percentile(id_score, 5)

    # AUROC, AUPR 계산
    auroc = roc_auc_score(y_true, y_scores)
    aupr_id = average_precision_score(y_true, y_scores)

    return {
        'fpr95': np2float(fpr_at_95),
        'threshold': np2float(threshold),
        'auroc': np2float(auroc),
        'aupr': np2float(aupr_id)
    }


def make_result_table(args, base_data, model_name, model_type, score_type, metric_df, include_average=False):
    """
    위 스크립트에서 계산한 OOD 결과 DataFrame(metric_df)을 받아,
    ID, Model, Method와 각 OOD 데이터셋(FPR95, AUROC, AUPR)을
    pandas DataFrame을 생성/반환하는 함수.

    Parameters
    ----------
    base_data : str
        ID 데이터셋 이름 (예: "cifar10", "cifar100", "imagenet")
    model_name : str
        모델 이름 (예: "DenseNet", "ResNet50", ...)
    score_type : str
        OOD 점수 유형 (예: "MSP", "energy", "react", "our" ...)
    metric_df : pd.DataFrame
        OOD 결과를 담고 있는 DataFrame
        - index : 각 OOD 데이터셋 이름(예: svhn, iSUN, places365 등)
        - columns : fpr95, auroc, aupr (추가로 threshold가 있을 수 있음)
        - 'Average' 행이 있을 수도 있음
    include_average : bool, optional
        True이면 Average 행의 결과도 함께 테이블에 포함

    Returns
    -------
    pd.DataFrame
        멀티 인덱스(컬럼)에 (ID, ""), (Model, ""), (Method, ""), (OOD, FPR95), ...
        형태가 포함된 단일 행짜리 DataFrame
    """
    # 우선 사용할 OOD 데이터셋 리스트 추출
    # 'Average' 행은 디폴트로 제외하되, include_average=True면 포함
    ood_list = []
    for idx in metric_df.index:
        if idx == "Average" and not include_average:
            continue
        ood_list.append(idx)

    # 1) 멀티레벨 컬럼 정의
    #    ID, Model, Method는 상위레벨만 두고 하위레벨은 공백("")
    #    그 뒤에 각 OOD -> (FPR95, AUROC, AUPR) 형태로 배치
    columns_tuples = [
        ("ID", " "),
        ("Model", " "),
        ("Method", " ")
    ]

    for ood_data in ood_list:
        columns_tuples.append((ood_data, "FPR95"))
        columns_tuples.append((ood_data, "AUROC"))
        # columns_tuples.append((ood_data, "AUPR"))

    columns = pd.MultiIndex.from_tuples(columns_tuples)

    if score_type == "react":
        score_type += str(args.percentile)

    elif score_type == "scale":
        score_type += str(args.scale_p)
        
    # 2) 단일 행에 들어갈 값 구성
    data_row = [
        base_data,    # (ID, "")
        model_name + '_' + model_type,   # (Model, "")
        score_type    # (Method, "")
    ]

    # metric_df에는 fpr95, auroc, aupr 컬럼이 있다고 가정
    # ood_list 순서대로 테이블에 채워 넣음
    for ood_data in ood_list:
        # fpr95, auroc, aupr가 없는 경우가 있다면 예외처리 필요
        # 여기서는 있다고 가정
        fpr95_val = metric_df.loc[ood_data, "fpr95"]
        auroc_val = metric_df.loc[ood_data, "auroc"]
        # aupr_val  = metric_df.loc[ood_data, "aupr"]

        # data_row.extend([fpr95_val, auroc_val, aupr_val])
        data_row.extend([fpr95_val, auroc_val])

    # 3) DataFrame 생성
    df_result = pd.DataFrame([data_row], columns=columns)

    return df_result

