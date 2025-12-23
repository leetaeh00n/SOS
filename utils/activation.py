import numpy as np
import torch
import torch.nn.functional as F

def compute_softmax_score(logits, temperature=1.0):
    softmax_scores = F.softmax(logits / temperature, dim=1)
    return torch.max(softmax_scores, dim=1).values.cpu().numpy()

def compute_energy_score(logits):
    '''
    negative energy score로써 큰 energy score: ID
    '''
    if isinstance(logits, np.ndarray):  # numpy 배열이면 변환
        logits = torch.from_numpy(logits).float()
    
    return torch.logsumexp(logits, dim=1).detach().cpu().numpy()

def scale(x, percentile=95):
    input = x.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale1 = s1 / s2
    
    return input * torch.exp(scale1[:, None, None, None])

def get_react_score(args, model, dataloader, mode, device):
    scores = []
    if mode.upper() == 'ID':
        # [1] 먼저 ID 데이터에 대해 penultimate layer의 activation 수집
        all_activations = []
        for images, *_ in dataloader:
            model.to(device)
            images = images.to(device)
            with torch.no_grad():
                # 모델에 구현된 penultimate activation 추출 함수 사용 
                _, penultimate = model.forward_virtual(images)  
            all_activations.append(penultimate.cpu().numpy())
        all_activations = np.concatenate(all_activations, axis=0)
        computed_threshold = np.percentile(all_activations, args.percentile)
        args.threshold = computed_threshold
        
        # [2] 계산된 threshold를 바탕으로 react 스코어 계산 (ID 데이터)
        for images, *_ in dataloader:
            model.to(device)
            images = images.to(device)
            with torch.no_grad():
                logits = model.forward_react(images, args.threshold)
                scores_batch = compute_energy_score(logits)
            scores.extend(scores_batch)
        
        return np.array(scores)
    
    else:  # mode가 'OOD'인 경우
        # OOD 모드에서 threshold가 args에 없다면 기본값 사용
        if not hasattr(args, 'threshold') or args.threshold is None:
            default_threshold = 1.0  # 필요에 따라 변경 가능
            args.threshold = default_threshold

        for images, *_ in dataloader:
            model.to(device)
            images = images.to(device)
            with torch.no_grad():
                logits = model.forward_react(images, args.threshold)
                scores_batch = compute_energy_score(logits)
            scores.extend(scores_batch)
        return np.array(scores)