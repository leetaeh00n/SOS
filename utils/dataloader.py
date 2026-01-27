import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 데이터셋 경로 설정
data_dict = {
    "cifar10": "/home/data/cifar10",          # train, test로 나눠야함
    "cifar100": "/home/data/cifar100",        # train, test로 나눠야함
    "imagenet": "/home/data/imagenet_1k",     # train, val로 나눠야함
    "imagenet-c": "/home/data/imagenet-c",    # ood로 사용시 label = -1 or 0 필요
    "imagenet-r": "/home/data/imagenet-r",    # ood로 사용시 label = -1 or 0 필요
    "inaturalist": "/home/data/inaturalist",
    "iSUN": "/home/data/iSUN/test",
    "LSUN-C": "/home/data/LSUN-C/test",
    "LSUN-R": "/home/data/LSUN-R/test",
    "ninco": "/home/data/ninco",              # ood로 사용시 label = -1 or 0 필요
    "openimage_o": "/home/data/openimage-o",
    "PACS": "/home/data/PACS",
    "places365": "/home/data/places365",      # ood로 사용시 label = -1 or 0 필요
    "ssb_hard": "/home/data/ssb_hard",        # ood로 사용시 label = -1 or 0 필요
    "svhn": "/home/data/svhn",
    "texture": "/home/data/texture",          # ood로 사용시 label = -1 or 0 필요
    "VLCS": "/home/data/VLCS"
}

# normalization and image size
info_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010], 32],
    'cifar100': [[0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761], 32],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224],
}

# 기본 이미지 변환 (CIFAR 및 ImageNet 스타일)

def get_dataloader(device, base_data, dataname, batch_size=128, phase="train"):
    """
    학습에 사용한 data(base_data), 데이터셋 이름(dataname), batchsize, phase(train, test, ood)에 맞는 DataLoader를 반환.
    
    Args:
        base_data (str): 학습에 사용한 데이터셋 이름 ("cifar10","cifar100","imagenet" 중 하나)
        dataname (str): 사용할 데이터셋 이름
        batch_size (int): DataLoader의 batch size
        phase (str): "train","val","test","ood" 중 하나
    Returns:
        DataLoader: 해당 데이터셋에 대한 DataLoader
    """
    if base_data not in ["cifar10", "cifar100", 'imagenet']:
        raise ValueError(f"Dataset {base_data} is not used for training")
    
    if dataname not in data_dict:
        raise ValueError(f"Dataset {dataname} is not found in data_dict.")
    
    img_size = info_dict[base_data][2]
    MEAN, STD = info_dict[base_data][0], info_dict[base_data][1]


    # Train Transform (Data Augmentation 포함)
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),  # 랜덤 크롭 후 리사이즈
        transforms.RandomHorizontalFlip(),       # 수평 플립
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Test/Val/OOD Transform (Data Augmentation 없음)
    test_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),  # 크기 맞추기
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    # Phase에 따라 transform 선택
    transform = train_transform if phase == "train" else test_transform

    dataset_path = data_dict[dataname]

    # Phase에 따라 train/test 디렉토리 설정
    if phase == "ood":
        pass

    elif phase in ["train","val", "test"]:
        dataset_path = os.path.join(dataset_path, phase)

    # ImageFolder로 모든 데이터셋 로드
    dataset = ImageFolder(dataset_path, transform=transform)

    # OOD 데이터셋이면 모든 label을 -1로 설정
    if phase == "ood":
        dataset.targets = [-1] * len(dataset)

    # DataLoader 생성
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(phase == "train"), 
        num_workers=12,                        # 16코어 기준으로 12개 워커 사용
        pin_memory=(device.type == 'cuda'),    # GPU 사용시에만 pin_memory 활성화
        persistent_workers=(phase == "train"), # 워커 재사용으로 성능 향상
        prefetch_factor=4                      # 배치 4개씩 미리 로드
    )
    
    print(f"Loaded {dataname} ({phase}) with {len(dataset)} samples.")
    return dataloader

if __name__ == "__main__":
    import time
    import torch
    import torch.nn as nn

    # ---------------------------
    # 실험 설정
    # ---------------------------
    base_data = 'cifar10'
    dataname = 'svhn'   # 임의 OOD도 OK (라벨은 사용 안 함)
    phase = 'ood'
    batch_size = 128
    warmup_batches = 20
    measure_batches = 200  # 데이터가 부족하면 자동으로 줄어듦

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")

    # ---------------------------
    # (1) 기존 DataLoader (원본 함수)
    # ---------------------------
    dl_base = get_dataloader(device, base_data=base_data, dataname=dataname,
                             batch_size=batch_size, phase=phase)

    # 같은 Dataset으로 (2) 튜닝 DataLoader 구성
    dataset = dl_base.dataset
    dl_fast = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,                 # 공정 비교: 학습이 아니라 추론 경로 시간 비교 목적
        num_workers=8,                # 워커 수 증대
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True,      # 워커 재사용
        prefetch_factor=4,            # 사전 로드 배수
        drop_last=False
    )

    # ---------------------------
    # 간단한 모델: Conv2d(3->32) -> ReLU -> Conv2d(32->32)
    # ---------------------------
    # 입력은 (B, 3, H, W) CIFAR나 ImageNet 모두 호환
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
    ).to(device)
    model.eval()

    # ---------------------------
    # 타이밍 유틸 (함수 없이 인라인로직 사용)
    # ---------------------------
    def _sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # ---------------------------
    # (A) 기존 DataLoader 측정
    # ---------------------------
    print(f"\n=== Benchmark: Original DataLoader (pin_memory=False, num_workers={dl_base.num_workers} in get_dataloader) ===")

    # 워밍업
    seen = 0
    for x, _ in dl_base:
        x = x.to(device, non_blocking=(device.type == 'cuda'))
        with torch.no_grad():
            _ = model(x)
        seen += 1
        if seen >= warmup_batches:
            break
    _sync()

    # 측정
    measured = 0
    total_imgs = 0
    t0 = time.perf_counter()
    for x, _ in dl_base:
        if measured >= measure_batches:
            break
        x = x.to(device, non_blocking=(device.type == 'cuda'))
        with torch.no_grad():
            _ = model(x)
        measured += 1
        total_imgs += x.size(0)
    _sync()
    t1 = time.perf_counter()
    elapsed_base = t1 - t0
    ms_per_batch_base = (elapsed_base / max(1, measured)) * 1000.0
    ips_base = total_imgs / max(elapsed_base, 1e-9)
    print(f"[RESULT] original: {measured} batches, {total_imgs} images | "
          f"{ms_per_batch_base:.2f} ms/batch | {ips_base:.1f} imgs/s")

    # ---------------------------
    # (B) 튜닝 DataLoader 측정 (pin_memory/워커/프리페치)
    # ---------------------------
    print("\n=== Benchmark: Tuned DataLoader (pin_memory, more workers, prefetch) ===")

    # 워밍업
    seen = 0
    for x, _ in dl_fast:
        x = x.to(device, non_blocking=(device.type == 'cuda'))  # non_blocking 중요!
        with torch.no_grad():
            _ = model(x)
        seen += 1
        if seen >= warmup_batches:
            break
    _sync()

    # 측정
    measured = 0
    total_imgs = 0
    t0 = time.perf_counter()
    for x, _ in dl_fast:
        if measured >= measure_batches:
            break
        x = x.to(device, non_blocking=(device.type == 'cuda'))
        with torch.no_grad():
            _ = model(x)
        measured += 1
        total_imgs += x.size(0)
    _sync()
    t1 = time.perf_counter()
    elapsed_fast = t1 - t0
    ms_per_batch_fast = (elapsed_fast / max(1, measured)) * 1000.0
    ips_fast = total_imgs / max(elapsed_fast, 1e-9)
    print(f"[RESULT] tuned:    {measured} batches, {total_imgs} images | "
          f"{ms_per_batch_fast:.2f} ms/batch | {ips_fast:.1f} imgs/s")

    # ---------------------------
    # 요약
    # ---------------------------
    if elapsed_base > 0 and elapsed_fast > 0:
        speedup = elapsed_base / elapsed_fast
        print(f"\n[SUMMARY] imgs/s: {ips_base:.1f} -> {ips_fast:.1f} | "
              f"ms/batch: {ms_per_batch_base:.2f} -> {ms_per_batch_fast:.2f} | "
              f"speedup x {speedup:.2f}")
    else:
        print("\n[SUMMARY] 측정 시간 0으로 계산 불가")
