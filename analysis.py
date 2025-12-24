# [Cell 1] 라이브러리 및 설정
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm  # 진행상황 표시용

from utils.dataloader import get_dataloader

# 그래프 스타일
sns.set(style="whitegrid")
colors = {'ID': 'blue', 'vOOD': 'red', 'iSUN': 'green', 'Texture': 'orange'}
plt.rcParams['figure.figsize'] = (12, 6)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# [Cell 2] 데이터 로드 및 Real OOD Feature 추출

# ================= 사용자 설정 =================
# 1. 파일 경로 설정
seed = 0 # 0,1,2,3,4
metric = "auroc_ma" # energy_metric
root_dir = f"./sos_rho_schedule/ce_binary_E100/seed{seed}/{metric}/cifar10"
target_feat_folder = f"feat_ce_binary_s{seed}_wrn_mode_{metric}_rho0.0-0.5_E0.1"
epoch = 100

# 2. 데이터셋 설정
base_data = "cifar10"
ood_datasets = ["iSUN", "texture"]  # "texture"는 보통 'dtd'로 불립니다. (dataloader 확인 필요)
# =============================================

# 1) ID & vOOD (Saved .npy) 로드
feat_dir_path = os.path.join(root_dir, target_feat_folder)
orig_path = os.path.join(feat_dir_path, f"features_ep{epoch:03d}_orig.npy")
pert_path = os.path.join(feat_dir_path, f"features_ep{epoch:03d}_pert.npy")

data_dict = {}

try:
    data_dict['ID'] = np.load(orig_path)
    data_dict['vOOD'] = np.load(pert_path)
    print(f"✅ [Saved] ID: {data_dict['ID'].shape}, vOOD: {data_dict['vOOD'].shape}")
except FileNotFoundError:
    raise FileNotFoundError("ID/vOOD .npy 파일을 찾을 수 없습니다. 경로를 확인하세요.")

# 2) Model 로드 (Real OOD 추출용)
# feat_ -> models_ 로 변환하여 모델 경로 추론
model_folder = target_feat_folder.replace("feat_", "models_")
ckpt_path = os.path.join(root_dir, model_folder, f"model_ep{epoch}.pth")

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"모델 체크포인트가 필요합니다: {ckpt_path}")

print(f"🔄 Loading Model from {ckpt_path}...")
# 모델 아키텍처는 utils/config 등에서 가져오거나 직접 선언해야 함
# 여기서는 WideResNet 가정 (import 필요)
from model.WideResNet import WideResNet 
# from model.cifar_resnet import resnet18 # ResNet인 경우

# 모델 생성 (Args에 맞게 파라미터 수정 필요)
num_classes = 10 if base_data == 'cifar10' else 100
model = WideResNet(depth=40, widen_factor=2, dropRate=0.3, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# 3) Real OOD Feature Extraction Function
def extract_features(loader, model):
    feat_list = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting"):
            images = images.to(device)
            # model.forward_virtual(x) returns (logits, features)
            _, features = model.forward_virtual(images)
            feat_list.append(features.cpu().numpy())
    return np.concatenate(feat_list, axis=0)

# 4) Real OOD 로드 및 추출 실행
for ood_name in ood_datasets:
    print(f"Processing Real OOD: {ood_name}...")
    try:
        # User Provided Code Snippet
        ood_loader = get_dataloader(device=device, base_data=base_data, dataname=ood_name, batch_size=200, phase='ood')
        
        # Feature 추출
        feat_real_ood = extract_features(ood_loader, model)
        data_dict[ood_name] = feat_real_ood
        print(f"✅ [Extracted] {ood_name}: {feat_real_ood.shape}")
        
    except Exception as e:
        print(f"⚠️ {ood_name} 데이터 로드 실패: {e}")

# 최종 확인
print("\n📊 Final Data Shapes:")
for k, v in data_dict.items():
    print(f" - {k}: {v.shape}")


# [Cell 3] Activation Distribution Analysis

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# 1. Global Activation (모든 값 펼쳐서)
for name, feat in data_dict.items():
    # 데이터가 너무 많으면 downsampling
    flat_feat = feat.flatten()
    if len(flat_feat) > 100000: flat_feat = np.random.choice(flat_feat, 100000, replace=False)
    
    sns.kdeplot(flat_feat, label=name, color=colors.get(name, 'gray'), ax=axes[0], fill=True, alpha=0.1)

axes[0].set_title("1. Global Activation Value Distribution")
axes[0].set_xlabel("Activation Value")
axes[0].legend()

# 2. Sample-wise Mean Activation
for name, feat in data_dict.items():
    sample_mean = feat.mean(axis=1)
    sns.kdeplot(sample_mean, label=name, color=colors.get(name, 'gray'), ax=axes[1], fill=True, alpha=0.1)

axes[1].set_title("2. Sample-wise Mean Activation")
axes[1].set_xlabel("Mean Value")

# 3. Sample-wise Max Activation (Peak Response)
for name, feat in data_dict.items():
    sample_max = feat.max(axis=1)
    sns.kdeplot(sample_max, label=name, color=colors.get(name, 'gray'), ax=axes[2], fill=True, alpha=0.1)

axes[2].set_title("3. Sample-wise MAX Activation")
axes[2].set_xlabel("Max Value")

plt.tight_layout()
plt.show()


# [Cell 4] Feature L2 Norm Distribution

plt.figure(figsize=(10, 6))

for name, feat in data_dict.items():
    # L2 Norm 계산
    norms = np.linalg.norm(feat, axis=1)
    
    # Histogram 그리기
    sns.kdeplot(norms, label=f"{name} (μ={norms.mean():.2f})", 
                color=colors.get(name, 'gray'), fill=True, alpha=0.1)

plt.title(f"Feature L2 Norm Distribution (Epoch {epoch})")
plt.xlabel("L2 Norm (Magnitude)")
plt.ylabel("Density")
plt.legend()
plt.show()


# [Cell 5] Energy Score Analysis
# 모델의 가중치를 가져와서 모든 Feature에 대해 일관되게 계산합니다.

# FC Layer Weights 추출
state_dict = model.state_dict()
keys = state_dict.keys()
# 키 이름 찾기 (fc.weight or linear.weight)
w_key = [k for k in keys if 'fc.weight' in k or 'linear.weight' in k][0]
b_key = [k for k in keys if 'fc.bias' in k or 'linear.bias' in k][0]

weight = state_dict[w_key].cpu().numpy()
bias = state_dict[b_key].cpu().numpy()
temp = 1.0

def compute_energy_numpy(features, W, b, T=1.0):
    logits = np.matmul(features, W.T) + b
    # LogSumExp trick for stability
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp((logits - max_logits) / T)
    log_sum_exp = np.log(np.sum(exp_logits, axis=1)) + max_logits.squeeze() / T
    energy = -T * log_sum_exp
    return energy

plt.figure(figsize=(10, 6))

for name, feat in data_dict.items():
    energy = compute_energy_numpy(feat, weight, bias, temp)
    
    sns.kdeplot(energy, label=f"{name} (μ={energy.mean():.2f})", 
                color=colors.get(name, 'gray'), fill=True, alpha=0.1)

plt.title("Energy Score Distribution (Lower Energy = ID-like)")
plt.xlabel("Energy Score")
plt.ylabel("Density")
plt.legend()
plt.show()


# [Cell 6] t-SNE Visualization (Sampled)
# 전체 데이터는 너무 많으므로 클래스별로 N개씩 샘플링

n_sample_per_class = 5000
tsne_data = []
tsne_labels = []

print("Sampling data for t-SNE...")
for name, feat in data_dict.items():
    if len(feat) > n_sample_per_class:
        idx = np.random.choice(len(feat), n_sample_per_class, replace=False)
        sampled = feat[idx]
    else:
        sampled = feat
    
    tsne_data.append(sampled)
    tsne_labels.extend([name] * len(sampled))

X_all = np.concatenate(tsne_data, axis=0)
y_all = np.array(tsne_labels)

# Run t-SNE
print(f"Running t-SNE on {X_all.shape[0]} samples...")
tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
X_embedded = tsne.fit_transform(X_all)

# Plotting
plt.figure(figsize=(12, 10))
for name in data_dict.keys(): # 순서대로 그리기
    indices = np.where(y_all == name)
    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], 
                c=colors.get(name, 'gray'), label=name, s=15, alpha=0.6)

plt.title(f"t-SNE Visualization of Feature Space (Epoch {epoch})")
plt.legend()
plt.show()