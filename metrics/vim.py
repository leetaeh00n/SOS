import torch
import numpy as np
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from torch import Tensor

class ViM:
    def __init__(self, model, d, w, b):
        """
        ViM 모델 초기화
        :param model: 신경망 모델 (특징 추출용)
        :param d: 주요 서브스페이스의 차원
        :param w: 네트워크의 마지막 레이어 가중치
        :param b: 네트워크의 마지막 레이어 바이어스
        """
        self.model = model
        self.n_dim = d
        self.w = w.detach().cpu().numpy()
        self.b = b.detach().cpu().numpy()
        self.u = -np.matmul(pinv(self.w), self.b)  # 새로운 원점
        self.principal_subspace = None
        self.alpha = None

    def _get_logits(self, features):
        """특징으로부터 로짓 계산"""
        assert features.shape[1] == self.w.shape[1], (
            f"Feature dimension mismatch: features.shape[1] ({features.shape[1]}) "
            f"!= self.w.shape[1] ({self.w.shape[1]})"
        )
        return np.matmul(features, self.w.T) + self.b

    def fit(self, data_loader, device="cpu"):
        """
        특징과 로짓을 추출하고 주요 서브스페이스 및 알파 값을 계산
        :param data_loader: 데이터 로더
        :param device: 실행 디바이스
        """
        features, labels = self._extract_features(data_loader, device)
        self.fit_features(features, labels)

    def _extract_features(self, data_loader, device):
        """데이터 로더에서 특징 추출"""
        features_list = []
        labels_list = []

        self.model.eval()
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(device)
                _, features = self.model.forward_virtual(data)
                features_list.append(features.cpu())
                labels_list.append(labels.cpu())

        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return features, labels

    def fit_features(self, features, labels):
        """
        특징 및 로짓으로 주요 서브스페이스와 알파 계산
        """
        from sklearn.covariance import EmpiricalCovariance

        features = features.cpu().numpy()

        if features.shape[1] < self.n_dim:
            n = features.shape[1] // 2
            print(f"특징 차원이 주요 서브스페이스 차원보다 작음: {features.shape[1]=}, {self.n_dim=}. {n}로 조정됨")
            self.n_dim = n

        logits = self._get_logits(features)

        print("주요 서브스페이스 계산 중...")
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(features - self.u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

        largest_eigvals_idx = np.argsort(eig_vals * -1)[self.n_dim:]
        self.principal_subspace = np.ascontiguousarray((eigen_vectors.T[largest_eigvals_idx]).T)

        print("알파 계산 중...")
        x_p_t = np.matmul(features - self.u, self.principal_subspace)
        vlogits = norm(x_p_t, axis=-1)
        self.alpha = logits.max(axis=-1).mean() / vlogits.mean()
        print(f"Alpha 값: {self.alpha:.4f}")

    def predict(self, x):
        """
        주어진 입력에 대한 ViM 점수 예측
        """
        with torch.no_grad():
            _, features = self.model.forward_virtual(x)
        return self.predict_features(features)

    def predict_features(self, features):
        """
        모델에서 생성된 특징에 대한 ViM 점수 계산
        """
        features = features.detach().cpu().numpy()
        logits = self._get_logits(features)

        x_p_t = norm(np.matmul(features - self.u, self.principal_subspace), axis=-1)
        vlogit = x_p_t * self.alpha
        energy = logsumexp(np.clip(logits, -100, 100), axis=-1)
        score = -vlogit + energy
        return Tensor(score)
