import torch
from collections import defaultdict

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, "SAM requires non-negative rho."
        self.rho = rho
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.state = defaultdict(dict)  # <- state 변수 초기화

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p.device)
                p.add_(e_w)  # perturbation

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]  # 복원
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self):
        raise NotImplementedError("Call first_step and second_step instead")

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norms.append(torch.norm(p.grad.detach(), p=2))
        return torch.norm(torch.stack(norms), p=2).to(shared_device)
    
# --- 여기부터: 추가할 유틸 함수 ---
@torch.no_grad()
def sam_restore(opt: SAM):
    """
    SAM의 first_step 이후 파라미터를 원복하고, 불필요한 state 키를 정리한다.
    - 정상 루틴: second_step() 이후에 호출해 state를 깨끗하게 유지
    - 예외 루틴: first_step() 이후 오류 시 finally에서 호출해 파라미터 원복
    """
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state[p]
            if "old_p" in st:
                p.data.copy_(st["old_p"])  # 원복
                del st["old_p"]            # 메모리 정리
            # 아래 키는 없을 수도 있으므로 안전하게 제거
            st.pop("e_w", None)
            st.pop("grad_backup", None)