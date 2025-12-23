import torch
from utils.sam import SAM, sam_restore
from utils.tools import compute_energy


def extract_all_sam_features_for_epoch(model, dataloader, device, args, criterion, rho_ood):
    """
    Unified Feature Extraction
    """
    model.eval()
    model.zero_grad()
    
    # 임시 SAM 옵티마이저 생성
    temp_sam = SAM(
        model.parameters(),
        base_optimizer=torch.optim.SGD,
        rho=rho_ood,
        lr=0.1,
        momentum=0.9,
    )

    all_feat_clean = []
    all_feat_pert = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 1) Gradient Calculation Step
        temp_sam.zero_grad()
        with torch.enable_grad():
            logits_clean, feat_clean = model.forward_virtual(images)
            
            if args.ood_gen_mode == 'energy':
                energy_clean = compute_energy(logits_clean, T=args.temperature)
                loss = energy_clean.mean()
            else: # 'ce'
                loss = criterion(logits_clean, labels)
            
            loss.backward()

        # 2) SAM Perturbation Step
        temp_sam.first_step(zero_grad=True)

        # 3) Get Perturbed Features
        with torch.no_grad():
            _, feat_pert = model.forward_virtual(images)

        sam_restore(temp_sam)

        all_feat_clean.append(feat_clean.cpu())
        all_feat_pert.append(feat_pert.cpu())
        all_labels.append(labels.cpu())

        # Clear gradients manually to be safe
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    model.train()
    return (
        torch.cat(all_feat_clean),
        torch.cat(all_feat_pert),
        torch.cat(all_labels),
    )