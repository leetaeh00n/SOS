import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.sephead import LossPredLoss
from utils.tools import compute_energy


def train_epoch(args, model, loader, epoch, device, criterion, sam_optimizer, scheduler, 
                save_epochs, save_dir, energy_head=None, sep_head=None, vood_loader=None):
    
    model.train()
    if energy_head: energy_head.train()
    if sep_head: sep_head.train()

    collect = epoch in save_epochs
    if collect: feat_orig_list, feat_pert_list = [], []
    
    avg_loss = 0.0
    n_seen = 0
    stage2 = (epoch > args.start_epoch)

    id_energy_list = []
    vood_energy_list = []

    if stage2:
        if vood_loader is None:
            raise ValueError("Need vood_loader for Stage 2")
        assert len(vood_loader) == len(loader), f"ID steps={len(loader)} != vOOD steps={len(vood_loader)}"
        vood_iter = iter(vood_loader)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- Stage 1: Standard SGD Training ---
        if not stage2:
            sam_optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            sam_optimizer.first_step(zero_grad=True)
            
            logits_p = model(images)
            loss_p = criterion(logits_p, labels)
            loss_p.backward()
            sam_optimizer.second_step(zero_grad=True)
            current = loss_p.item()

        # --- Stage 2: vOOD Training ---
        else:
            feat_pert = next(vood_iter)[0].to(device)

            with torch.no_grad():
                _, feat_orig = model.forward_virtual(images)
            
            if collect:
                feat_orig_list.append(feat_orig.detach().cpu())
                feat_pert_list.append(feat_pert.detach().cpu())

            # Step 1 of SAM
            sam_optimizer.rho = args.rho
            sam_optimizer.zero_grad()
            
            logits_id1, feat_id1 = model.forward_virtual(images)
            L_cls1 = criterion(logits_id1, labels)
            total_loss1 = L_cls1
            
            feat_p1 = feat_pert.detach().requires_grad_(True)
            v_logit_p1 = model.fc(feat_p1)
            
            # [Branch Logic]
            if args.ood_train_mode == 'binary':
                energy_id1 = compute_energy(logits_id1, T=args.temperature).unsqueeze(1)
                energy_vood1 = compute_energy(v_logit_p1, T=args.temperature).unsqueeze(1)
                energy_all1 = torch.cat([energy_id1, energy_vood1], dim=0)
                
                bin_labels = torch.cat([
                    torch.ones(energy_id1.size(0), dtype=torch.long, device=device),
                    torch.zeros(energy_vood1.size(0), dtype=torch.long, device=device)
                ], dim=0)
                
                bin_logits1 = energy_head(energy_all1)
                L_energy1 = criterion(bin_logits1, bin_labels)
                total_loss1 += args.lambda_energy * L_energy1
                
            elif args.ood_train_mode == 'regularization':
                with torch.no_grad():
                    v_pseudo_labels = v_logit_p1.argmax(dim=1)

                if args.use_sep_loss:
                    L_sep1 = F.cross_entropy(v_logit_p1, v_pseudo_labels)
                    total_loss1 -= args.lambda_sep * L_sep1
                
                if args.use_kl_loss:
                    uniform_dist = torch.ones(logits_id1.size(1), device=device) / logits_id1.size(1)
                    L_kl1 = F.kl_div(F.log_softmax(v_logit_p1, dim=-1),
                                     uniform_dist.expand(v_logit_p1.size(0), -1),
                                     reduction='batchmean')
                    total_loss1 += args.lambda_kl * L_kl1
                    
                if args.use_rank_loss and sep_head is not None:
                    id_pseudo = logits_id1.argmax(dim=1)
                    id_conf = F.cross_entropy(logits_id1, id_pseudo, reduction='none')
                    v_conf = F.cross_entropy(v_logit_p1, v_pseudo_labels, reduction='none')
                    
                    combined_target = torch.cat([id_conf, v_conf], dim=0)
                    combined_feat = torch.cat([feat_id1.detach(), feat_p1], dim=0)
                    combined_pred = sep_head(combined_feat).squeeze()
                    
                    L_rank1 = LossPredLoss(combined_pred, combined_target, margin=args.margin)
                    total_loss1 += args.lambda_rank * L_rank1

            total_loss1.backward()
            sam_optimizer.first_step(zero_grad=True)

            # Step 2 of SAM
            logits_id2, feat_id2 = model.forward_virtual(images)
            L_cls2 = criterion(logits_id2, labels)
            total_loss2 = L_cls2
            
            feat_p2 = feat_pert.detach().requires_grad_(True)
            v_logit_p2 = model.fc(feat_p2)

            if args.ood_train_mode == 'binary':
                energy_id2 = compute_energy(logits_id2, T=args.temperature).unsqueeze(1)
                energy_vood2 = compute_energy(v_logit_p2, T=args.temperature).unsqueeze(1)
                energy_all2 = torch.cat([energy_id2, energy_vood2], dim=0)
                
                bin_labels = torch.cat([
                    torch.ones(energy_id2.size(0), dtype=torch.long, device=device),
                    torch.zeros(energy_vood2.size(0), dtype=torch.long, device=device)
                ], dim=0)
                
                bin_logits2 = energy_head(energy_all2)
                L_energy2 = criterion(bin_logits2, bin_labels)
                total_loss2 += args.lambda_energy * L_energy2

                id_energy_list.append(energy_id2.detach().cpu())
                vood_energy_list.append(energy_vood2.detach().cpu())

            elif args.ood_train_mode == 'regularization':
                with torch.no_grad(): v_pseudo_labels = v_logit_p2.argmax(dim=1)
                if args.use_sep_loss:
                    L_sep2 = F.cross_entropy(v_logit_p2, v_pseudo_labels)
                    total_loss2 -= args.lambda_sep * L_sep2
                if args.use_kl_loss:
                    uniform_dist = torch.ones(logits_id2.size(1), device=device) / logits_id2.size(1)
                    L_kl2 = F.kl_div(F.log_softmax(v_logit_p2, dim=-1),
                                     uniform_dist.expand(v_logit_p2.size(0), -1),
                                     reduction='batchmean')
                    total_loss2 += args.lambda_kl * L_kl2
                if args.use_rank_loss and sep_head is not None:
                    id_pseudo = logits_id2.argmax(dim=1)
                    id_conf = F.cross_entropy(logits_id2, id_pseudo, reduction='none')
                    v_conf = F.cross_entropy(v_logit_p2, v_pseudo_labels, reduction='none')
                    combined_target = torch.cat([id_conf.detach(), v_conf.detach()], dim=0)
                    combined_feat = torch.cat([feat_id2.detach(), feat_p2], dim=0)
                    combined_pred = sep_head(combined_feat).squeeze()
                    L_rank2 = LossPredLoss(combined_pred, combined_target, margin=args.margin)
                    total_loss2 += args.lambda_rank * L_rank2

            total_loss2.backward()
            sam_optimizer.second_step(zero_grad=True)
            current = total_loss2.item()
            n_seen += images.size(0)

        scheduler.step()
        avg_loss = 0.8 * avg_loss + 0.2 * current

    if collect and n_seen > 0:
        np.save(os.path.join(save_dir, f"features_ep{epoch:03d}_orig.npy"), torch.cat(feat_orig_list).numpy())
        np.save(os.path.join(save_dir, f"features_ep{epoch:03d}_pert.npy"), torch.cat(feat_pert_list).numpy())
    
    if stage2 and args.ood_train_mode == 'binary' and len(id_energy_list) > 0:
        id_energy_all = torch.cat(id_energy_list, dim=0).view(-1)
        vood_energy_all = torch.cat(vood_energy_list, dim=0).view(-1)
        return avg_loss, id_energy_all, vood_energy_all
    else:
        return avg_loss, None, None
    

# **Evaluation Function**
def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():  # No gradient computation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(dataloader)
    return avg_loss, accuracy