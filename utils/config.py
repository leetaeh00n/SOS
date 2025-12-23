import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Virtual-OOD Training")
    
    # --- Mode Selectors ---
    parser.add_argument("--ood_gen_mode", type=str, default="energy", choices=["energy", "ce"],
                        help="Method to generate vOOD: 'energy' (ascent) or 'ce' (loss ascent)")
    parser.add_argument("--ood_train_mode", type=str, default="binary", choices=["binary", "regularization"],
                        help="Training objective: 'binary' (Energy Head) or 'regularization' (KL/Sep/Rank)")

    # WandB
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument('--use_val', action='store_true', default=False,
                        help='Use validation split from training data')
    
    # Mode Specific
    parser.add_argument("--lambda_energy", type=float, default=0.1)
    parser.add_argument('--use_sep_loss', action='store_true', default=False)
    parser.add_argument('--use_kl_loss', action='store_true', default=False)
    parser.add_argument('--use_rank_loss', action='store_true', default=False)
    parser.add_argument('--lambda_sep', type=float, default=0.1)
    parser.add_argument('--lambda_kl', type=float, default=0.1)
    parser.add_argument('--lambda_rank', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--use_select', action='store_true')

    # Hyperparams
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--base_data", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=40)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--model_name", type=str, default="WideResNet", choices=["WideResNet", "ResNet"])
    
    # Model Arch
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--drop_rate", type=float, default=0.3)

    # System
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_dir_base", type=str, default="./unified_exp")
    
    # vOOD Param
    parser.add_argument("--rho_ood_mode", type=str, default="linear_dec", 
                        choices=["energy_metric", "auroc_ma", "linear_inc", "linear_dec", "cosine", "const"])
    parser.add_argument("--rho_ood_min", type=float, default=0.1)
    parser.add_argument("--rho_ood_max", type=float, default=0.5)

    return parser.parse_args()