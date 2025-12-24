#!/bin/bash
nohup bash -c '
set -e

# --- 1. 환경 설정 ---
source ~/anaconda3/etc/profile.d/conda.sh  # 본인의 conda 경로 확인 필요
conda activate SOS

mkdir -p ./log

# --- 2. 기본 하이퍼파라미터 ---
batch_size=128
base_data="cifar100"      # cifar10 or cifar100
model_name="DenseNet"  # WideResNet or ResNet


# EPOCHS=200
# START_EPOCHS=80

EPOCHS_LIST=(100 200 500)

# --- 3. 실험 변수 (Loop 대상) ---
seeds=(0 1 2 3 4)
# rho_modes=("auroc_ma" "linear_inc" "linear_dec" "cosine" "energy_metric" "const")
rho_modes=("auroc_ma" "energy_metric")

# "GenMode:TrainMode" 형식
MODE_LIST=(
  "ce:binary"
)

# Regularization 모드일 때 사용 (Sep:KL:Rank:Margin)
# Binary 모드일 때는 무시되거나 고정값 사용 (아래 로직 참조)
LAMBDA=(
  "0.1:0.1:0.1:1.0"
)
for EPOCHS in "${EPOCHS_LIST[@]}"; do

  # EPOCHS에 따른 START_EPOCHS 자동 설정
  case $EPOCHS in
    100)
      START_EPOCHS=40
      ;;
    200)
      START_EPOCHS=80
      ;;
    500)
      START_EPOCHS=200
      ;;
    *)
      echo "Unsupported EPOCHS=${EPOCHS}"
      exit 1
      ;;
  esac
    # --- 4. 실험 루프 시작 ---
  for seed in "${seeds[@]}"; do
    for rho_mode in "${rho_modes[@]}"; do
      for mode_pair in "${MODE_LIST[@]}"; do
        for lambda_set in "${LAMBDA[@]}"; do

            # 문자열 파싱 (예: ce:binary -> GEN_MODE=ce, TRAIN_MODE=binary)
            IFS=":" read -r GEN_MODE TRAIN_MODE <<< "${mode_pair}"

            # 실험별 Argument 동적 생성
            current_args=""
            
            # [학습 방식에 따른 추가 파라미터 설정]
            if [ "$TRAIN_MODE" == "binary" ]; then
            # Binary Mode (Energy Head)
            # 필요하다면 lambda_set에서 값을 가져오거나 고정값 사용
            current_args="--lambda_energy 0.1"
            
            elif [ "$TRAIN_MODE" == "regularization" ]; then
            # Regularization Mode (KL/Sep/Rank)
            # lambda_sep:lambda_kl:lambda_rank:margin 순서 파싱
            IFS=":" read -r l_sep l_kl l_rank marg <<< "${lambda_set}"
            current_args="--use_kl_loss --lambda_kl ${l_kl} \
                            --use_sep_loss --lambda_sep ${l_sep} \
                            --use_rank_loss --lambda_rank ${l_rank} \
                            --margin ${marg}"
            fi

            # 데이터셋에 따른 rho 범위 설정 (필요시 수정)
            if [ "$base_data" == "cifar10" ]; then
            rho_min=0.0
            rho_max=0.5
            GPU_ID=6
            elif [ "$base_data" == "cifar100" ]; then
            rho_min=0.0
            rho_max=1.0
            GPU_ID=4
            fi

            # 실행 ID 및 로그 파일명 생성
            RUN_ID="${base_data}_${model_name}_${GEN_MODE}_${TRAIN_MODE}_ep${EPOCHS}_seed${seed}_${rho_mode}"
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            OUT_LOG_FILE="./log/${TIMESTAMP}_${RUN_ID}.out"
            ERR_LOG_FILE="./log/${TIMESTAMP}_${RUN_ID}.err"

            echo "========================================================"
            echo "[$(date)] Starting Run: ${RUN_ID}"
            echo " -> Gen: ${GEN_MODE}, Train: ${TRAIN_MODE}, Rho: ${rho_mode}, GPU: ${GPU_ID}"
            echo "========================================================"

            SAVE_DIR="./sos_rho_schedule/${GEN_MODE}_${TRAIN_MODE}_E${EPOCHS}/seed${seed}/${rho_mode}/"

            # --- python main.py 실행 ---
            if ! CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
                --use_wandb \
                --batch_size ${batch_size} \
                --base_data ${base_data} \
                --model_name ${model_name} \
                --epochs ${EPOCHS} \
                --start_epoch ${START_EPOCHS} \
                --rho_ood_mode ${rho_mode} \
                --rho_ood_min ${rho_min} \
                --rho_ood_max ${rho_max} \
                --seed ${seed} \
                --temperature 1.0 \
                --save_dir_base ${SAVE_DIR} \
                --ood_gen_mode ${GEN_MODE} \
                --ood_train_mode ${TRAIN_MODE} \
                ${current_args} \
                > "${OUT_LOG_FILE}" 2> "${ERR_LOG_FILE}"; then
            
            echo "[$(date)] ERROR: Run failed for ${RUN_ID}" | tee -a "./log/failed_runs_${TIMESTAMP}.log"
            # 에러 발생 시 로그 내용을 일부 출력하여 확인 (선택 사항)
            tail -n 20 "${ERR_LOG_FILE}"
            else
            echo "[$(date)] SUCCESS: Run completed for ${RUN_ID}"
            rm -f "${ERR_LOG_FILE}"  # 성공 시 에러 로그 삭제
            fi

            sleep 5
        done
      done
    done
  done
done
echo "[$(date)] All unified experiments done."
' > ./nohup_launcher_densenet.out 2>&1 &