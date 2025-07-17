   #!/usr/bin/env bash
set -euo pipefail


its=2000
echo "▶ Training tuned simple for ${its} iters…"
python train_pong.py \
  --mode train \
  --strategy simple \
  --iters "${its}"

its=3000
echo "▶ Training tuned dense for ${its} iters…"
python train_pong.py \
  --mode train \
  --strategy dense \
  --iters "${its}"

# 2) selfplay train with a custom profile for 5 iterations
its=6000
echo "▶ Training selfplay (train) for ${its} iters with custom profile…"
python train_pong.py \
  --mode train \
  --strategy selfplay \
  --profile configs/selfplay_local.json \
  --iters "${its}"