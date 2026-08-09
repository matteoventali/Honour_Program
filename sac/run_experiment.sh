#!/usr/bin/env bash
set -Eeuo pipefail

framework_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
image_name="${IMAGE_NAME:-tesi-sac}"
container_name="${CONTAINER_NAME:-tesi-sac-$(date +%Y%m%d-%H%M%S)-$$}"

docker build -t "$image_name" "$framework_dir"
docker run --rm --gpus all --entrypoint python "$image_name" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA non disponibile nel container"; print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}")'
docker run -d \
  --name "$container_name" \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$framework_dir,target=/workspace" \
  "$image_name" "$@"

printf '\nEsperimento avviato: %s\n' "$container_name"
printf 'Log:       docker logs -f %s\n' "$container_name"
printf 'Stato:     docker ps -a --filter name=%s\n' "$container_name"
printf 'Stop:      docker stop %s\n' "$container_name"
printf 'Risultati: %s/{results,img,logs,policy}\n' "$framework_dir"
