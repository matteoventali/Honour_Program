#!/usr/bin/env bash
set -Eeuo pipefail

framework_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
image_name="${IMAGE_NAME:-tesi-multilevel-convention}"
container_name="${CONTAINER_NAME:-tesi-multilevel-convention-$(date +%Y%m%d-%H%M%S)-$$}"

docker build -t "$image_name" "$framework_dir"
docker run -d \
  --name "$container_name" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$framework_dir,target=/workspace" \
  "$image_name" "$@"

printf '\nEsperimento avviato: %s\n' "$container_name"
printf 'Log:       docker logs -f %s\n' "$container_name"
printf 'Stato:     docker ps -a --filter name=%s\n' "$container_name"
printf 'Stop:      docker stop %s\n' "$container_name"
printf 'Risultati: %s/{results,img,logs,policy}\n' "$framework_dir"
