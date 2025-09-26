#!/bin/bash
set -euo pipefail

SERVICE_NAME="groot"       # service name in docker-compose
CONTAINER_NAME="groot15"   # container_name in docker-compose

# ---- Compose isolation ----
PROJECT_NAME="edu_groot" 
export COMPOSE_PROJECT_NAME="${PROJECT_NAME}"
export COMPOSE_IGNORE_ORPHANS=true

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_DIR="${SCRIPT_DIR}/docker"
DOCKER_COMPOSE_FILE="${COMPOSE_DIR}/docker-compose-24.04.yml"

if [ ! -f "${DOCKER_COMPOSE_FILE}" ]; then
  echo "❌ Cannot find ${DOCKER_COMPOSE_FILE}"
  echo "   Make sure docker-compose.yml is in ~/Isaac-GR00T/docker/"
  exit 1
fi

compose() {
  docker compose \
    -p "${PROJECT_NAME}" \
    -f "${DOCKER_COMPOSE_FILE}" \
    --project-directory "${COMPOSE_DIR}" \
    "$@"
}

is_running() {
  docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

menu() {
  if is_running; then
    echo "1) Build and run (background)"
    echo "2) 🟢 Connect to container"
  else
    echo "1) Build and run (background)"
    echo "2) 🔴 Connect to container"
  fi
  echo "3) List docker containers"
  echo "4) Docker re-build && run"
  echo "5) Stop container"
  echo "6) Exit (or press ESC)"
  echo -n "Choice: "
}

while true; do
  menu
  read -rsn1 c
  [[ $c == $'\e' ]] && echo -e "\n[ESC] Exit" && break
  echo ""

  case "${c:-1}" in
    1)
      if is_running; then
        echo "⚠️  '${CONTAINER_NAME}' is already running."
        read -p "Restart ONLY this service? (y/N): " yn
        if [[ "${yn:-n}" =~ ^[Yy]$ ]]; then
          # Recreate only this service. Do not touch deps or other services.
          compose up -d --build --no-deps --force-recreate "${SERVICE_NAME}"
        else
          echo "➡️  Leaving it running."
        fi
      else
        # Build and start only this service; don't touch deps or other services.
        compose build "${SERVICE_NAME}"
        compose up -d --no-deps "${SERVICE_NAME}"
      fi
      ;;
    2)
      if is_running; then
        docker exec -it "${CONTAINER_NAME}" bash
      else
        echo "❌ Container '${CONTAINER_NAME}' is not running."
        echo "   Use option 1 to start it."
      fi
      ;;
    3)
      docker ps
      ;;
    4)
      # Full rebuild of only this service; don’t affect others.
      compose build --no-cache "${SERVICE_NAME}"
      compose up -d --no-deps --force-recreate "${SERVICE_NAME}"
      ;;
    5)
      echo "🛑 Stopping ${CONTAINER_NAME} only…"
      compose stop "${SERVICE_NAME}" || true
      compose rm -f "${SERVICE_NAME}" || true
      ;;
    6)
      echo "Exit"
      break
      ;;
    *)
      echo "Invalid option"
      ;;
  esac
  echo ""
done
