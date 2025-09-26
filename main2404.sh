#!/bin/bash
set -euo pipefail

CONTAINER_NAME="groot15"

# Script and compose location (works no matter where you run it from)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_DIR="${SCRIPT_DIR}/docker"
DOCKER_COMPOSE_FILE="${COMPOSE_DIR}/docker-compose-24.04.yml"

# Check that docker-compose.yml exists
if [ ! -f "${DOCKER_COMPOSE_FILE}" ]; then
  echo "❌ Cannot find ${DOCKER_COMPOSE_FILE}"
  echo "   Make sure docker-compose.yml is in ~/Isaac-GR00T/docker/"
  exit 1
fi

menu() {
  if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
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
      if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "⚠️  '${CONTAINER_NAME}' is already running."
        read -p "Restart? (y/N): " yn
        if [[ "${yn:-n}" =~ ^[Yy]$ ]]; then
          docker compose -f "${DOCKER_COMPOSE_FILE}" up -d --build
        else
          echo "➡️  Leaving it running."
        fi
      else
        docker compose -f "${DOCKER_COMPOSE_FILE}" build
        docker compose -f "${DOCKER_COMPOSE_FILE}" up -d --remove-orphans
      fi
      ;;
    2) docker exec -it "${CONTAINER_NAME}" bash ;;
    3) docker ps ;;
    4)
      docker compose -f "${DOCKER_COMPOSE_FILE}" build --no-cache
      docker compose -f "${DOCKER_COMPOSE_FILE}" up -d --remove-orphans
      ;;
    5) echo "🛑 Stopping..."; docker compose -f "${DOCKER_COMPOSE_FILE}" down ;;
    6) echo "Exit"; break ;;
    *) echo "Invalid option" ;;
  esac
  echo ""
done
