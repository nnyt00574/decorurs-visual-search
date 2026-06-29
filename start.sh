#!/usr/bin/env bash
# Brings up the full stack and opens the app in your browser.
# Run from the project root: ./start.sh
set -e

if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Starting Qdrant, API, and frontend..."
docker compose up -d --build

echo "Waiting for services to settle..."
sleep 5

echo "Opening http://localhost:3000 ..."
open http://localhost:3000

echo ""
echo "Done. If this is the first run, the catalog isn't indexed yet:"
echo "  docker compose run --rm indexer"
