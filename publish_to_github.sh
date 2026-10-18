#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="fast-yolov8-broadwell"

if ! command -v git >/dev/null 2>&1; then
  echo "Git is missing. Install the Git command-line tools first." >&2
  exit 1
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is missing. On macOS: brew install gh" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Please sign in to GitHub. Follow the prompts below."
  gh auth login
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "Initialize YOLOv8 Broadwell starter project"
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "This repository already has an origin remote; skipping creation."
  git push -u origin HEAD
else
  echo "Creating a PRIVATE GitHub repository: ${REPO_NAME}"
  gh repo create "$REPO_NAME" --private --source=. --remote=origin --push
fi

echo "Success. Your repository is published."
gh repo view --web
