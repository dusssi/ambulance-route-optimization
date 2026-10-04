#!/usr/bin/env bash
# Install the declared stack and prefer headless OpenCV when this environment
# lacks the system libGL runtime required by the GUI-enabled wheel pulled by
# Ultralytics. Both wheels expose the same cv2 API for this project.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt
if ! "$PYTHON_BIN" -c 'import cv2' >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip uninstall -y opencv-python || true
  "$PYTHON_BIN" -m pip install --force-reinstall --no-deps 'opencv-python-headless>=4.10,<5'
fi
"$PYTHON_BIN" -c 'import cv2; print("OpenCV", cv2.__version__)'
