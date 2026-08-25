#!/usr/bin/env bash
# Launch Jupyter Lab with the project venv, free of any leaked PYTHONPATH.
# Usage: double-click in Git Bash, or run: ./start_jupyter.sh
cd "$(dirname "$0")"
unset PYTHONPATH
exec .venv/Scripts/jupyter-lab.exe
