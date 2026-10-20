# Resuming the final v2 run after sleep, reboot, or a crash

Every step appends finished results to `results/` and skips work already saved, so the restart is safe at any time.

1. Start the pinned Ollama if it is not running:
   `cd ~/opt/ollama-0.13.1 && OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_KEEP_ALIVE=24h nohup ./ollama serve >> serve.log 2>&1 &`
2. Make sure the Ollama app is open; the system Ollama on port 11434 serves qwen3.6-35b and medgemma-27b.
3. Restart the finisher:
   `cd ~/waymark-local/packaging/causal_orchestration_mhealth && tmux new-session -d -s v2robust "caffeinate -dims zsh scripts/v2_final_robust.sh"`
4. Check progress: `python3 scripts/v2_progress.py`. It is finished when the output shows `done True`.
