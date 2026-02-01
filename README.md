# thesis-cyberrange (fase 1)

Generación reproducible de episodios:
- Red Team determinista (runner)
- Telemetría estructurada (logs JSONL)
- Ground truth por episodio

## Run
python src/generate_episodes.py --out data --episodes 10 --base-seed 1337 --noise-per-episode 2000
