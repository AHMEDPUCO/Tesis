# Cyber Range - Blue Agent + FAISS Memory

Proyecto de tesis para evaluar un Blue Agent sobre telemetria sintetica, comparando:
- `baseline` (sin memoria)
- `blue` con memoria vectorial FAISS

Incluye generacion reproducible de episodios, evaluacion con ground truth y agregacion de resultados experimentales.

## Componentes implementados
- `Red Team / Generador`: `src/generate_episodes.py`
- `Validador Fase 1`: `src/validating_phase1.py`
- `Baseline`: `src/run_phase_2_baseline.py`
- `Blue Agent`: `src/blue/blue_agent_graph.py`
- `MCP local (tool-swap search_logs)`: `src/mcp/local_client.py`
- `Memoria FAISS`: `src/memory/faiss_store.py`
- `Judge confusion (TP/FP/TN/FN)`: `src/judge/judge_confusion.py`
- `Judge tiempos (MTTD/MTTR)`: `src/judge/judge_mtd_mttr.py`
- `Runner de experimentos`: `src/eval/run_experiments.py`
- `Agregador de resultados`: `src/eval/aggregate_results.py`

## Flujo del Blue Agent
`observe -> normalize_schema -> enrich -> retrieve_memory -> correlate -> decide -> act -> log`

`search_logs` se ejecuta via capa MCP local (por defecto) con swap de backend
(`backend_a` / `backend_b`) bajo un contrato de tool explicito.

## Estructura de datos relevante
- Logs: `data/logs_backend_a/episode_XXX.jsonl`
- Ground truth: `data/ground_truth/episode_XXX.json`
- Decisiones por run: `data/runs/<run_id>/decisions.jsonl`
- Acciones por run: `data/runs/<run_id>/enforcement_actions.jsonl`
- Resultados por run: `data/runs/<run_id>/results_*.csv`
- Experimentos agregados: `data/experiments/<experiment_id>/`

## Requisitos
Se recomienda Python 3.11+ y entorno virtual.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nota: el proyecto usa librerias adicionales para memoria/embeddings (FAISS, sentence-transformers, etc.).
Si tu entorno no las tiene, instalalas segun el error de import.

## Ejecucion rapida (end-to-end)
1. Generar episodios:

```powershell
python src/generate_episodes.py --out data --episodes 40 --base-seed 1337 --noise-per-episode 2000 --benign-rate 0.35
```

2. Validar telemetria:

```powershell
python src/validating_phase1.py --logs-dir data/logs_backend_a
```

3. Correr baseline:

```powershell
python src/run_phase_2_baseline.py --logs-dir data/logs_backend_a --gt-dir data/ground_truth --decisions-path data/decisions/decisions.jsonl --actions-path data/actions/enforcement_actions.jsonl --out-csv data/results/results_phase2.csv
```

4. Correr Blue Agent (ejemplo episodio unico):

```powershell
python -m src.blue.run_blue_agent --episode-id 1 --run-id blue_manual --logs-dir data/logs_backend_a --gt-dir data/ground_truth --non-interactive
```

Ejemplo con MCP explicito:

```powershell
python -m src.blue.run_blue_agent --episode-id 1 --run-id blue_mcp --logs-dir data/logs_backend_a --backend backend_a --mcp-tool search_logs --non-interactive
```

Para desactivar MCP y usar acceso directo legacy:

```powershell
python -m src.blue.run_blue_agent --episode-id 1 --run-id blue_no_mcp --logs-dir data/logs_backend_a --backend backend_a --no-mcp --non-interactive
```

## Bateria de experimentos (recomendado para reporte)
1. Ejecutar experimentos multiseed:

```powershell
$py = ".\\.venv\\Scripts\\python.exe"
& $py -m src.eval.run_experiments --experiment-id entregable1_v3 --repetitions 3 --episodes 40 --base-seed 1337 --seed-step 100 --benign-rate 0.35 --noise-per-episode 2000 --memory-seed-dir data/memory
```

2. Agregar resultados:

```powershell
$py = ".\\.venv\\Scripts\\python.exe"
& $py -m src.eval.aggregate_results --manifest data/experiments/entregable1_v3/manifest.json
```

3. Revisar salidas agregadas:
- `data/experiments/entregable1_v3/summary_confusion.csv`
- `data/experiments/entregable1_v3/summary_mttd.csv`
- `data/experiments/entregable1_v3/diagnosis_memory.csv`
- `data/experiments/entregable1_v3/summary_report.md`

## Comandos Git para ir subiendo al repositorio
Desde la raiz del proyecto:

```powershell
git status
git add README.md
git commit -m "docs: actualizar README con flujo y comandos de experimentacion"
git push origin <tu-rama>
```

Si aun no creaste rama de trabajo:

```powershell
git checkout -b docs/readme-update
git push -u origin docs/readme-update
```

Para subir todos los cambios actuales (no solo README):

```powershell
git status
git add .
git commit -m "chore: actualizar docs y scripts de experimentacion"
git push origin <tu-rama>
```
