# Cyber Range - Blue Team Agent con Normalizacion Semantica y Memoria FAISS

Proyecto de tesis para evaluar la resiliencia de un agente defensivo ante cambios de backend y variaciones de esquema en herramientas de telemetria. El sistema genera episodios reproducibles, ejecuta un Blue Team Agent sobre logs sinteticos, compara escenarios con y sin memoria, mide degradacion pre/post-swap y evalua normalizacion semantica asistida por LLM.

## Objetivo

El objetivo experimental es responder una pregunta concreta:

> Que ocurre con un agente defensivo cuando cambia el backend de telemetria y el esquema de los eventos deja de coincidir con el formato esperado?

Para medirlo, el proyecto compara:

- backends `backend_a` y `backend_b`
- mapeo estatico vs mapeo dinamico
- ejecucion con y sin memoria vectorial FAISS
- ejecucion con y sin capa MCP-like local
- modelos de normalizacion semantica como `qwen3:8b` via Ollama y `gemini-2.5-flash` via API
- metricas de deteccion, contencion, MTTD, MTTR, latencia, uso de cache y cobertura de memoria

## Arquitectura

Flujo principal del Blue Team Agent:

```text
observe -> normalize_schema -> enrich -> retrieve_memory -> correlate -> decide -> act -> log
```

Componentes principales:

- `src/generate_episodes.py`: genera episodios sinteticos reproducibles con ground truth.
- `src/backend_a/search_logs.py`: backend A con esquema base.
- `src/backend_b/search_logs.py`: backend B con drift de esquema.
- `src/mcp/local_client.py`: capa MCP-like local para desacoplar herramientas.
- `src/blue/blue_agent_graph.py`: flujo de decision del Blue Team Agent.
- `src/blue/schema_mapper.py`: normalizador estatico/dinamico y cache de mappings.
- `src/memory/faiss_store.py`: memoria vectorial FAISS.
- `src/eval/run_experiments.py`: runner de experimentos multiseed.
- `src/eval/aggregate_results.py`: agregador de metricas.
- `src/eval/analyze_recurrent_benign.py`: analisis de benignos recurrentes.
- `scripts/run_final_experiments.ps1`: bateria completa A0-A3 y S0-S4.
- `scripts/run_recurrent_benign_memory.ps1`: prueba especifica de memoria operativa.
- `scripts/run_normalizer_diagnostics.ps1`: bateria diagnostica para normalizacion dinamica.

## Que se versiona y que no

El repositorio debe contener codigo, scripts y configuracion reproducible.

No se versionan por defecto:

- `data/`: logs, ground truth generado, runs y experimentos agregados.
- `.venv/`: entorno virtual local.
- caches de Python y herramientas.
- modelos, indices FAISS, vector stores y bases locales.
- archivos `.env`, credenciales y llaves.
- documentos locales de tesis o notas privadas.

Los datos experimentales se regeneran con seeds fijas usando los comandos de este README.

## Requisitos

- Windows PowerShell recomendado para los scripts `.ps1`.
- Python 3.11+.
- Espacio en disco suficiente para modelos de embeddings y dependencias de `sentence-transformers`.
- Para pruebas con Gemini: variable de entorno `GEMINI_API_KEY`.
- Para pruebas con Ollama: Ollama ejecutandose localmente y modelo descargado.

## Instalacion

Desde la raiz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Verificar dependencias:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

## Configuracion de LLM

### Gemini 2.5 Flash
Configura la api key como variable de entorno:

```powershell
$env:GEMINI_API_KEY = "TU_API_KEY"
```

La bateria final usa:

```text
LLMProvider = gemini
GeminiModel = gemini-2.5-flash
```

## Ejecucion rapida

Generar episodios:

```powershell
.\.venv\Scripts\python.exe src/generate_episodes.py `
  --out data `
  --episodes 40 `
  --base-seed 1337 `
  --noise-per-episode 2000 `
  --benign-rate 0.35
```

Validar logs generados:

```powershell
.\.venv\Scripts\python.exe src/validating_phase1.py `
  --logs-dir data/logs_backend_a
```

Ejecutar un episodio con Blue Team Agent usando backend A:

```powershell
.\.venv\Scripts\python.exe -m src.blue.run_blue_agent `
  --episode-id 1 `
  --run-id blue_manual_backend_a `
  --logs-dir data/logs_backend_a `
  --gt-dir data/ground_truth `
  --backend backend_a `
  --mcp-tool search_logs `
  --non-interactive
```

Ejecutar un episodio sin MCP-like:

```powershell
.\.venv\Scripts\python.exe -m src.blue.run_blue_agent `
  --episode-id 1 `
  --run-id blue_manual_no_mcp `
  --logs-dir data/logs_backend_a `
  --gt-dir data/ground_truth `
  --backend backend_a `
  --no-mcp `
  --non-interactive
```

## Bateria completa con Gemini

Este es el comando principal para reproducir la bateria completa comparable con Gemini 2.5 Flash.

Incluye:

- fases `A0-A3` en `backend_a`
- fases `S0-S4` con tool-swap `backend_a -> backend_b`
- 40 episodios por corrida
- 5 repeticiones por fase
- corrida robusta `S3` con 10 repeticiones
- drift `hard4`
- memoria entrenada con separacion train/eval
- cache de mappings

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$env:GEMINI_API_KEY = "TU_API_KEY"

powershell -ExecutionPolicy Bypass -File scripts/run_final_experiments.ps1 `
  -ExperimentTag "tesis_gemini_full_$stamp" `
  -Repetitions 5 `
  -Episodes 40 `
  -BaseSeed 1337 `
  -SeedStep 100 `
  -TrainEvalSeedOffset 10000 `
  -BenignRate 0.35 `
  -NoisePerEpisode 2000 `
  -BackendBDriftProfile hard4 `
  -LLMProvider gemini `
  -GeminiModel gemini-2.5-flash `
  -LLMTimeoutSec 12 `
  -SchemaMapMinConfidence 0.70 `
  -SchemaCacheScope run `
  -IncludeRobustness
```

Salidas esperadas:

```text
data/experiments/phase_a0_backend_a_nomem_<tag>/
data/experiments/phase_a1_backend_a_train_mem_<tag>/
data/experiments/phase_a2_backend_a_with_mem_<tag>/
data/experiments/phase_a3_backend_a_nomem_posttrain_<tag>/
data/experiments/phase_s0_swap_ab_static_nomem_<tag>/
data/experiments/phase_s1_swap_ab_dynamic_llm_nomem_<tag>/
data/experiments/phase_s2_swap_ab_static_mem_<tag>/
data/experiments/phase_s3_swap_ab_dynamic_llm_mem_<tag>/
data/experiments/phase_s4_swap_ab_dynamic_llm_mem_nomcp_<tag>/
data/experiments/phase_s3_swap_ab_dynamic_llm_mem_robust_<tag>/
```


## Evaluacion de memoria para benignos recurrentes

Esta prueba mide si la memoria operativa ayuda a reducir ruido benigno recurrente. Usa semillas separadas para poblar memoria y evaluar con episodios distintos.

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

powershell -ExecutionPolicy Bypass -File scripts/run_recurrent_benign_memory.ps1 `
  -ExperimentTag "recurrent_benign_$stamp" `
  -Repetitions 5 `
  -Episodes 12 `
  -TrainBaseSeed 1337 `
  -EvalBaseSeed 11337 `
  -SeedStep 100 `
  -BenignRate 1.0 `
  -RecurrentBenignRate 0.70 `
  -RecurrentBenignProfiles 3 `
  -NoisePerEpisode 2000
```

Artefactos principales:

```text
recurrent_benign_analysis_nomem_summary.csv
recurrent_benign_analysis_withmem_summary.csv
memory_coverage_summary.csv
summary_confusion.csv
latency_breakdown.csv
```

## Artefactos de evaluacion

Cada experimento agregado produce archivos como:

- `manifest.json`: configuracion y runs incluidos.
- `summary_confusion.csv`: TP, FP, TN, FN, precision, recall, FPR.
- `summary_mttd.csv`: MTTD/MTTR agregados.
- `swap_phase_summary.csv`: comparacion pre-swap vs post-swap.
- `latency_breakdown.csv`: latencia por etapa del pipeline.
- `schema_mapper_usage_summary.csv`: llamadas al LLM, cache hit, fallback y fuente de mapping.
- `schema_fallback_summary.csv`: impacto del fallback.
- `memory_coverage_summary.csv`: cobertura e influencia de memoria.
- `diagnosis_memory.csv`: diagnostico de uso de memoria.
- `tradeoff_fp_fn_action.csv`: acciones finales y trade-offs.
- `summary_report.md`: resumen textual del experimento.

## Metricas principales

Metricas de clasificacion:

- `TP`: episodio malicioso correctamente contenido o detectado.
- `FP`: accion sobre episodio benigno.
- `TN`: benigno correctamente no bloqueado.
- `FN`: malicioso no detectado o no contenido.
- `precision`: proporcion de acciones positivas correctas.
- `recall`: proporcion de casos positivos recuperados.
- `fpr`: tasa de falsos positivos.

Metricas operativas:

- `MTTD`: mean time to detect.
- `MTTR`: mean time to respond/remediate.
- `pipeline_duration_ms`: duracion total del pipeline del agente.
- `normalize_schema_ms`: costo de normalizacion semantica.

Metricas de schema mapping:

- `llm_call_rate`: proporcion de eventos que invocan LLM.
- `cache_hit_rate`: proporcion de eventos resueltos por cache.
- `gemini_source_rate`: proporcion resuelta directamente por Gemini.
- `ollama_source_rate`: proporcion resuelta por Ollama.
- `fallback_source_rate`: proporcion resuelta por fallback.

Metricas de memoria:

- `memory_coverage_rate`: proporcion de episodios con memoria recuperada.
- `memory_override_rate`: proporcion de decisiones afectadas por memoria.
- `memory_suppressed_rate`: benignos recurrentes suprimidos por memoria.
- `memory_fp_consensus_rate`: consenso de memoria sobre falsos positivos previos.

## Escenarios de la bateria final

| Escenario | Proposito |
|---|---|
| `A0` | Linea base en `backend_a` sin memoria |
| `A1` | Entrenamiento/poblado de memoria |
| `A2` | Evaluacion en `backend_a` con memoria precargada |
| `A3` | Control posterior sin memoria |
| `S0` | Tool-swap con mapper estatico y sin memoria |
| `S1` | Tool-swap con mapper dinamico y sin memoria |
| `S2` | Tool-swap estatico con memoria |
| `S3` | Tool-swap dinamico con memoria |
| `S4` | Ablacion sin MCP-like |
| `S3 robusta` | Repeticion robusta del escenario principal con 10 repeticiones |

## Interpretacion esperada de resultados

La bateria inicial con Qwen/Ollama permitio observar una degradacion fuerte en normalizacion dinamica bajo ciertas condiciones locales: el agente podia detectar, pero no siempre contener si faltaban campos criticos como `src_ip` o tiempo.

La bateria completa con Gemini 2.5 Flash permite evaluar si esa degradacion era inherente al enfoque o dependia del modelo. En los resultados finales del proyecto, Gemini sostuvo la contencion post-swap bajo drift fuerte y el cache redujo las llamadas al LLM en corridas posteriores.

La prueba de benignos recurrentes evalua otro eje: si la memoria FAISS puede conservar conocimiento operativo para evitar escalaciones o bloqueos repetidos sobre actividad ya validada como benigna.

## Reproducibilidad

Para reproducir los experimentos:

1. Clona el repositorio.
2. Crea el entorno virtual.
3. Instala `requirements-dev.txt`.
4. Configura `GEMINI_API_KEY` si usaras Gemini.
5. Ejecuta `scripts/run_final_experiments.ps1` con los parametros indicados.
6. Revisa los artefactos agregados en `data/experiments/<experiment_id>/`.

Los datasets no se versionan porque se generan deterministicamente a partir de:

- `BaseSeed`
- `SeedStep`
- `TrainEvalSeedOffset`
- numero de episodios
- tasa de benignos
- drift profile



## Solucion de problemas

Si falla FAISS:

```powershell
python -m pip install faiss-cpu==1.13.2
```

Si falla `sentence-transformers`, verifica que `torch`, `transformers` y `huggingface-hub` esten instalados:

```powershell
python -m pip show torch transformers sentence-transformers
```

Si Gemini devuelve error de autenticacion:

```powershell
if ($env:GEMINI_API_KEY) { "GEMINI_API_KEY configurada" } else { "GEMINI_API_KEY no configurada" }
```


Si PowerShell bloquea scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_final_experiments.ps1
```

## Estado del proyecto

El repositorio contiene la plataforma experimental:

- generacion reproducible de episodios
- tool-swap entre backends
- normalizacion semantica dinamica
- memoria FAISS
- gating de acciones defensivas
- evaluacion por batch
- agregacion de metricas
- analisis de memoria para benignos recurrentes

Los resultados finales no se versionan por tamano y reproducibilidad, pero pueden regenerarse con los comandos anteriores.
