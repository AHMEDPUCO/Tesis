param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$OutDir = "data/experiments",
    [string]$ExperimentTag = "",
    [int]$Repetitions = 5,
    [int]$Episodes = 40,
    [int]$BaseSeed = 1337,
    [int]$SeedStep = 100,
    [int]$TrainEvalSeedOffset = 10000,
    [double]$BenignRate = 0.35,
    [int]$NoisePerEpisode = 2000,
    [string]$MemorySeedDir = "data/memory",
    [ValidateSet("classic", "hard4")]
    [string]$BackendBDriftProfile = "classic",
    [ValidateSet("gemini", "ollama")]
    [string]$LLMProvider = "ollama",
    [string]$GeminiModel = "gemini-2.5-flash",
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [string]$OllamaModel = "qwen3:8b",
    [double]$LLMTimeoutSec = 12.0,
    [double]$SchemaMapMinConfidence = 0.70,
    [ValidateSet("run", "persistent")]
    [string]$SchemaCacheScope = "run",
    [ValidateSet("contract_first", "llm_first")]
    [string]$SchemaAdaptMode = "contract_first",
    [ValidateSet("full", "minimal")]
    [string]$BackendBAliasMode = "full",
    [switch]$IncludeRobustness
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RunsRoot {
    if ($env:CYBER_RANGE_RUNS_DIR -and -not [string]::IsNullOrWhiteSpace($env:CYBER_RANGE_RUNS_DIR)) {
        return $env:CYBER_RANGE_RUNS_DIR
    }

    foreach ($candidate in @("data/runs", "data/experiments/_runs")) {
        try {
            New-Item -ItemType Directory -Path $candidate -Force -ErrorAction Stop | Out-Null
            $probe = Join-Path $candidate ".write_probe"
            Set-Content -Path $probe -Value "ok" -Encoding UTF8 -ErrorAction Stop
            Remove-Item -LiteralPath $probe -Force -ErrorAction Stop
            $env:CYBER_RANGE_RUNS_DIR = $candidate
            return $candidate
        } catch {
            continue
        }
    }

    throw "No se encontro un directorio escribible para runs."
}

function Get-ExperimentId {
    param(
        [string]$BaseId
    )

    if ([string]::IsNullOrWhiteSpace($ExperimentTag)) {
        return $BaseId
    }

    $cleanTag = $ExperimentTag.Trim()
    return "${BaseId}_$cleanTag"
}

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$CommandArgs
    )

    Write-Host ""
    Write-Host ("=" * 80)
    Write-Host $Title
    Write-Host ("=" * 80)
    Write-Host ("RUN: " + ($CommandArgs -join " "))
    & $PythonExe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo la ejecucion: $Title"
    }
}

function Invoke-Experiment {
    param(
        [string]$ExperimentId,
        [string[]]$ExtraArgs,
        [int]$RepetitionsOverride = $Repetitions,
        [int]$BaseSeedOverride = $BaseSeed,
        [string]$MemorySeedDirOverride = $MemorySeedDir,
        [ValidateSet("classic", "hard4")]
        [string]$BackendBDriftProfileOverride = $BackendBDriftProfile,
        [ValidateSet("contract_first", "llm_first")]
        [string]$SchemaAdaptModeOverride = $SchemaAdaptMode,
        [ValidateSet("full", "minimal")]
        [string]$BackendBAliasModeOverride = $BackendBAliasMode
    )

    $runArgs = @(
        "-m", "src.eval.run_experiments",
        "--experiment-id", $ExperimentId,
        "--repetitions", $RepetitionsOverride,
        "--episodes", $Episodes,
        "--base-seed", $BaseSeedOverride,
        "--seed-step", $SeedStep,
        "--benign-rate", $BenignRate,
        "--noise-per-episode", $NoisePerEpisode,
        "--backend-b-drift-profile", $BackendBDriftProfileOverride,
        "--memory-seed-dir", $MemorySeedDirOverride,
        "--out-dir", $OutDir,
        "--schema-cache-scope", $SchemaCacheScope,
        "--schema-adapt-mode", $SchemaAdaptModeOverride,
        "--backend-b-alias-mode", $BackendBAliasModeOverride
    ) + $ExtraArgs

    Invoke-Step -Title "Experimento: $ExperimentId" -CommandArgs $runArgs

    $manifest = Join-Path $OutDir "$ExperimentId\manifest.json"
    Invoke-Step -Title "Agregacion: $ExperimentId" -CommandArgs @(
        "-m", "src.eval.aggregate_results",
        "--manifest", $manifest
    )
}

function Get-BlueMemoryDir {
    param(
        [string]$ExperimentId,
        [int]$Repetition = 1
    )

    $repName = "rep_{0:D2}" -f $Repetition
    $runId = "blue_mem_{0}_{1}" -f $ExperimentId, $repName
    return Join-Path (Join-Path $script:RunsRoot $runId) "memory"
}

function New-MemorySeedBundle {
    param(
        [string]$SourceExperimentId,
        [int]$RepetitionCount = $Repetitions
    )

    $bundleDir = Join-Path $OutDir ("_memory_seed_bundle_{0}_{1}" -f $SourceExperimentId, [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss"))
    New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null

    for ($rep = 1; $rep -le $RepetitionCount; $rep++) {
        $repName = "rep_{0:D2}" -f $rep
        $srcDir = Get-BlueMemoryDir -ExperimentId $SourceExperimentId -Repetition $rep
        if (-not (Test-Path $srcDir)) {
            throw "No se encontro memoria entrenada para $SourceExperimentId/$repName en $srcDir"
        }
        $dstDir = Join-Path $bundleDir $repName
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
        foreach ($name in @("cases.jsonl", "index.faiss")) {
            $srcFile = Join-Path $srcDir $name
            if (Test-Path $srcFile) {
                Copy-Item -LiteralPath $srcFile -Destination (Join-Path $dstDir $name) -Force
            }
        }
    }

    return $bundleDir
}

function Test-OllamaReady {
    param(
        [string]$Url
    )

    try {
        $null = Invoke-WebRequest -Uri "$Url/api/tags" -UseBasicParsing -TimeoutSec 5
    } catch {
        throw "Ollama no responde en $Url. Inicia 'ollama serve' antes de correr las pruebas dinamicas."
    }
}

function Invoke-OllamaPrewarm {
    param(
        [string]$Url,
        [string]$Model
    )

    $body = @{
        model = $Model
        prompt = "Respond with exactly: ok"
        stream = $false
        options = @{ temperature = 0 }
    } | ConvertTo-Json -Depth 4

    try {
        $resp = Invoke-RestMethod -Method Post -Uri "$Url/api/generate" -ContentType "application/json" -Body $body -TimeoutSec 90
        Write-Host "Ollama prewarm: ok ($Model)"
        return $true
    } catch {
        Write-Warning "Ollama prewarm fallo: $($_.Exception.Message)"
        return $false
    }
}

function Get-LLMArgs {
    return @(
        "--llm-provider", $LLMProvider,
        "--gemini-model", $GeminiModel,
        "--ollama-url", $OllamaUrl,
        "--ollama-model", $OllamaModel,
        "--llm-timeout-sec", "$LLMTimeoutSec"
    )
}

if (-not (Test-Path $PythonExe)) {
    throw "No se encontro Python en $PythonExe"
}

$script:RunsRoot = Get-RunsRoot
Write-Host "Runs root: $script:RunsRoot"

Write-Host "Python: $PythonExe"
Write-Host "Salida experimentos: $OutDir"
Write-Host "Repeticiones: $Repetitions | Episodios: $Episodes"
Write-Host "BaseSeed entrenamiento: $BaseSeed | Offset train/eval: $TrainEvalSeedOffset"
if (-not [string]::IsNullOrWhiteSpace($ExperimentTag)) {
    Write-Host "Experiment tag: $ExperimentTag"
}
Write-Host "Bateria por fases: A0-A3 (backend A/memoria) y S0-S4 (tool-swap A->B)."
Write-Host "LLM se usa solo en pruebas dinamicas (S1, S3, S4)."
Write-Host "LLM provider: $LLMProvider | Gemini model: $GeminiModel | Ollama model: $OllamaModel"

if ($Episodes -lt 2) {
    throw "Para pruebas de swap se requiere --Episodes >= 2."
}
$SwapEpisode = [math]::Floor($Episodes / 2)
if ($SwapEpisode -lt 1) { $SwapEpisode = 1 }
if ($SwapEpisode -ge $Episodes) { $SwapEpisode = $Episodes - 1 }
Write-Host "Swap episode definido en: $SwapEpisode"

# Seed de memoria vacia para escenarios "sin memoria inicial".
$emptySeedDir = Join-Path $OutDir ("_memory_seed_empty_{0}" -f [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $emptySeedDir -Force | Out-Null
Write-Host "Memory seed vacio: $emptySeedDir"

$backendATrainSeed = $BaseSeed
$backendAEvalSeed = $BaseSeed + $TrainEvalSeedOffset
$swapEvalSeed = $BaseSeed + (2 * $TrainEvalSeedOffset)
Write-Host "Seed backend A entrenamiento: $backendATrainSeed"
Write-Host "Seed backend A evaluacion: $backendAEvalSeed"
Write-Host "Seed escenarios swap: $swapEvalSeed"

# ----------------------------
# Fase A: Backend A
# ----------------------------

# A0: Control sin memoria inicial.
$expA0 = Get-ExperimentId "phase_a0_backend_a_nomem"
Invoke-Experiment -ExperimentId $expA0 -ExtraArgs @(
    "--blue-backend", "backend_a",
    "--blue-schema-mapper", "static",
    "--mcp-enabled"
) -BaseSeedOverride $backendAEvalSeed -MemorySeedDirOverride $emptySeedDir -BackendBDriftProfileOverride "classic" -SchemaAdaptModeOverride "contract_first" -BackendBAliasModeOverride "full"

# A1: Entrenamiento de memoria (genera casos para FAISS).
$expA1 = Get-ExperimentId "phase_a1_backend_a_train_mem"
Invoke-Experiment -ExperimentId $expA1 -ExtraArgs @(
    "--blue-backend", "backend_a",
    "--blue-schema-mapper", "static",
    "--mcp-enabled"
) -BaseSeedOverride $backendATrainSeed -MemorySeedDirOverride $emptySeedDir -BackendBDriftProfileOverride "classic" -SchemaAdaptModeOverride "contract_first" -BackendBAliasModeOverride "full"

$trainedMemoryBundleDir = New-MemorySeedBundle -SourceExperimentId $expA1 -RepetitionCount $Repetitions
Write-Host "Memoria entrenada por repeticion detectada en: $trainedMemoryBundleDir"

# A2: Evaluacion con memoria precargada.
$expA2 = Get-ExperimentId "phase_a2_backend_a_with_mem"
Invoke-Experiment -ExperimentId $expA2 -ExtraArgs @(
    "--blue-backend", "backend_a",
    "--blue-schema-mapper", "static",
    "--mcp-enabled"
) -BaseSeedOverride $backendAEvalSeed -MemorySeedDirOverride $trainedMemoryBundleDir -BackendBDriftProfileOverride "classic" -SchemaAdaptModeOverride "contract_first" -BackendBAliasModeOverride "full"

# A3: Memoria desactivada otra vez (sin seed) tras entrenamiento.
$expA3 = Get-ExperimentId "phase_a3_backend_a_nomem_posttrain"
Invoke-Experiment -ExperimentId $expA3 -ExtraArgs @(
    "--blue-backend", "backend_a",
    "--blue-schema-mapper", "static",
    "--mcp-enabled"
) -BaseSeedOverride $backendAEvalSeed -MemorySeedDirOverride $emptySeedDir -BackendBDriftProfileOverride "classic" -SchemaAdaptModeOverride "contract_first" -BackendBAliasModeOverride "full"

# ----------------------------
# Fase S: Tool-swap A -> B
# ----------------------------

if ($LLMProvider -eq "ollama") {
    Test-OllamaReady -Url $OllamaUrl
    $null = Invoke-OllamaPrewarm -Url $OllamaUrl -Model $OllamaModel
} elseif ($LLMProvider -eq "gemini") {
    if (-not $env:GEMINI_API_KEY -or [string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) {
        throw "GEMINI_API_KEY no esta configurada. Define `$env:GEMINI_API_KEY antes de correr con -LLMProvider gemini."
    }
    Write-Host "Gemini API key detectada en variable de entorno."
}
$llmArgs = Get-LLMArgs

# S0: Swap estatico sin LLM ni memoria.
$expS0 = Get-ExperimentId "phase_s0_swap_ab_static_nomem"
Invoke-Experiment -ExperimentId $expS0 -ExtraArgs @(
    "--blue-schema-mapper", "static",
    "--mcp-enabled",
    "--swap-episode", "$SwapEpisode",
    "--swap-phase1-backend", "backend_a",
    "--swap-phase2-backend", "backend_b"
 ) -BaseSeedOverride $swapEvalSeed -MemorySeedDirOverride $emptySeedDir -BackendBDriftProfileOverride "hard4" -SchemaAdaptModeOverride "contract_first" -BackendBAliasModeOverride "full"

# S1: Swap dinamico con LLM, sin memoria.
$expS1 = Get-ExperimentId "phase_s1_swap_ab_dynamic_llm_nomem"
$extraS1 = @(
    "--blue-schema-mapper", "dynamic",
    "--schema-map-min-confidence", $SchemaMapMinConfidence,
    "--mcp-enabled",
    "--swap-episode", "$SwapEpisode",
    "--swap-phase1-backend", "backend_a",
    "--swap-phase2-backend", "backend_b"
 ) + $llmArgs
Invoke-Experiment -ExperimentId $expS1 -ExtraArgs $extraS1 -BaseSeedOverride $swapEvalSeed -MemorySeedDirOverride $emptySeedDir -BackendBDriftProfileOverride "hard4" -SchemaAdaptModeOverride "llm_first" -BackendBAliasModeOverride "minimal"

# S2: Swap estatico con memoria.
$expS2 = Get-ExperimentId "phase_s2_swap_ab_static_mem"
Invoke-Experiment -ExperimentId $expS2 -ExtraArgs @(
    "--blue-schema-mapper", "static",
    "--mcp-enabled",
    "--swap-episode", "$SwapEpisode",
    "--swap-phase1-backend", "backend_a",
    "--swap-phase2-backend", "backend_b"
 ) -BaseSeedOverride $swapEvalSeed -MemorySeedDirOverride $trainedMemoryBundleDir -BackendBDriftProfileOverride "hard4" -SchemaAdaptModeOverride "contract_first" -BackendBAliasModeOverride "full"

# S3: Swap dinamico + LLM + memoria (objetivo principal).
$expS3 = Get-ExperimentId "phase_s3_swap_ab_dynamic_llm_mem"
$extraS3 = @(
    "--blue-schema-mapper", "dynamic",
    "--schema-map-min-confidence", $SchemaMapMinConfidence,
    "--mcp-enabled",
    "--swap-episode", "$SwapEpisode",
    "--swap-phase1-backend", "backend_a",
    "--swap-phase2-backend", "backend_b"
 ) + $llmArgs
Invoke-Experiment -ExperimentId $expS3 -ExtraArgs $extraS3 -BaseSeedOverride $swapEvalSeed -MemorySeedDirOverride $trainedMemoryBundleDir -BackendBDriftProfileOverride "hard4" -SchemaAdaptModeOverride "llm_first" -BackendBAliasModeOverride "minimal"

# S4: Ablacion MCP (mide valor de la capa MCP en swap duro).
$expS4 = Get-ExperimentId "phase_s4_swap_ab_dynamic_llm_mem_nomcp"
$extraS4 = @(
    "--blue-schema-mapper", "dynamic",
    "--schema-map-min-confidence", $SchemaMapMinConfidence,
    "--no-mcp",
    "--swap-episode", "$SwapEpisode",
    "--swap-phase1-backend", "backend_a",
    "--swap-phase2-backend", "backend_b"
 ) + $llmArgs
Invoke-Experiment -ExperimentId $expS4 -ExtraArgs $extraS4 -BaseSeedOverride $swapEvalSeed -MemorySeedDirOverride $trainedMemoryBundleDir -BackendBDriftProfileOverride "hard4" -SchemaAdaptModeOverride "llm_first" -BackendBAliasModeOverride "minimal"

# Robustez opcional enfocada en el escenario principal S3.
if ($IncludeRobustness) {
    $expS3Robust = Get-ExperimentId "phase_s3_swap_ab_dynamic_llm_mem_robust"
    $extraS3Robust = @(
        "--blue-schema-mapper", "dynamic",
        "--schema-map-min-confidence", $SchemaMapMinConfidence,
        "--mcp-enabled",
        "--swap-episode", "$SwapEpisode",
        "--swap-phase1-backend", "backend_a",
        "--swap-phase2-backend", "backend_b"
    ) + $llmArgs
    Invoke-Experiment -ExperimentId $expS3Robust -ExtraArgs $extraS3Robust -RepetitionsOverride 10 -BaseSeedOverride $swapEvalSeed -MemorySeedDirOverride $trainedMemoryBundleDir -BackendBDriftProfileOverride "hard4" -SchemaAdaptModeOverride "llm_first" -BackendBAliasModeOverride "minimal"
}

Write-Host ""
Write-Host "Bateria final completada."
Write-Host "Revisa los archivos summary_*.csv y summary_report.md dentro de cada carpeta en $OutDir."
