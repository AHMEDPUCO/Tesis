param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$OutDir = "data/experiments",
    [string]$ExperimentTag = "",
    [int]$Repetitions = 2,
    [int]$Episodes = 12,
    [int]$TrainBaseSeed = 1337,
    [int]$EvalBaseSeed = 11337,
    [int]$SeedStep = 100,
    [double]$BenignRate = 1.0,
    [double]$RecurrentBenignRate = 0.70,
    [int]$RecurrentBenignProfiles = 3,
    [int]$NoisePerEpisode = 2000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ExperimentId {
    param([string]$BaseId)
    if ([string]::IsNullOrWhiteSpace($ExperimentTag)) { return $BaseId }
    return "${BaseId}_$($ExperimentTag.Trim())"
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
        [int]$BaseSeed,
        [string]$MemorySeedDir
    )
    Invoke-Step -Title "Experimento: $ExperimentId" -CommandArgs @(
        "-m", "src.eval.run_experiments",
        "--experiment-id", $ExperimentId,
        "--repetitions", $Repetitions,
        "--episodes", $Episodes,
        "--base-seed", $BaseSeed,
        "--seed-step", $SeedStep,
        "--benign-rate", $BenignRate,
        "--recurrent-benign-rate", $RecurrentBenignRate,
        "--recurrent-benign-profiles", $RecurrentBenignProfiles,
        "--noise-per-episode", $NoisePerEpisode,
        "--out-dir", $OutDir,
        "--memory-seed-dir", $MemorySeedDir,
        "--blue-backend", "backend_a",
        "--blue-schema-mapper", "static",
        "--mcp-enabled"
    )
    Invoke-Step -Title "Agregacion: $ExperimentId" -CommandArgs @(
        "-m", "src.eval.aggregate_results",
        "--manifest", (Join-Path $OutDir "$ExperimentId\manifest.json")
    )
}

function New-MemorySeedBundle {
    param(
        [string]$SourceExperimentId
    )
    $bundleDir = Join-Path $OutDir ("_memory_seed_bundle_{0}_{1}" -f $SourceExperimentId, [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss"))
    New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null
    for ($rep = 1; $rep -le $Repetitions; $rep++) {
        $repName = "rep_{0:D2}" -f $rep
        $srcDir = Join-Path "data/experiments/_runs" ("blue_mem_{0}_{1}\memory" -f $SourceExperimentId, $repName)
        if (-not (Test-Path $srcDir)) {
            $srcDir = Join-Path "data/runs" ("blue_mem_{0}_{1}\memory" -f $SourceExperimentId, $repName)
        }
        if (-not (Test-Path $srcDir)) {
            throw "No se encontro memoria entrenada para $SourceExperimentId/$repName"
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

function Invoke-RecurrentAnalysis {
    param(
        [string]$ExperimentId,
        [string]$OutName
    )
    Invoke-Step -Title "Analisis recurrente: $ExperimentId" -CommandArgs @(
        "-m", "src.eval.analyze_recurrent_benign",
        "--manifest", (Join-Path $OutDir "$ExperimentId\manifest.json"),
        "--out-csv", (Join-Path $OutDir "$ExperimentId\$OutName.csv"),
        "--out-summary-csv", (Join-Path $OutDir "$ExperimentId\${OutName}_summary.csv")
    )
}

if (-not (Test-Path $PythonExe)) {
    throw "No se encontro Python en $PythonExe"
}

$emptySeedDir = Join-Path $OutDir ("_memory_seed_empty_recurrent_{0}" -f [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $emptySeedDir -Force | Out-Null

$trainId = Get-ExperimentId "recurrent_benign_train_mem"
$evalNoMemId = Get-ExperimentId "recurrent_benign_eval_nomem"
$evalWithMemId = Get-ExperimentId "recurrent_benign_eval_withmem"

Invoke-Experiment -ExperimentId $trainId -BaseSeed $TrainBaseSeed -MemorySeedDir $emptySeedDir
$memoryBundle = New-MemorySeedBundle -SourceExperimentId $trainId
Write-Host "Memory bundle: $memoryBundle"

Invoke-Experiment -ExperimentId $evalNoMemId -BaseSeed $EvalBaseSeed -MemorySeedDir $emptySeedDir
Invoke-RecurrentAnalysis -ExperimentId $evalNoMemId -OutName "recurrent_benign_analysis_nomem"

Invoke-Experiment -ExperimentId $evalWithMemId -BaseSeed $EvalBaseSeed -MemorySeedDir $memoryBundle
Invoke-RecurrentAnalysis -ExperimentId $evalWithMemId -OutName "recurrent_benign_analysis_withmem"

Write-Host ""
Write-Host "Listo."
Write-Host "Revisa:"
Write-Host (Join-Path $OutDir "$evalNoMemId\recurrent_benign_analysis_nomem_summary.csv")
Write-Host (Join-Path $OutDir "$evalWithMemId\recurrent_benign_analysis_withmem_summary.csv")
