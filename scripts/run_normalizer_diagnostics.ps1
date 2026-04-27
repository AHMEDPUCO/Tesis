param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$OutDir = "data/experiments",
    [string]$BatchId = "",
    [int]$Repetitions = 5,
    [int]$Episodes = 40,
    [int]$BaseSeed = 1337,
    [int]$SeedStep = 100,
    [int]$TrainEvalSeedOffset = 10000,
    [double]$BenignRate = 0.35,
    [int]$NoisePerEpisode = 2000,
    [int]$Delay = 30,
    [double]$SchemaMapMinConfidence = 0.70,
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [string]$OllamaModel = "qwen3:8b",
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

function Invoke-Experiment {
    param(
        [string]$ExperimentId,
        [string]$MemorySeedDir,
        [string]$BlueSchemaMapper,
        [bool]$McpEnabled,
        [bool]$SwapEnabled,
        [string]$SchemaAdaptMode,
        [string]$BackendBAliasMode,
        [string]$BackendBDriftProfile,
        [string]$LlmProvider = "gemini",
        [int]$RepetitionsOverride = $Repetitions,
        [int]$BaseSeedOverride = $BaseSeed
    )

    $args = @(
        "-m", "src.eval.run_experiments",
        "--experiment-id", $ExperimentId,
        "--repetitions", $RepetitionsOverride,
        "--episodes", $Episodes,
        "--base-seed", $BaseSeedOverride,
        "--seed-step", $SeedStep,
        "--benign-rate", $BenignRate,
        "--noise-per-episode", $NoisePerEpisode,
        "--delay", $Delay,
        "--memory-seed-dir", $MemorySeedDir,
        "--out-dir", $OutDir,
        "--blue-schema-mapper", $BlueSchemaMapper,
        "--schema-map-min-confidence", $SchemaMapMinConfidence,
        "--schema-cache-scope", "run",
        "--schema-adapt-mode", $SchemaAdaptMode,
        "--backend-b-alias-mode", $BackendBAliasMode,
        "--backend-b-drift-profile", $BackendBDriftProfile,
        "--llm-provider", $LlmProvider,
        "--ollama-url", $OllamaUrl,
        "--ollama-model", $OllamaModel
    )

    if ($McpEnabled) {
        $args += "--mcp-enabled"
    } else {
        $args += "--no-mcp"
    }

    if ($SwapEnabled) {
        $swapEpisode = [math]::Floor($Episodes / 2)
        if ($swapEpisode -lt 1) { $swapEpisode = 1 }
        if ($swapEpisode -ge $Episodes) { $swapEpisode = $Episodes - 1 }
        $args += @(
            "--swap-episode", "$swapEpisode",
            "--swap-phase1-backend", "backend_a",
            "--swap-phase2-backend", "backend_b"
        )
    } else {
        $args += @("--blue-backend", "backend_a")
    }

    Invoke-Step -Title "Experimento: $ExperimentId" -CommandArgs $args

    $manifest = Join-Path $OutDir "$ExperimentId\manifest.json"
    Invoke-Step -Title "Agregacion: $ExperimentId" -CommandArgs @(
        "-m", "src.eval.aggregate_results",
        "--manifest", $manifest
    )
}

function Get-ExperimentSummaryRow {
    param(
        [string]$ExperimentId,
        [string]$Label
    )

    $base = Join-Path $OutDir $ExperimentId
    $conf = Import-Csv (Join-Path $base "summary_confusion.csv")
    $mttd = Import-Csv (Join-Path $base "summary_mttd.csv")
    $lat = Import-Csv (Join-Path $base "latency_breakdown.csv")
    $swap = Import-Csv (Join-Path $base "swap_phase_summary.csv")
    $schemaUse = Import-Csv (Join-Path $base "schema_mapper_usage_summary.csv")
    $schemaFb = Import-Csv (Join-Path $base "schema_fallback_summary.csv")
    $mem = Import-Csv (Join-Path $base "memory_coverage_summary.csv")

    $blueCont = $conf | Where-Object { $_.system -eq "blue" -and $_.mode -eq "containment" }
    $blueDet = $conf | Where-Object { $_.system -eq "blue" -and $_.mode -eq "detection" }
    $baseCont = $conf | Where-Object { $_.system -eq "baseline" -and $_.mode -eq "containment" }
    $blueMttd = $mttd | Where-Object { $_.system -eq "blue" }
    $baseMttd = $mttd | Where-Object { $_.system -eq "baseline" }
    $blueLat = $lat | Where-Object { $_.system -eq "blue" }
    $swapRow = $swap | Select-Object -First 1
    $schemaUseRow = $schemaUse | Select-Object -First 1
    $schemaFbRow = $schemaFb | Select-Object -First 1
    $memRow = $mem | Select-Object -First 1

    return [pscustomobject]@{
        label = $Label
        experiment_id = $ExperimentId
        n_runs = $blueCont.n_runs
        recall_cont_blue = $blueCont.recall_mean
        recall_cont_baseline = $baseCont.recall_mean
        recall_det_blue = $blueDet.recall_mean
        fp_blue = $blueCont.FP_mean
        fn_blue = $blueCont.FN_mean
        mttd_blue = $blueMttd.mttd_mean
        mttd_baseline = $baseMttd.mttd_mean
        pipeline_ms_blue = $blueLat.pipeline_duration_ms_mean
        normalize_ms_blue = $blueLat.normalize_schema_ms_mean
        phase1_recall_cont = $swapRow.phase1_recall_containment_mean
        phase2_recall_cont = $swapRow.phase2_recall_containment_mean
        delta_recall_cont = $swapRow.delta_recall_containment_mean
        phase1_mttd = $swapRow.phase1_mttd_mean
        phase2_mttd = $swapRow.phase2_mttd_mean
        delta_mttd = $swapRow.delta_mttd_mean
        llm_call_rate = $schemaUseRow.llm_call_rate_mean
        ollama_source_rate = $schemaUseRow.ollama_source_rate_mean
        fallback_source_rate = $schemaUseRow.fallback_source_rate_mean
        fallback_rate = $schemaFbRow.fallback_rate_mean
        non_fallback_recall = $schemaFbRow.non_fallback_recall_mean
        fallback_recall = $schemaFbRow.fallback_recall_mean
        memory_coverage_rate = $memRow.memory_coverage_rate_mean
        memory_override_rate = $memRow.memory_override_rate_mean
    }
}

function Export-QualitativeFailures {
    param(
        [string]$ExperimentId,
        [string]$OutPath
    )

    $runId = "blue_mem_{0}_rep_01" -f $ExperimentId
    $decisionsPath = Join-Path (Join-Path (Join-Path $script:RunsRoot $runId) "") "decisions.jsonl"
    if (-not (Test-Path $decisionsPath)) {
        Write-Warning "No se encontro decisions.jsonl para $ExperimentId"
        return
    }

    function Get-OptionalPropertyValue {
        param(
            $Object,
            [string]$Name
        )

        if ($null -eq $Object) {
            return $null
        }

        $prop = $Object.PSObject.Properties[$Name]
        if ($null -eq $prop) {
            return $null
        }

        return $prop.Value
    }

    $swapEpisode = [math]::Floor($Episodes / 2)
    $rows = @()
    Get-Content $decisionsPath | ForEach-Object {
        if ([string]::IsNullOrWhiteSpace($_)) { return }
        $obj = $_ | ConvertFrom-Json
        if ([int]$obj.episode_id -lt ($swapEpisode + 1)) { return }

        $ev = $obj.evidence
        $mapping = $ev.schema_mapping
        $det = $ev.detection_event
        $gating = $ev.gating
        $timing = $ev.timing.stages.normalize_schema

        $detSrcIp = Get-OptionalPropertyValue -Object $det -Name "src_ip"
        $detHost = Get-OptionalPropertyValue -Object $det -Name "host"
        $detAction = Get-OptionalPropertyValue -Object $det -Name "action"
        $detOutcome = Get-OptionalPropertyValue -Object $det -Name "outcome"
        $detSeverity = Get-OptionalPropertyValue -Object $det -Name "severity"

        $missingCritical = (
            ($null -eq $detSrcIp -or [string]::IsNullOrWhiteSpace([string]$detSrcIp)) -or
            ($null -eq $detHost -or [string]::IsNullOrWhiteSpace([string]$detHost)) -or
            ($null -eq $detAction -or [string]::IsNullOrWhiteSpace([string]$detAction)) -or
            ($null -eq $detOutcome -or [string]::IsNullOrWhiteSpace([string]$detOutcome))
        )

        $isInteresting = (
            $obj.decision -ne "block_ip" -or
            $gating.reason -eq "missing_ip_or_time" -or
            $mapping.source -match "fallback" -or
            $missingCritical
        )

        if (-not $isInteresting) { return }

        $rows += [pscustomobject]@{
            episode_id = $obj.episode_id
            final_decision = $obj.decision
            proposed_decision = $ev.proposed_decision
            gating_reason = $gating.reason
            schema_backend = $mapping.backend
            schema_source = $mapping.source
            schema_confidence = $mapping.confidence
            schema_signature = $mapping.signature
            llm_called = $mapping.llm_called
            cache_hit = $mapping.cache_hit
            detection_src_ip = $detSrcIp
            detection_host = $detHost
            detection_action = $detAction
            detection_outcome = $detOutcome
            detection_severity = $detSeverity
            missing_src_ip = [string]([string]::IsNullOrWhiteSpace([string]$detSrcIp))
            missing_host = [string]([string]::IsNullOrWhiteSpace([string]$detHost))
            missing_action = [string]([string]::IsNullOrWhiteSpace([string]$detAction))
            missing_outcome = [string]([string]::IsNullOrWhiteSpace([string]$detOutcome))
            normalize_duration_ms = $timing.duration_ms
        }
    }

    if ($rows.Count -gt 0) {
        $rows | Export-Csv -NoTypeInformation -Encoding UTF8 $OutPath
    } else {
        "" | Out-File -Encoding UTF8 $OutPath
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "No se encontro Python en $PythonExe"
}

$script:RunsRoot = Get-RunsRoot
Write-Host "Runs root: $script:RunsRoot"

if ([string]::IsNullOrWhiteSpace($BatchId)) {
    $BatchId = [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss")
}

Test-OllamaReady -Url $OllamaUrl

$bundleDir = Join-Path $OutDir ("diagnostics_{0}" -f $BatchId)
New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null

$emptySeedDir = Join-Path $bundleDir "_memory_seed_empty"
New-Item -ItemType Directory -Path $emptySeedDir -Force | Out-Null

Write-Host "Python: $PythonExe"
Write-Host "OutDir: $OutDir"
Write-Host "BatchId: $BatchId"
Write-Host "BaseSeed entrenamiento: $BaseSeed | Offset train/eval: $TrainEvalSeedOffset"
Write-Host "Diagnostico: static vs dynamic, memoria, MCP y evidencia cualitativa."

$trainSeed = $BaseSeed
$evalSeed = $BaseSeed + $TrainEvalSeedOffset
Write-Host "Seed entrenamiento: $trainSeed"
Write-Host "Seed evaluacion: $evalSeed"

$trainId = "diag_t0_backend_a_train_mem_{0}" -f $BatchId
Invoke-Experiment `
    -ExperimentId $trainId `
    -MemorySeedDir $emptySeedDir `
    -BlueSchemaMapper "static" `
    -McpEnabled $true `
    -SwapEnabled $false `
    -SchemaAdaptMode "contract_first" `
    -BackendBAliasMode "full" `
    -BackendBDriftProfile "classic" `
    -LlmProvider "gemini" `
    -BaseSeedOverride $trainSeed

$trainedMemoryBundleDir = New-MemorySeedBundle -SourceExperimentId $trainId -RepetitionCount $Repetitions
Write-Host "Memoria entrenada por repeticion detectada en: $trainedMemoryBundleDir"

$staticNoMemId = "diag_c0_swap_static_nomem_{0}" -f $BatchId
$dynamicNoMemId = "diag_c1_swap_dynamic_nomem_{0}" -f $BatchId
$staticMemId = "diag_c2_swap_static_mem_{0}" -f $BatchId
$dynamicMemId = "diag_c3_swap_dynamic_mem_{0}" -f $BatchId
$dynamicMemNoMcpId = "diag_c4_swap_dynamic_mem_nomcp_{0}" -f $BatchId

Invoke-Experiment `
    -ExperimentId $staticNoMemId `
    -MemorySeedDir $emptySeedDir `
    -BlueSchemaMapper "static" `
    -McpEnabled $true `
    -SwapEnabled $true `
    -SchemaAdaptMode "contract_first" `
    -BackendBAliasMode "full" `
    -BackendBDriftProfile "hard4" `
    -LlmProvider "gemini" `
    -BaseSeedOverride $evalSeed

Invoke-Experiment `
    -ExperimentId $dynamicNoMemId `
    -MemorySeedDir $emptySeedDir `
    -BlueSchemaMapper "dynamic" `
    -McpEnabled $true `
    -SwapEnabled $true `
    -SchemaAdaptMode "llm_first" `
    -BackendBAliasMode "minimal" `
    -BackendBDriftProfile "hard4" `
    -LlmProvider "ollama" `
    -BaseSeedOverride $evalSeed

Invoke-Experiment `
    -ExperimentId $staticMemId `
    -MemorySeedDir $trainedMemoryBundleDir `
    -BlueSchemaMapper "static" `
    -McpEnabled $true `
    -SwapEnabled $true `
    -SchemaAdaptMode "contract_first" `
    -BackendBAliasMode "full" `
    -BackendBDriftProfile "hard4" `
    -LlmProvider "gemini" `
    -BaseSeedOverride $evalSeed

Invoke-Experiment `
    -ExperimentId $dynamicMemId `
    -MemorySeedDir $trainedMemoryBundleDir `
    -BlueSchemaMapper "dynamic" `
    -McpEnabled $true `
    -SwapEnabled $true `
    -SchemaAdaptMode "llm_first" `
    -BackendBAliasMode "minimal" `
    -BackendBDriftProfile "hard4" `
    -LlmProvider "ollama" `
    -BaseSeedOverride $evalSeed

Invoke-Experiment `
    -ExperimentId $dynamicMemNoMcpId `
    -MemorySeedDir $trainedMemoryBundleDir `
    -BlueSchemaMapper "dynamic" `
    -McpEnabled $false `
    -SwapEnabled $true `
    -SchemaAdaptMode "llm_first" `
    -BackendBAliasMode "minimal" `
    -BackendBDriftProfile "hard4" `
    -LlmProvider "ollama" `
    -BaseSeedOverride $evalSeed

$dynamicRobustId = $null
if ($IncludeRobustness) {
    $dynamicRobustId = "diag_c3_swap_dynamic_mem_robust_{0}" -f $BatchId
    Invoke-Experiment `
        -ExperimentId $dynamicRobustId `
        -MemorySeedDir $trainedMemoryBundleDir `
        -BlueSchemaMapper "dynamic" `
        -McpEnabled $true `
        -SwapEnabled $true `
        -SchemaAdaptMode "llm_first" `
        -BackendBAliasMode "minimal" `
        -BackendBDriftProfile "hard4" `
        -LlmProvider "ollama" `
        -RepetitionsOverride 10 `
        -BaseSeedOverride $evalSeed
}

$rows = @()
$rows += Get-ExperimentSummaryRow -ExperimentId $staticNoMemId -Label "static_nomem_control"
$rows += Get-ExperimentSummaryRow -ExperimentId $dynamicNoMemId -Label "dynamic_nomem_mapper_only"
$rows += Get-ExperimentSummaryRow -ExperimentId $staticMemId -Label "static_mem_control"
$rows += Get-ExperimentSummaryRow -ExperimentId $dynamicMemId -Label "dynamic_mem_main"
$rows += Get-ExperimentSummaryRow -ExperimentId $dynamicMemNoMcpId -Label "dynamic_mem_nomcp"
if ($dynamicRobustId) {
    $rows += Get-ExperimentSummaryRow -ExperimentId $dynamicRobustId -Label "dynamic_mem_robust"
}

$comparisonCsv = Join-Path $bundleDir "diagnostic_comparison.csv"
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 $comparisonCsv

$qualitativeTarget = if ($dynamicRobustId) { $dynamicRobustId } else { $dynamicMemId }
$qualitativeCsv = Join-Path $bundleDir "qualitative_mapping_failures.csv"
Export-QualitativeFailures -ExperimentId $qualitativeTarget -OutPath $qualitativeCsv

$summaryMd = Join-Path $bundleDir "diagnostic_summary.md"
$summaryLines = @(
    "# Diagnostic Batch",
    "",
    "- batch_id: $BatchId",
    "- train_memory_experiment: $trainId",
    "- static_no_memory: $staticNoMemId",
    "- dynamic_no_memory: $dynamicNoMemId",
    "- static_with_memory: $staticMemId",
    "- dynamic_with_memory: $dynamicMemId",
    "- dynamic_with_memory_no_mcp: $dynamicMemNoMcpId",
    "- dynamic_robust: $dynamicRobustId",
    "",
    "Artifacts:",
    "- diagnostic_comparison.csv",
    "- qualitative_mapping_failures.csv"
)
$summaryLines | Out-File -Encoding UTF8 $summaryMd

Write-Host ""
Write-Host "Diagnostico completado."
Write-Host "Comparacion: $comparisonCsv"
Write-Host "Ejemplos cualitativos: $qualitativeCsv"
Write-Host "Resumen: $summaryMd"
