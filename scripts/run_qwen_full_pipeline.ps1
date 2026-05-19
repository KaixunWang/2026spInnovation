# Qwen 4B/8B/14B: metrics -> judge -> merge -> scale analysis
# Continue: tqdm/progress writes to stderr and must not abort the script.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:INNOVATION_METRICS_DEVICE = "cuda"
$env:INNOVATION_HF_LOCAL_ONLY = "1"
$env:INNOVATION_METRICS_BATCH_SIZE = "16"

$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("qwen_full_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Write-Log($msg) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

$inputs = @(
    "data/generated/main_qwen3_4b.jsonl",
    "data/generated/main_qwen3_8b.jsonl",
    "data/generated/main_qwen3_14b.jsonl"
)

function Invoke-Step([string]$label, [scriptblock]$block) {
    Write-Log $label
    & $block *>> $log
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Log "FAILED $label exit=$code"
        exit $code
    }
}

Invoke-Step "=== [1/4] metrics (3 x 1140 rows, CUDA) ===" {
    py -3.12 -m src.run_experiment metrics --inputs @inputs
}

Invoke-Step "=== [2/4] judge (dual: deepseek-chat + gpt-4o, thinking disabled) ===" {
    py -3.12 -m src.run_experiment judge --inputs @inputs
}

Write-Log "=== [3/4] merge judge into metrics ==="
foreach ($tag in @("4b", "8b", "14b")) {
    $m = "data/generated/main_qwen3_${tag}_metrics.jsonl"
    $j = "data/generated/main_qwen3_${tag}_judged.jsonl"
    Invoke-Step "merge $tag" {
        py -3.12 scripts/merge_judge_into_metrics.py $m $j -o $m
    }
}

Write-Log "=== [4/4] qwen analysis scripts ==="
foreach ($script in @(
    "scripts/qwen_scale_regression.py",
    "scripts/qwen_judge_scale_regression.py",
    "scripts/plot_scale_inverted_u.py",
    "scripts/plot_scale_inverted_u_judge.py",
    "scripts/qwen_h_full_analysis.py"
)) {
    Invoke-Step "run $script" { py -3.12 $script }
}

Write-Log "=== DONE ==="
