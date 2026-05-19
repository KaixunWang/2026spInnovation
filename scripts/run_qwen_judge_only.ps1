# Qwen 4B/8B/14B: judge only (dual judges from configs/models.yaml)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("qwen_judge_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

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

Write-Log "=== judge only ==="
py -3.12 -m src.run_experiment judge --inputs @inputs *>> $log
if ($LASTEXITCODE -ne 0) { Write-Log "FAILED judge exit=$LASTEXITCODE"; exit $LASTEXITCODE }
Write-Log "=== DONE ==="
