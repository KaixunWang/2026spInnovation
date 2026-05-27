# =============================================================================
# pack_for_server.ps1
# Assembles all files needed for CrEval pairwise scoring on the SLURM server
# into a self-contained folder: creval_server_package/
#
# Usage (from repo root):
#   .\scripts\pack_for_server.ps1
#   .\scripts\pack_for_server.ps1 -OutputDir D:\upload\creval_server_package
# =============================================================================
param(
  [string]$OutputDir = "$PSScriptRoot\..\creval_server_package"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$Dest     = (New-Item -ItemType Directory -Force -Path $OutputDir).FullName

Write-Host "[pack] Repo root : $RepoRoot"
Write-Host "[pack] Output    : $Dest"
Write-Host ""

function Copy-Item-Ensure {
  param([string]$Src, [string]$DstRel)
  $DstFull = Join-Path $Dest $DstRel
  $DstDir  = Split-Path $DstFull -Parent
  if (-not (Test-Path $DstDir)) { New-Item -ItemType Directory -Force -Path $DstDir | Out-Null }
  if (-not (Test-Path $Src)) {
    Write-Warning "[pack] MISSING (skip): $Src"
    return
  }
  Copy-Item -Path $Src -Destination $DstFull -Force
  Write-Host "[pack] + $DstRel"
}

function Copy-Dir-Ensure {
  param([string]$Src, [string]$DstRel, [string[]]$Exclude = @())
  $DstFull = Join-Path $Dest $DstRel
  if (-not (Test-Path $Src)) {
    Write-Warning "[pack] MISSING dir (skip): $Src"
    return
  }
  $params = @{
    Path        = $Src
    Destination = $DstFull
    Recurse     = $true
    Force       = $true
  }
  if ($Exclude.Count -gt 0) { $params['Exclude'] = $Exclude }
  Copy-Item @params
  Write-Host "[pack] + $DstRel/"
}

# ---------------------------------------------------------------------------
# 1. SLURM / setup scripts  (placed at package root for easy upload)
# ---------------------------------------------------------------------------
Copy-Item-Ensure "$RepoRoot\scripts\server\setup_creval_env.sh" "setup_creval_env.sh"
Copy-Item-Ensure "$RepoRoot\scripts\server\run_creval_pairwise.sh" "run_creval_pairwise.sh"

# ---------------------------------------------------------------------------
# 2. Python scoring scripts
# ---------------------------------------------------------------------------
Copy-Item-Ensure "$RepoRoot\scripts\__init__.py"                        "scripts\__init__.py"
Copy-Item-Ensure "$RepoRoot\scripts\score_creval_all_pairs.py"          "scripts\score_creval_all_pairs.py"
Copy-Item-Ensure "$RepoRoot\scripts\merge_external_eval_into_metrics.py" "scripts\merge_external_eval_into_metrics.py"

# ---------------------------------------------------------------------------
# 3. src package (corpus loading)
# ---------------------------------------------------------------------------
Copy-Item-Ensure "$RepoRoot\src\__init__.py"      "src\__init__.py"
Copy-Item-Ensure "$RepoRoot\src\config_loader.py" "src\config_loader.py"
Copy-Item-Ensure "$RepoRoot\src\corpus.py"        "src\corpus.py"
Copy-Item-Ensure "$RepoRoot\src\io_utils.py"      "src\io_utils.py"

# ---------------------------------------------------------------------------
# 4. CrEval API files
# ---------------------------------------------------------------------------
Copy-Item-Ensure "$RepoRoot\CrEval\inference.py"         "CrEval\inference.py"
Copy-Item-Ensure "$RepoRoot\CrEval\requirements.api.txt" "CrEval\requirements.api.txt"

# ---------------------------------------------------------------------------
# 5. Generated rewriting data  (4 model files, repeat_idx=0 rows only)
# ---------------------------------------------------------------------------
$GenDir  = Join-Path $RepoRoot "data\generated"
$GenDest = Join-Path $Dest "data\generated"
New-Item -ItemType Directory -Force -Path $GenDest | Out-Null

foreach ($f in @("main.jsonl","main_qwen3_4b.jsonl","main_qwen3_8b.jsonl","main_qwen3_14b.jsonl")) {
  $src = Join-Path $GenDir $f
  if (-not (Test-Path $src)) { Write-Warning "[pack] MISSING: data\generated\$f"; continue }

  # Filter to repeat_idx=0 only to reduce file size
  $dst = Join-Path $GenDest $f
  $lines = Get-Content $src -Encoding UTF8
  $kept  = $lines | Where-Object {
    $_ -and ($_ | ConvertFrom-Json -ErrorAction SilentlyContinue).repeat_idx -eq 0
  }
  # Write without BOM — PowerShell 5 UTF8 default adds BOM which breaks Python json.loads
  [System.IO.File]::WriteAllLines($dst, $kept, [System.Text.UTF8Encoding]::new($false))
  Write-Host "[pack] + data\generated\$f ($($kept.Count) rows, repeat_idx=0 only)"
}

# ---------------------------------------------------------------------------
# 6. Source texts  (needed for load_sources() corpus query building)
# ---------------------------------------------------------------------------
Copy-Dir-Ensure `
  "$RepoRoot\data\source_texts" `
  "data\source_texts" `
  @("*.pyc", "__pycache__")

# Remove any .pyc / __pycache__ that Copy-Item may have included
Get-ChildItem $Dest -Recurse -Filter "__pycache__" -Directory |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $Dest -Recurse -Filter "*.pyc" |
  Remove-Item -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------------
# 7. pyproject.toml  (install marker; not strictly needed at runtime)
# ---------------------------------------------------------------------------
Copy-Item-Ensure "$RepoRoot\pyproject.toml" "pyproject.toml"

# ---------------------------------------------------------------------------
# 8. Placeholder directories on server
# ---------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path (Join-Path $Dest "models\Qwen2.5-7B-Instruct") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Dest "models\CrEval-7b")            | Out-Null
"# Upload model files here via XFTP before running the job." |
  Set-Content -Path (Join-Path $Dest "models\README.txt") -Encoding UTF8

New-Item -ItemType Directory -Force -Path (Join-Path $Dest "offload") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Dest "run_status") | Out-Null

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "========================================"
$fileCount = (Get-ChildItem $Dest -Recurse -File).Count
Write-Host "[pack] Done. $fileCount files in: $Dest"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Upload models/Qwen2.5-7B-Instruct/  to the server via XFTP"
Write-Host "  2. Upload models/CrEval-7b/             to the server via XFTP"
Write-Host "  3. Upload the entire creval_server_package/ to the server"
Write-Host "  4. On the server: bash setup_creval_env.sh"
Write-Host "  5. On the server: sbatch run_creval_pairwise.sh"
Write-Host "========================================"
