<#
.SYNOPSIS
    Windows task runner mirroring the Makefile (for machines without GNU Make).
.EXAMPLE
    ./make.ps1 setup
    ./make.ps1 test
    ./make.ps1 demo
#>
param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
$PY = ".venv/Scripts/python.exe"

function Invoke-Py { param([string[]]$CmdArgs) & $PY @CmdArgs; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

switch ($Target) {
    "help" {
        Write-Host "ABSA targets: setup, setup-gpu, prepare-data, train, evaluate, serve, demo, export, benchmark, test, lint, type, format, check, clean"
    }
    "setup" {
        python -m venv .venv
        Invoke-Py @("-m", "pip", "install", "--upgrade", "pip", "wheel")
        Invoke-Py @("-m", "pip", "install", "-r", "requirements-cpu.txt")
        & $PY -m spacy download en_core_web_sm
        & $PY -m pre_commit install
    }
    "setup-gpu" {
        python -m venv .venv
        Invoke-Py @("-m", "pip", "install", "--upgrade", "pip", "wheel")
        Invoke-Py @("-m", "pip", "install", "-r", "requirements-gpu.txt")
        & $PY -m spacy download en_core_web_sm
    }
    "prepare-data" { Invoke-Py @("scripts/prepare_data.py") }
    "train" { Invoke-Py @("scripts/train.py", "--track", "baseline"); Invoke-Py @("scripts/train.py", "--track", "transformer") }
    "train-demo" { Invoke-Py @("scripts/train.py", "--track", "transformer", "--encoder", "bert-base-uncased", "--epochs", "3", "--output-dir", "artifacts/transformer-bert", "--no-mlflow") }
    "evaluate" { Invoke-Py @("scripts/evaluate.py") }
    "export" { Invoke-Py @("scripts/export_model.py") }
    "benchmark" { Invoke-Py @("scripts/benchmark.py") }
    "serve" { Invoke-Py @("-m", "uvicorn", "absa.serving.app:app", "--host", "0.0.0.0", "--port", "8000") }
    "demo" { Invoke-Py @("-m", "streamlit", "run", "app/streamlit_app.py") }
    "test" { Invoke-Py @("-m", "pytest") }
    "lint" { Invoke-Py @("-m", "ruff", "check", "."); Invoke-Py @("-m", "black", "--check", ".") }
    "format" { Invoke-Py @("-m", "ruff", "check", "--fix", "."); Invoke-Py @("-m", "black", ".") }
    "type" { Invoke-Py @("-m", "mypy", "src/absa") }
    "check" { & $PSCommandPath lint; & $PSCommandPath type; & $PSCommandPath test }
    default { Write-Host "Unknown target: $Target"; exit 1 }
}
