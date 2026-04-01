Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$env:PYTHONPATH = $RepoRoot
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"

$DataPath = Join-Path $RepoRoot "offline_assets\datasets"
$DataList = @("kg_qa_dataset")
$Split = "train"
$IndexLen = 2

$ModelPaths = @(
    (Join-Path $RepoRoot "offline_assets\models\GCR-Qwen2-0.5B-Instruct")
)

$KValues = @(3, 5, 10)

$ReasonerModelName = "gpt-4o"
$ThreadCount = 3
$RequestDelay = 0.5
$KWaitTime = 20

Push-Location $RepoRoot
try {
    Write-Host "=========================================="
    Write-Host "Batch Inductive Reasoning Script"
    Write-Host "=========================================="

    foreach ($ModelPath in $ModelPaths) {
        $ModelNameBase = Split-Path -Leaf $ModelPath
        Write-Host "=========================================="
        Write-Host "Processing model: $ModelNameBase"
        Write-Host "=========================================="

        foreach ($Data in $DataList) {
            Write-Host "Dataset: $Data"

            foreach ($K in $KValues) {
                Write-Host "Running inductive reasoning with k=$K..."

                $ReasoningPath = Join-Path $RepoRoot "results/GenPaths/$Data/$ModelNameBase/$Split/zero-shot-group-beam-k$K-index_len$IndexLen/predictions.jsonl"

                if (-not (Test-Path $ReasoningPath)) {
                    Write-Warning "Reasoning path not found: $ReasoningPath"
                    Write-Host "Skipping k=$K..."
                    Write-Host "------------------------------------------"
                    continue
                }

                Write-Host "Using reasoning path: $ReasoningPath"

                & python "workflow/predict_final_answer.py" `
                    --data_path $DataPath `
                    --d $Data `
                    --split $Split `
                    --model_name $ReasonerModelName `
                    --reasoning_path $ReasoningPath `
                    --add_path "True" `
                    -n $ThreadCount `
                    --request_delay $RequestDelay

                Write-Host "Completed k=$K"
                Write-Host "------------------------------------------"

                if ($K -ne $KValues[-1]) {
                    Write-Host "Waiting $KWaitTime seconds before next k value..."
                    Start-Sleep -Seconds $KWaitTime
                }
            }
        }
    }

    Write-Host "=========================================="
    Write-Host "All inductive reasoning tasks completed!"
    Write-Host "=========================================="
}
finally {
    Pop-Location
}
