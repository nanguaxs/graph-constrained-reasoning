Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$env:PYTHONPATH = $RepoRoot
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"

$DataPath = Join-Path $RepoRoot "offline_assets\datasets"
$DataList = @("industry-kg")
$Split = "train"

$ModelName = "gpt-4o"
$ThreadCount = 3
$RequestDelay = 0.5

Push-Location $RepoRoot
try {
    foreach ($Data in $DataList) {
        $ReasoningPath = Join-Path $RepoRoot "results/GenPaths/$Data/GCR-Qwen2-0.5B-Instruct/train/zero-shot-group-beam-k10-index_len2/predictions.jsonl"

        & python "workflow/predict_final_answer.py" `
            --data_path $DataPath `
            --d $Data `
            --split $Split `
            --model_name $ModelName `
            --reasoning_path $ReasoningPath `
            --add_path "True" `
            -n $ThreadCount `
            --request_delay $RequestDelay
    }
}
finally {
    Pop-Location
}
