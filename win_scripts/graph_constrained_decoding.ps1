Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$env:PYTHONPATH = $RepoRoot
$env:HF_HUB_OFFLINE = "1"

$DataPath = Join-Path $RepoRoot "offline_assets\datasets"
$DataList = @("COKG_QA")
$Split = "test"
$IndexLen = 2
$AttnImplementation = "sdpa"
$DType = "fp16"

# Uses the model directory currently present under offline_assets/models.
$ModelPath = Join-Path $RepoRoot "offline_assets\models\Qwen_Qwen3.5-0.8B"
$ModelName = Split-Path -Leaf $ModelPath

$KValues = @(8)

Push-Location $RepoRoot
try {
    foreach ($Data in $DataList) {
        foreach ($K in $KValues) {
            & python "workflow/predict_paths_and_answers.py" `
                --data_path $DataPath `
                --d $Data `
                --split $Split `
                --index_path_length $IndexLen `
                --model_name $ModelName `
                --model_path $ModelPath `
                --k $K `
                --prompt_mode "zero-shot" `
                --generation_mode "group-beam" `
                --dtype $DType `
                --attn_implementation $AttnImplementation
        }
    }
}
finally {
    Pop-Location
}
