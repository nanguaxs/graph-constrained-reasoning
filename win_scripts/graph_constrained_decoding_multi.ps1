Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = $RepoRoot
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"

$DataPath = Join-Path $RepoRoot "offline_assets\datasets"
$DataList = @("kg_qa_dataset")
$Split = "train"
$IndexLen = 2
$AttnImplementation = "sdpa"
$DType = "fp16"

$ModelPaths = @(
    (Join-Path $RepoRoot "offline_assets\models\GCR-Qwen2-0.5B-Instruct")
)

$KValues = @(3, 5, 10)

Push-Location $RepoRoot
try {
    foreach ($ModelPath in $ModelPaths) {
        $ModelName = Split-Path -Leaf $ModelPath
        Write-Host "=========================================="
        Write-Host "Testing model: $ModelName"
        Write-Host "=========================================="

        foreach ($Data in $DataList) {
            Write-Host "Dataset: $Data"

            foreach ($K in $KValues) {
                Write-Host "Running with k=$K..."
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

                Write-Host "Completed k=$K"
                Write-Host "------------------------------------------"
            }
        }
    }

    Write-Host "=========================================="
    Write-Host "All tests completed!"
    Write-Host "=========================================="
}
finally {
    Pop-Location
}
