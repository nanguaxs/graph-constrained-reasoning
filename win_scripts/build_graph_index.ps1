Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$env:PYTHONPATH = $RepoRoot

Push-Location $RepoRoot
try {
    $DataList = @("RoG-webqsp", "RoG-cwq")
    $Split = "train"
    $ProcessCount = 8

    foreach ($Data in $DataList) {
        & python "workflow/build_shortest_path_index.py" `
            --d $Data `
            --split $Split `
            --n $ProcessCount
    }

    # Evaluation example:
    # $Split = "test"
    # $ProcessCount = 8
    # $Hop = 2
    # foreach ($Data in $DataList) {
    #     & python "workflow/build_graph_index.py" `
    #         --d $Data `
    #         --split $Split `
    #         --n $ProcessCount `
    #         --K $Hop
    # }
}
finally {
    Pop-Location
}
