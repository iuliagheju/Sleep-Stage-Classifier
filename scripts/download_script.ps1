cd C:\Users\Simona\Documents\GitHub\Sleep-Stage-Classifier

$root = "https://physionet.org/files/sleep-edfx/1.0.0"
$dest = Join-Path (Get-Location) "data\raw\sleep-edfx-1.0.0"
$subsets = @("sleep-telemetry", "sleep-cassette")

New-Item -ItemType Directory -Force -Path $dest | Out-Null

# 1) Download checksum manifest
$shaFile = Join-Path $dest "SHA256SUMS.txt"
Invoke-WebRequest -Uri "$root/SHA256SUMS.txt?download" -OutFile $shaFile

# 2) Download all files listed in SHA256SUMS for the wanted subsets
Get-Content $shaFile | ForEach-Object {
    $parts = ($_ -split "\s+")
    if ($parts.Count -lt 2) { return }

    $rel = $parts[-1].Trim()
    if (-not $rel) { return }
    if (-not ($subsets | Where-Object { $rel.StartsWith("$_/") })) { return }

    $outFile = Join-Path $dest $rel
    if (Test-Path $outFile) {
        Write-Host "Skip (exists): $rel"
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path $outFile -Parent) | Out-Null
        Invoke-WebRequest -Uri "$root/$rel?download" -OutFile $outFile
        Write-Host "Downloaded: $rel"
    }
}
