param(
    [ValidateSet("bdd100k", "idd", "both")]
    [string]$Dataset = "bdd100k"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host ""
Write-Host "Ambulance Route Optimization - local dataset setup" -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot"
Write-Host "Dataset selection: $Dataset"
Write-Host ""
Write-Warning "This downloads complete Kaggle dataset archives, not a small sample. Check available disk space first."
Write-Host "Raw datasets are ignored by Git and must NOT be pushed to GitHub."
Write-Host ""

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found on PATH. Install Python 3.10+ and reopen PowerShell."
}

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Kaggle CLI..."
    python -m pip install --upgrade kaggle
    if ($LASTEXITCODE -ne 0) { throw "Kaggle CLI installation failed." }
}

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "Kaggle CLI is still unavailable. Close and reopen PowerShell, then rerun this script."
}

$KaggleConfig = Join-Path $HOME ".kaggle\kaggle.json"
if (-not (Test-Path $KaggleConfig) -and
    (-not $env:KAGGLE_USERNAME -or -not $env:KAGGLE_KEY)) {
    Write-Host "Kaggle authentication is not configured." -ForegroundColor Yellow
    Write-Host "1. Sign in at https://www.kaggle.com/"
    Write-Host "2. Open Account settings and create/download an API token."
    Write-Host "3. Save kaggle.json to: $KaggleConfig"
    Write-Host "4. Keep that token private; never commit it."
    throw "Configure Kaggle authentication, then rerun this script."
}

$Jobs = @()
if ($Dataset -in @("bdd100k", "both")) {
    $Jobs += @{
        Name = "BDD100K"
        Slug = "alvaromalfaro/bdd100k"
        Target = Join-Path $RepoRoot "data\bdd100k"
    }
}
if ($Dataset -in @("idd", "both")) {
    $Jobs += @{
        Name = "IDD"
        Slug = "mitanshuchakrawarty/new-idd-dataset"
        Target = Join-Path $RepoRoot "data\idd"
    }
}

foreach ($Job in $Jobs) {
    New-Item -ItemType Directory -Force -Path $Job.Target | Out-Null
    Write-Host ""
    Write-Host "Downloading $($Job.Name) into $($Job.Target)" -ForegroundColor Cyan
    Write-Host "Review and accept the Kaggle dataset's terms/license before proceeding."
    kaggle datasets download -d $Job.Slug -p $Job.Target --unzip
    if ($LASTEXITCODE -ne 0) {
        throw "Download/extraction failed for $($Job.Name). Check Kaggle access, disk space, and the dataset page."
    }
}

Write-Host ""
Write-Host "Download command(s) completed." -ForegroundColor Green
Write-Host "Next: inspect the actual extracted layout in the Dataset status tab and click Refresh local file discovery."
Write-Host "Do not git add data/, dataset archives, model weights, or Kaggle credentials."
