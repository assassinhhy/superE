# Copy to news.local.ps1 and fill in. 启动superE.bat runs it if it exists.
# news.local.ps1 is in .gitignore; never commit a key.
$env:SUPERE_LLM_BASE_URL = 'https://example.invalid/v1'
$env:SUPERE_LLM_API_KEY = '<your key>'
$env:SUPERE_LLM_MODEL = 'deepseek-v4.1-flash'
$env:HTTP_PROXY = 'http://127.0.0.1:7890'
$env:HTTPS_PROXY = 'http://127.0.0.1:7890'
$env:NO_PROXY = '127.0.0.1,localhost'
$env:PYTHONUTF8 = '1'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot 'data\logs') | Out-Null
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -m supere news *> (Join-Path $PSScriptRoot 'data\logs\news.stdout.log')
