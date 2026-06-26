$ErrorActionPreference = 'Stop'
$tools = 'H:\Python_cls\YOLO1111111\ultralytics-main\scripts\tools'
New-Item -ItemType Directory -Force -Path $tools | Out-Null

# cleanup leftover test extraction folders
foreach ($f in @('G:\BaiduNetdiskDownload\骑行图\__verify_extract',
                 'G:\BaiduNetdiskDownload\骑行图\__test_extract')) {
  if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Recurse -Force -ErrorAction SilentlyContinue }
}

$msi = Join-Path $tools '7zip.msi'
$urls = @(
  'https://www.7-zip.org/a/7z2409-x64.msi',
  'https://www.7-zip.org/a/7z2408-x64.msi',
  'https://www.7-zip.org/a/7z2301-x64.msi'
)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ok = $false
foreach ($u in $urls) {
  try {
    Write-Output ("Downloading " + $u)
    Invoke-WebRequest -Uri $u -OutFile $msi -UseBasicParsing -TimeoutSec 120
    if ((Get-Item $msi).Length -gt 200000) { $ok = $true; break }
  } catch {
    Write-Output ("  failed: " + $_.Exception.Message)
  }
}
if (-not $ok) { throw "Could not download 7-Zip MSI from any mirror." }
Write-Output ("MSI size: " + (Get-Item $msi).Length)

# administrative (extract-only) install: no admin rights, no system install
$extractDir = Join-Path $tools 'extracted'
if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
$log = Join-Path $tools 'msi_admin.log'
$p = Start-Process msiexec.exe -ArgumentList @('/a', "`"$msi`"", '/qn', "TARGETDIR=`"$extractDir`"", '/L*v', "`"$log`"") -Wait -PassThru
Write-Output ("msiexec exit: " + $p.ExitCode)

$exe = Get-ChildItem -Path $extractDir -Recurse -Filter '7z.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
$dll = Get-ChildItem -Path $extractDir -Recurse -Filter '7z.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($exe) {
  Copy-Item $exe.FullName (Join-Path $tools '7z.exe') -Force
  if ($dll) { Copy-Item $dll.FullName (Join-Path $tools '7z.dll') -Force }
  Write-Output ("7Z_READY: " + (Join-Path $tools '7z.exe'))
} else {
  Write-Output "7z.exe not found after extraction."
}
