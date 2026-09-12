param([string]$PythonExe = "D:\python\python.exe")
$ErrorActionPreference = "Stop"
$q4Dir = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $q4Dir "..\..")).Path
$specPath = Join-Path $q4Dir "Q4_Robot_v3.spec"
$distPath = Join-Path $projectRoot "dist"
$buildRoot = Join-Path $projectRoot "build"
$workPath = Join-Path $buildRoot "q4_exe_v3"
$exePath = Join-Path $distPath "Q4_Robot_v3.exe"
$simulatorPath = Join-Path $projectRoot "Jammers-simulator-win64\Jammers-simulator\Q4_Robot_v3.exe"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "找不到Python解释器：$PythonExe" }
if (Test-Path -LiteralPath $exePath) { throw "目标EXE已存在，拒绝覆盖：$exePath" }
if (Test-Path -LiteralPath $simulatorPath) { throw "仿真器目录已有同名EXE，拒绝覆盖：$simulatorPath" }
New-Item -ItemType Directory -Force -Path $distPath,$workPath | Out-Null
& $PythonExe -m PyInstaller --noconfirm --clean --distpath $distPath --workpath $workPath $specPath
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $exePath -PathType Leaf)) { throw "Q4 V3打包失败" }
& $exePath --self-check
if ($LASTEXITCODE -ne 0) { throw "Q4 V3离线自检失败" }
Copy-Item -LiteralPath $exePath -Destination $simulatorPath
$resolvedBuildRoot = (Resolve-Path -LiteralPath $buildRoot).Path
$resolvedWorkPath = (Resolve-Path -LiteralPath $workPath).Path
if (-not $resolvedWorkPath.StartsWith($resolvedBuildRoot.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw "拒绝清理项目外目录：$resolvedWorkPath" }
Remove-Item -LiteralPath $resolvedWorkPath -Recurse -Force
Write-Host "Q4 V3构建成功：$exePath"
Write-Host "仿真器目录副本：$simulatorPath"
Write-Host "SHA256：$((Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash)"
