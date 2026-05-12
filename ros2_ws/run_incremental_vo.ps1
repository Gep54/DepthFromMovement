#Requires -Version 5.1
<#
  Build (unless -SkipBuild), source this workspace, run incremental_vo_node.
  Examples (source ROS 2 / pixi first so colcon and ros2 are on PATH):
    & "C:\...\DepthFromMovement\ros2_ws\run_incremental_vo.ps1"
    & "...\run_incremental_vo.ps1" -SkipBuild
    & "...\run_incremental_vo.ps1" -SkipBuild --ros-args -p use_sim_time:=true
#>
param(
    [switch] $SkipBuild
)

$ErrorActionPreference = "Stop"
$wsRoot = $PSScriptRoot
Set-Location -LiteralPath $wsRoot

if (-not $SkipBuild) {
    # Windows: --symlink-install needs Developer Mode or admin (WinError 1314 otherwise).
    if (($null -ne $env:OS) -and ($env:OS -eq "Windows_NT")) {
        colcon build --packages-select incremental_vo_ros2
    } else {
        colcon build --packages-select incremental_vo_ros2 --symlink-install
    }
}

$setupPs1 = Join-Path $wsRoot "install\setup.ps1"
if (-not (Test-Path -LiteralPath $setupPs1)) {
    Write-Error "Missing $setupPs1 — build failed or wrong directory. wsRoot=$wsRoot (from cmd.exe after build, run install\setup.bat)"
}

. $setupPs1
ros2 run incremental_vo_ros2 incremental_vo_node @args
