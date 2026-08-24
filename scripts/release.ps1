param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version
)

$ErrorActionPreference = 'Stop'
$root = (git rev-parse --show-toplevel).Trim()
Set-Location -LiteralPath $root

# Codex, Git Bash, and other Unix-flavoured Windows shells can put GNU
# `link.exe` ahead of Microsoft's linker. Import the official developer-shell
# environment before invoking Cargo so both link.exe and cl.exe are correct.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path -LiteralPath $vswhere) {
    $installation = (& $vswhere -latest -products * -property installationPath).Trim()
    if ($installation) {
        $vsDevCmd = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
        if (Test-Path -LiteralPath $vsDevCmd) {
            & cmd.exe /d /s /c "`"$vsDevCmd`" -no_logo -arch=x64 && set" |
                ForEach-Object {
                    if ($_ -match '^([^=]+)=(.*)$') {
                        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
                    }
                }
        }
    }
}

if (git status --porcelain) { throw 'working tree is not clean' }
if ((git branch --show-current).Trim() -ne 'main') { throw 'not on main' }
$tag = "v$Version"
git rev-parse --verify --quiet "refs/tags/$tag" 2>$null
if ($LASTEXITCODE -eq 0) { throw "tag $tag already exists" }

$releaseFiles = @('Cargo.toml', 'Cargo.lock', 'pyproject.toml', 'CHANGELOG.md', 'index.html', 'guide.html', 'docs/index.html', 'tests/python_smoke.py')
$committed = $false
try {
    python scripts/sync-version.py $Version
    $date = Get-Date -Format 'yyyy-MM-dd'
    $changelog = Get-Content -Raw CHANGELOG.md
    $changelog = $changelog -replace '## \[Unreleased\]', "## [Unreleased]`n`n## [$Version] - $date"
    [IO.File]::WriteAllText((Join-Path $root 'CHANGELOG.md'), $changelog.Replace("`r`n", "`n"))

    cargo build --locked
    if ($LASTEXITCODE) { throw 'cargo build failed' }
    cargo publish --dry-run --locked --allow-dirty
    if ($LASTEXITCODE) { throw 'cargo publish dry-run failed' }

    git add -- $releaseFiles
    git commit -m "Release $Version"
    $committed = $true
    git tag $tag

    $reply = Read-Host "Push main + $tag now? This publishes to crates.io and PyPI. [y/N]"
    if ($reply -in @('y', 'Y')) {
        git push origin main $tag
    } else {
        Write-Host "Not pushed. When ready: git push origin main $tag"
    }
} catch {
    if (-not $committed) { git restore --staged --worktree -- $releaseFiles 2>$null }
    throw
}
