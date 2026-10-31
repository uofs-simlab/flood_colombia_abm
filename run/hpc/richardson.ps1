param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Deploy", "Gate", "Run", "RunAll", "Status", "Pull", "Await", "AwaitAll")]
    [string]$Action,
    [ValidateSet("Primary", "Sensitivity")]
    [string]$Study = "Primary",
    [string]$RemoteUser = "oaa721",
    [string]$TuxUser = "oaa721",
    [string]$TuxHost = "tuxworld.usask.ca",
    [string]$RemoteRepo = "/scratch/oaa721/flood_columbia",
    [string]$SshKey = (Join-Path $env:USERPROFILE ".ssh\id_ed25519_usask_hpc"),
    [string]$Python = "C:\Users\markm\.venvs\flood-columbia-v2\Scripts\python.exe",
    [ValidateRange(1, 64)]
    [int]$Workers = 64,
    [switch]$ReplaceCompleted,
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"
if ($ReplaceCompleted -and ($Action -ne "Deploy" -or $Study -ne "Primary")) {
    throw "ReplaceCompleted is only valid for a primary deployment."
}
if ($SkipCompile -and $Action -ne "Pull") {
    throw "SkipCompile is only valid when retrieving a cohort."
}
if ($Action -in @("RunAll", "AwaitAll") -and $Study -ne "Primary") {
    throw "RunAll and AwaitAll coordinate both cohorts; use Study Primary."
}
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ($RemoteRepo -notmatch '^/[A-Za-z0-9_/-]+$' -or $RemoteRepo -match '(^|/)\.\.(/|$)') {
    throw "RemoteRepo must be a simple absolute path without parent traversal."
}
foreach ($value in @($RemoteUser, $TuxUser, $TuxHost)) {
    if ($value -notmatch '^[A-Za-z0-9_.-]+$') { throw "Invalid SSH account or hostname." }
}
$proxy = 'ssh -i "' + $SshKey + '" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -W %h:%p ' + $TuxUser + '@' + $TuxHost
$options = @("-i", $SshKey, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=20",
    "-o", "ServerAliveCountMax=3", "-o", "ProxyCommand=$proxy")
$target = "${RemoteUser}@richardson"
$cohort = $Study.ToLowerInvariant()
$profileArgs = @()
$remoteTransfer = "data_collection/hpc_transfer"
$remoteBatch = "data_collection/study_hpc"
$sessionPrefix = "flood_columbia"
if ($Study -eq "Sensitivity") {
    $profileArgs = @("--sensitivity")
    $remoteTransfer += "/sensitivity"
    $remoteBatch = "data_collection/sensitivity_hpc"
    $sessionPrefix += "_sensitivity"
}
$transfer = Join-Path $root ($remoteTransfer -replace '/', '\')

function Invoke-Remote {
    param([string]$Script)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($Script -replace "`r`n", "`n")))
    & ssh @options $target "echo $encoded | base64 -d | bash"
    if ($LASTEXITCODE -ne 0) { throw "Richardson command failed (exit $LASTEXITCODE)." }
}

if ($Action -eq "Deploy") {
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $replace = if ($ReplaceCompleted) { "1" } else { "0" }
    $archive = Join-Path $transfer ("deployment_" + $stamp + ".zip")
    & $Python -B (Join-Path $PSScriptRoot "transfer.py") bundle --archive $archive @profileArgs
    if ($LASTEXITCODE -ne 0) { throw "Could not prepare the verified transfer bundle." }
    Invoke-Remote @"
set -euo pipefail
mkdir -p '$RemoteRepo'
test ! -e '$RemoteRepo/data_collection/study_hpc/parent.lock'
test ! -e '$RemoteRepo/data_collection/sensitivity_hpc/parent.lock'
if [ '$cohort' = primary ] && [ '$replace' != 1 ]; then
    test ! -e '$RemoteRepo/model/flood_model.py'
fi
"@
    & scp @options -- $archive "${target}:$RemoteRepo/deployment_$cohort.zip"
    if ($LASTEXITCODE -ne 0) { throw "Deployment upload failed." }
    $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    Invoke-Remote @"
set -euo pipefail
cd '$RemoteRepo'
echo '$hash  deployment_$cohort.zip' | sha256sum --check
python3.12 - <<'PY'
from pathlib import Path, PurePosixPath
import shutil
import zipfile
pending = []
snapshots = []
with zipfile.ZipFile('deployment_$cohort.zip') as archive:
    if len(archive.namelist()) != len(set(archive.namelist())):
        raise ValueError('Duplicate deployment paths')
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if (path.is_absolute() or '..' in path.parts or ':' in str(path)
                or info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError('Unsafe deployment path')
        target = Path(*path.parts)
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
            raise ValueError('Deployment may not follow links')
        data = archive.read(info)
        if target.exists():
            if (target.is_file() and target.read_bytes() == data
                    and not ('$replace' == '1'
                             and path.parts[:2] == ('data_collection', 'hpc_transfer'))):
                continue
            mutable = path.parts[0] in ('analysis', 'tests') or path.parts[:2] in (
                ('run', 'hpc'), ('run', 'sensitivity'))
            replacement = ('$replace' == '1' and (
                path.parts[:2] == ('data_collection', 'hpc_transfer')
                or (path.parts[0] in ('agents', 'model', 'space', 'data_collection', 'run', 'analysis', 'tests')
                    and path.suffix in ('.py', '.ps1', '.sh'))))
            if (not target.is_file()
                    or not (replacement or ('$cohort' == 'sensitivity' and mutable))):
                raise ValueError(f'Existing frozen source/input or execution evidence differs: {path}')
        if target.is_file():
            snapshots.append((path.as_posix(), target.read_bytes()))
        pending.append((target, data))
    if '$replace' == '1':
        folders = [Path('data_collection') / name
                   for name in ('study_hpc', 'sensitivity_hpc', 'hpc_transfer')]
        for folder in folders:
            if folder.exists() and (folder.is_symlink() or list(folder.rglob('parent.lock'))):
                raise ValueError(f'Cannot archive active or linked output: {folder}')
        transfer = Path('data_collection/hpc_transfer')
        for status in (transfer / 'run.status', transfer / 'sensitivity/run.status'):
            if not status.is_file() or status.read_text().strip() != '0':
                raise ValueError(f'Previous cohorts must have completed successfully: {status}')
        saved = Path('data_collection/archive/completed_$stamp')
        saved.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(saved / 'replaced_source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as snapshot:
            for name, data in snapshots:
                snapshot.writestr(name, data)
        with zipfile.ZipFile(saved / 'replaced_source.zip') as snapshot:
            for name, data in snapshots:
                if snapshot.read(name) != data:
                    raise ValueError(f'Archived source differs: {name}')
        for folder in folders:
            if folder.exists():
                shutil.move(str(folder), str(saved / folder.name))
        print(f'Previous completed execution evidence preserved in {saved}')
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
PY
if [ ! -x .venv/bin/python ]; then
python3.12 -m venv --without-pip .venv
/scratch/$RemoteUser/floodDisease_abm/.venv/bin/python -m pip --python '$RemoteRepo/.venv/bin/python' install -r environment-lock.txt
fi
.venv/bin/python -m pip check
.venv/bin/python -B -c "import sys; from pathlib import Path; sys.path.insert(0, 'run/hpc'); from transfer import execution_manifest; execution_manifest(Path('$remoteTransfer')); print('Compatible pinned scientific runtime verified.')"
"@
    Write-Output "Deployment complete. Run -Action Gate before -Action Run."
} elseif ($Action -eq "RunAll") {
    Invoke-Remote @"
set -euo pipefail
cd '$RemoteRepo'
for directory in data_collection/study_hpc data_collection/sensitivity_hpc; do
    test ! -e "`$directory/parent.lock"
done
for directory in data_collection/hpc_transfer data_collection/hpc_transfer/sensitivity; do
    test -f "`$directory/plan.json"
    test ! -e "`$directory/gate.status"
    test ! -e "`$directory/run.status"
done
test ! -e data_collection/hpc_transfer/pipeline.status
tmux new-session -d -s flood_columbia_all "cd '$RemoteRepo' && bash run/hpc/execute.sh all $Workers > data_collection/hpc_transfer/pipeline.log 2>&1"
sleep 3
tmux has-session -t flood_columbia_all
tail -n 8 data_collection/hpc_transfer/pipeline.log
echo 'Durable two-cohort pipeline active; primary and sensitivity run sequentially.'
"@
} elseif ($Action -eq "AwaitAll") {
    Invoke-Remote @"
set -euo pipefail
cd '$RemoteRepo'
echo 'Waiting for both verified cohorts, up to six hours.'
deadline=`$((SECONDS + 21600))
while [ ! -f data_collection/hpc_transfer/pipeline.status ]; do
    if [ "`$SECONDS" -ge "`$deadline" ]; then
        echo 'Pipeline wait timed out; remote jobs were not stopped.' >&2
        exit 124
    fi
    sleep 60
done
status="`$(cat data_collection/hpc_transfer/pipeline.status)"
if [ "`$status" != 0 ]; then
    tail -n 60 data_collection/hpc_transfer/pipeline.log >&2
    exit "`$status"
fi
echo 'Both cohorts completed with verified exports.'
"@
    $connection = @{
        RemoteUser = $RemoteUser; TuxUser = $TuxUser; TuxHost = $TuxHost
        RemoteRepo = $RemoteRepo; SshKey = $SshKey; Python = $Python; Workers = $Workers
    }
    foreach ($studyName in @("Primary", "Sensitivity")) {
        $deferCompile = $studyName -eq "Primary"
        & $PSCommandPath -Action Pull -Study $studyName -SkipCompile:$deferCompile @connection
    }
    $generated = Join-Path $root "analysis\paper_generated"
    $primary = Get-Content -Raw -LiteralPath (Join-Path $generated "validation.json") | ConvertFrom-Json
    $sensitivity = Get-Content -Raw -LiteralPath (Join-Path $generated "sensitivity_validation.json") | ConvertFrom-Json
    if ($primary.runs -ne 480 -or $sensitivity.new_runs -ne 960 -or
            $sensitivity.baseline_manifest_sha256 -ne $primary.study_manifest_sha256) {
        throw "The complete publication cohorts do not match the approved design."
    }
    foreach ($validation in @($primary, $sensitivity)) {
        foreach ($asset in $validation.generated_sha256.PSObject.Properties) {
            $actual = (Get-FileHash -LiteralPath (Join-Path $generated $asset.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne $asset.Value) { throw "Publication asset hash differs: $($asset.Name)" }
        }
    }
    & svn add --force (Join-Path $generated "study_discussion_organization.tex") (Join-Path $generated "study_discussion_logistics.tex")
    if ($LASTEXITCODE -ne 0) { throw "Could not schedule generated discussion assets for SVN." }
    $pdf = Join-Path (Split-Path -Parent $root) "flood_columbia_paper\flood_columbia.pdf"
    @{
        completed_utc = [DateTime]::UtcNow.ToString("o")
        primary_runs = $primary.runs; sensitivity_runs = $sensitivity.new_runs
        primary_manifest_sha256 = $primary.study_manifest_sha256
        sensitivity_manifest_sha256 = $sensitivity.sensitivity_manifest_sha256
        pdf_sha256 = (Get-FileHash -LiteralPath $pdf -Algorithm SHA256).Hash.ToLowerInvariant()
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $transfer "publication.json") -Encoding UTF8
    Write-Output "Both corrected cohorts verified; manuscript and all publication assets regenerated."
} elseif ($Action -eq "Gate" -or $Action -eq "Run") {
    $phase = $Action.ToLowerInvariant()
    $count = if ($phase -eq "gate") { [Math]::Min(4, $Workers) } else { $Workers }
    Invoke-Remote @"
set -euo pipefail
cd '$RemoteRepo'
if [ '$phase' = run ]; then
    test -f $remoteTransfer/gate.status
    test "`$(cat $remoteTransfer/gate.status)" = 0
    test -f $remoteBatch/bridge.json
fi
test ! -e $remoteBatch/parent.lock
test ! -e $remoteTransfer/$phase.status
tmux new-session -d -s ${sessionPrefix}_$phase "cd '$RemoteRepo' && bash run/hpc/execute.sh $phase $count $cohort > $remoteTransfer/$phase.log 2>&1"
sleep 3
tmux has-session -t ${sessionPrefix}_$phase
tail -n 8 $remoteTransfer/$phase.log
echo 'Durable Richardson session active: ${sessionPrefix}_$phase'
"@
} elseif ($Action -eq "Status") {
    Invoke-Remote @"
set -euo pipefail
cd '$RemoteRepo'
hostname
uptime
if [ -f data_collection/hpc_transfer/pipeline.log ]; then
    echo 'Latest two-cohort pipeline activity:'
    tail -n 5 data_collection/hpc_transfer/pipeline.log
fi
for phase in gate run; do
    if [ -f $remoteTransfer/`$phase.status ]; then
        echo "`$phase exit status: `$(cat $remoteTransfer/`$phase.status)"
    fi
    if [ -f $remoteTransfer/`$phase.log ]; then tail -n 5 $remoteTransfer/`$phase.log; fi
done
if [ -f data_collection/hpc_transfer/pipeline.status ]; then
    echo "two-cohort pipeline exit status: `$(cat data_collection/hpc_transfer/pipeline.status)"
fi
if [ -f $remoteBatch/progress.json ]; then cat $remoteBatch/progress.json; fi
if [ -f $remoteBatch/parent.lock ]; then
    cat $remoteBatch/parent.lock
fi
"@
} else {
    if ($Action -eq "Await") {
        Invoke-Remote @"
set -euo pipefail
cd '$RemoteRepo'
echo 'Waiting for verified $cohort completion, up to six hours.'
deadline=`$((SECONDS + 21600))
while [ ! -f $remoteTransfer/run.status ]; do
    if [ "`$SECONDS" -ge "`$deadline" ]; then
        echo 'Completion wait timed out; remote jobs were not stopped.' >&2
        exit 124
    fi
    sleep 60
done
status="`$(cat $remoteTransfer/run.status)"
if [ "`$status" != 0 ]; then
    tail -n 40 $remoteTransfer/run.log >&2
    exit "`$status"
fi
echo 'Remote execution and verified export completed successfully.'
"@
    }
    $remoteHash = @(Invoke-Remote @"
set -euo pipefail
cd '$RemoteRepo'
test "`$(cat $remoteTransfer/run.status)" = 0
test -f $remoteTransfer/results.zip
sha256sum $remoteTransfer/results.zip
"@)
    $expectedHash = ($remoteHash[-1] -split '\s+')[0]
    if ($expectedHash -notmatch '^[a-f0-9]{64}$') { throw "Remote archive digest was not returned." }
    $archive = Join-Path $transfer "results_richardson.zip"
    if (-not (Test-Path -LiteralPath $archive)) {
        & scp @options -- "${target}:$RemoteRepo/$remoteTransfer/results.zip" $archive
        if ($LASTEXITCODE -ne 0) { throw "Result download failed; any partial archive is preserved for inspection." }
    } else {
        Write-Output "Rechecking the existing download against Richardson before reusing it."
    }
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) { throw "Downloaded archive digest differs from Richardson." }
    Write-Output ("Downloaded archive SHA256 verified: " + $actualHash)
    & $Python -B (Join-Path $PSScriptRoot "transfer.py") import --archive $archive @profileArgs
    if ($LASTEXITCODE -ne 0) { throw "Verified import failed. Local results were not relabeled." }
    $analyzer = if ($Study -eq "Sensitivity") { "sensitivity_results.py" } else { "paper_results.py" }
    $compileArgs = @()
    if (-not $SkipCompile) { $compileArgs = @("--compile") }
    & $Python -B (Join-Path $root "analysis\$analyzer") @compileArgs
    if ($LASTEXITCODE -ne 0) { throw "Final analysis or manuscript compilation failed." }
}
