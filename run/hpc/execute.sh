#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase="${1:?Specify gate, run or all}"
workers="${2:?Specify worker count}"
if [ "$phase" = all ]; then
    status=0
    gate_workers=$((workers < 4 ? workers : 4))
    for study in primary sensitivity; do
        bash "$0" gate "$gate_workers" "$study" || status=$?
        if [ "$status" -ne 0 ]; then break; fi
        bash "$0" run "$workers" "$study" || status=$?
        if [ "$status" -ne 0 ]; then break; fi
    done
    printf '%s\n' "$status" > data_collection/hpc_transfer/pipeline.status.tmp
    mv data_collection/hpc_transfer/pipeline.status.tmp data_collection/hpc_transfer/pipeline.status
    exit "$status"
fi
cohort="${3:-primary}"
arguments=()
transfer=data_collection/hpc_transfer
if [ "$cohort" = sensitivity ]; then
    arguments=(--sensitivity)
    transfer="$transfer/sensitivity"
elif [ "$cohort" != primary ]; then
    echo "Unknown cohort: $cohort" >&2
    exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$transfer"
status=0
.venv/bin/python -u -B run/hpc/transfer.py "$phase" --workers "$workers" "${arguments[@]}" || status=$?
if [ "$phase" = run ] && [ "$status" -eq 0 ]; then
    .venv/bin/python -u -B run/hpc/transfer.py export \
        --archive "$transfer/results.zip" "${arguments[@]}" || status=$?
fi
printf '%s\n' "$status" > "$transfer/${phase}.status.tmp"
mv "$transfer/${phase}.status.tmp" "$transfer/${phase}.status"
exit "$status"
