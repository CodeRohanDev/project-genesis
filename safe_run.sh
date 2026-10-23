#!/bin/bash
# Run an experiment script under a hard RAM cap so the PC never hangs (Rohan's limit: ~12 GB total system RAM).
# Above the cap a run is killed instead of swapping; run scripts skip finished runs, so just rerun to resume.
#   ./safe_run.sh run_long.sh
cd "$(dirname "$0")"
systemd-run --user --scope -q -p MemoryHigh=8G -p MemoryMax=8500M -p MemorySwapMax=0 /bin/bash "$@"
