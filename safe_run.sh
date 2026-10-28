#!/bin/bash
# Run an experiment script under a hard RAM cap so the PC never hangs (the author's limit: ~12 GB total system RAM).
# Above the cap a run is killed instead of swapping; run scripts skip finished runs, so just rerun to resume.
#   ./safe_run.sh run_long.sh      (optional env: CAP=, HIGH=, UNIT= scope name, to change limits live)
cd "$(dirname "$0")"
systemd-run --user --scope -q ${UNIT:+--unit=$UNIT} -p MemoryHigh=${HIGH:-8G} -p MemoryMax=${CAP:-8500M} -p MemorySwapMax=0 /bin/bash "$@"
