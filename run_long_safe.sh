#!/bin/bash
# Hard RAM cap for the whole experiment: above 7 GB a run is killed (and skipped/rerun later), the system never swaps or hangs.
cd "$(dirname "$0")"
systemd-run --user --scope -q -p MemoryHigh=6500M -p MemoryMax=7G -p MemorySwapMax=0 /bin/bash ./run_long.sh
