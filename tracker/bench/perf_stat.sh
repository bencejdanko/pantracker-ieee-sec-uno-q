#!/usr/bin/env bash
# A53 PMU counters per frame for one .pte:  ./perf_stat.sh model.pte [N]
# per-frame = (counters with N runs - counters with 0 runs) / N   (user-space events; perf_event_paranoid=2)
set -euo pipefail
pte=$1; n=${2:-200}; B=$(dirname "$0")/../lib/infer_bench
EV=cycles,instructions,armv8_cortex_a53/l1d_cache_refill/,armv8_cortex_a53/l2d_cache_refill/,armv8_cortex_a53/bus_access/,armv8_cortex_a53/mem_access/
a=$(perf stat -x, -e $EV -- "$B" "$pte" -w 0 -n "$n" 2>&1 >/dev/null | grep -v "^I 00:")
b=$(perf stat -x, -e $EV -- "$B" "$pte" -w 0 -n 0 2>&1 >/dev/null | grep -v "^I 00:")
paste -d, <(echo "$a" | cut -d, -f1,3) <(echo "$b" | cut -d, -f1) | awk -F, -v n="$n" '{printf "%-45s %14.0f per frame\n", $2, ($1-$3)/n}'
