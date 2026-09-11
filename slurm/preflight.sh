#!/usr/bin/env bash
# Check the cluster and the inputs before slurm/pipeline.sh commits GPU hours.
#
#   bash slurm/preflight.sh
#
# Every check reads something the pipeline depends on and prints one line, so a
# run that would have died three jobs in fails here instead, on a login node,
# in a few seconds.  Exit 0 means the chain is safe to submit.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LCSA_VARS_ONLY=1 . slurm/env.sh

FAIL=0
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; FAIL=1; }
note() { printf '  note  %s\n' "$*"; }

echo "scheduler"
if command -v sbatch >/dev/null; then ok "sbatch $(sbatch --version 2>/dev/null | awk '{print $2}')"
else bad "no sbatch on $(hostname); run this on a login node"; fi

echo "partitions"
for part in scu-cpu scu-gpu; do
    if sinfo -h -p "$part" -o "%P" 2>/dev/null | grep -q .; then
        lim=$(sinfo -h -p "$part" -o "%l" | head -1)
        ok "$part exists, time limit $lim"
    else
        bad "$part is not a partition on this cluster"
    fi
done

echo "gpus"
# The gres string the jobs ask for must name a type this cluster actually has,
# or every GPU job sits in PENDING for ever with reason ReqNodeNotAvail.
for g in "$LCSA_GPU_GRES" "$LCSA_SWEEP_GPU_GRES"; do
    type=$(echo "$g" | cut -d: -f2)
    case "$type" in
        [0-9]*) ok "$g asks for any GPU type" ;;
        *) if sinfo -h -p scu-gpu -o "%G" 2>/dev/null | grep -q "$type"; then ok "$g is available on scu-gpu"
           else bad "no $type card on scu-gpu; set LCSA_GPU_GRES to a type sinfo -p scu-gpu -o %G lists"; fi ;;
    esac
done

echo "storage"
for d in "$LCSA_ROOT" "$HF_HOME"; do
    if mkdir -p "$d" 2>/dev/null && [ -w "$d" ]; then ok "$d is writable"
    else bad "$d is not writable from $(hostname)"; fi
done
case "$LCSA_ROOT" in
    /scu-storage03/*) bad "LCSA_ROOT is on /scu-storage03, which the compute nodes cannot see" ;;
    /athena/*|/scratch/*) ok "LCSA_ROOT is on a filesystem the compute nodes mount" ;;
    *) note "LCSA_ROOT is $LCSA_ROOT; confirm the compute nodes mount it" ;;
esac

echo "install"
if [ -f "$LCSA_VENV/bin/activate" ]; then ok "venv at $LCSA_VENV"
else bad "no venv; sbatch slurm/setup.sbatch and wait for it"; fi

echo "corpus"
for f in "$LCSA_PROVO/Provo_Corpus-Predictability_Norms.csv" \
         "$LCSA_PROVO/Provo_Corpus-Eyetracking_Data.csv" "$LCSA_SUBTLEX"; do
    [ -f "$f" ] && ok "$(basename "$f")" || bad "missing $f"
done

echo "checkpoints"
if [ -f "$HF_HOME/lcsa_prefetch.done" ]; then
    ok "$(wc -l < "$HF_HOME/lcsa_prefetch.done" | tr -d ' ') fetched; jobs will run with HF_HUB_OFFLINE=1"
    for m in "$LCSA_MODEL" $LCSA_REFS; do
        grep -qxF "$m" "$HF_HOME/lcsa_prefetch.done" || bad "$m is registered but was not fetched"
    done
else
    note "not fetched yet; sbatch slurm/prefetch.sbatch runs first in the chain anyway"
fi
GATED=""
for m in $LCSA_SWEEP_REFS; do
    case "$m" in
        meta-llama/*|mistralai/*|google/gemma*) GATED="$GATED $m" ;;
    esac
done
if [ -n "$GATED" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then ok "a token is set, so the gated appendix checkpoints can be fetched:$GATED"
    else note "no HF_TOKEN, so these appendix checkpoints will be skipped:$GATED"; fi
fi

echo "design"
NREADERS=$(set -- $E3_READERS; echo $#)
TASKS=$(( E2_SHARDS + NREADERS * E3_SHARDS + E3_BOOT_SHARDS + 1 + E4_SHARDS + E6_SHARDS ))
[ "$TASKS" -le "$LCSA_MAX_RUNNING" ] \
    && ok "$TASKS array tasks, under the $LCSA_MAX_RUNNING the QOS runs at once" \
    || note "$TASKS array tasks; QOS normal runs $LCSA_MAX_RUNNING at a time and queues the rest"
RUNNING=$(squeue -h -u "$USER" -t RUNNING 2>/dev/null | wc -l | tr -d ' ')
[ "${RUNNING:-0}" -eq 0 ] && ok "no jobs of yours are running" \
    || note "$RUNNING of your jobs are already running and count against the same $LCSA_MAX_RUNNING"

echo
[ "$FAIL" -eq 0 ] && echo "preflight passed; bash slurm/pipeline.sh" \
    || echo "preflight failed; fix the lines marked FAIL before submitting"
exit "$FAIL"
