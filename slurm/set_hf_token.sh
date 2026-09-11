#!/usr/bin/env bash
# Put a Hugging Face read token where the jobs look for it, without the token
# ever appearing on screen, in the shell history, or in a process listing.
#
#   bash slurm/set_hf_token.sh        # prompts; paste the token, press Return
#
# The prompt is silent, so nothing is echoed as you paste.  The token goes to
# $HF_HOME/token with mode 600, which is the first file slurm/env.sh reads.
# `huggingface-cli login` writes the same file when HF_HOME is exported and
# ~/.cache/huggingface/token when it is not, and env.sh reads both, so either
# route works; this one exists because it needs no network and no venv, which
# matters on a login node where the venv's python cannot run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LCSA_VARS_ONLY=1 . slurm/env.sh

if [ -t 0 ]; then
    printf 'Paste the Hugging Face token (input is hidden), then press Return: ' >&2
    IFS= read -rs TOKEN
    printf '\n' >&2
else
    # Piped in, for the case where the token already sits in a password manager:
    #   pass show hf/read | bash slurm/set_hf_token.sh
    # `read` returns 1 at end of file even when it filled the variable, which
    # is what `printf '%s' "$TOKEN" |` produces, and set -e would then exit
    # with nothing written and nothing said; the emptiness check below is
    # the one that reports.
    IFS= read -r TOKEN || true
fi
TOKEN="$(printf '%s' "$TOKEN" | tr -d " \t\n\r")"

[ -n "$TOKEN" ] || { echo "nothing pasted; no file written" >&2; exit 2; }
case "$TOKEN" in
    hf_*) ;;
    *) echo "a Hugging Face token starts with hf_; that does not, so nothing was written" >&2; exit 2 ;;
esac

umask 077
mkdir -p "$HF_HOME"
printf '%s' "$TOKEN" > "$HF_HOME/token"
chmod 600 "$HF_HOME/token"
unset TOKEN

# Length and prefix only.  Never print the token itself: this output is read by
# whoever is driving the run, and a token in a transcript is a leaked credential.
echo "wrote $HF_HOME/token, $(wc -c < "$HF_HOME/token" | tr -d ' ') characters starting hf_"
echo "accept the licence at https://huggingface.co/meta-llama/Llama-3.1-8B with the same account,"
echo "then bash slurm/preflight.sh will report the gated appendix checkpoint as reachable"
