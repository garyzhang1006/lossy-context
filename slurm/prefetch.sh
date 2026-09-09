#!/usr/bin/env bash
# Run on a login node before the first GPU job.  Downloads the reference
# checkpoint into HF_HOME (for clusters whose compute nodes have no outbound
# network) and checks that the two corpora are where the jobs expect them.
# Provo (https://osf.io/sjefs/) and SUBTLEX-US
# (https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus)
# are downloaded by hand behind a browser, so this script only verifies them.
#
#   bash slurm/prefetch.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(os.environ["LCSA_MODEL"],
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print(f"{os.environ['LCSA_MODEL']} -> {p}")
PY
missing=0
for f in "$LCSA_PROVO/Provo_Corpus-Predictability_Norms.csv" \
         "$LCSA_PROVO/Provo_Corpus-Eyetracking_Data.csv" "$LCSA_SUBTLEX"; do
    if [ -f "$f" ]; then echo "found   $f"; else echo "MISSING $f"; missing=1; fi
done
[ "$missing" = 0 ] || { echo "copy the missing files into place, then submit"; exit 1; }
echo "prefetch complete; submit with HF_HUB_OFFLINE=1 if the compute nodes are offline"
