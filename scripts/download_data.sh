#!/usr/bin/env bash
# Downloads training/eval data onto the Nano. Verify each dataset's license and record it in README.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/dfire data/benign data/demo data/bench data/pyro data/gold

# 1) D-Fire (YOLO format; classes 0=smoke, 1=fire). Hosted via link on
#    https://github.com/gaiasd/DFireDataset. Download on laptop if needed, then:
#    scp D-Fire.zip hpX@<nano-ip>:~/sentinel/data/dfire/ && unzip there.
if [ ! -d data/dfire/train ]; then
  echo "MANUAL: place D-Fire under data/dfire/{train,test}/{images,labels}"; fi

# 2) PyroNear lookout-tower smoke (optional extra detector data / eval).
#    `hf` is the current Hugging Face CLI; older installs only ship `huggingface-cli`.
HF=$(command -v hf || command -v huggingface-cli || true)
if [ -n "$HF" ]; then
  "$HF" download pyronear/pyro-sdis --repo-type dataset --local-dir data/pyro || \
    echo "WARN: pyro-sdis download failed; continuing with D-Fire only"
else
  echo "WARN: no Hugging Face CLI found (pip install huggingface_hub); skipping pyro-sdis"
fi

# 3) FIgLib (HPWREN Fire Ignition Library) sequences for demo + trend eval:
#    browse https://www.hpwren.ucsd.edu/FIgLib/ and download 4-6 sequences into
#    data/demo/<name>/ (each a folder of time-ordered JPGs).
echo "MANUAL: FIgLib sequences -> data/demo/<sequence>/*.jpg"

# 4) Benign look-alikes (campfire, BBQ, chimney, stack, fog, low cloud): 150-300 JPGs
#    from openly licensed sources -> data/benign/*.jpg ; short clips -> data/demo/benign_*/
echo "MANUAL: benign images -> data/benign/*.jpg"

# 5) End-to-end benchmark list (scripts/bench.py): 20-40 rows of `path,label`,
#    label alert (FIgLib/wildfire) or no_alert (campfire, fog, stack, BBQ).
echo "MANUAL: data/bench/clips.csv with header 'path,label'"

# 6) Optional hand-checked gold split for the context VLM (same JSONL format as teacher labels).
echo "OPTIONAL: data/gold/gold.jsonl"
