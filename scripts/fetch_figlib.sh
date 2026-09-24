#!/usr/bin/env bash
# Download HPWREN FIgLib ignition sequences (81 frames, -40..+40 min around ignition, ~59 MB each) into
# $ASSETS/figlib/<sequence>/*.jpg, outside the repo. Rate-limited so a shared uplink is not saturated.
# Credit: HPWREN FIgLib, https://www.hpwren.ucsd.edu/  (required in derivative work)
#   ./scripts/fetch_figlib.sh                  # the default list below
#   ./scripts/fetch_figlib.sh 20250709_SteeleFire_lp-w-mobo-c ...
set -euo pipefail
ASSETS=${ASSETS:-$HOME/sentinel-assets}
RATE=${RATE:-15M}
BASE=https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar
DEFAULT=(
  # Junction Fire: one fire seen by six towers (triangulation)
  20260629_JunctionFire_bm-e-mobo-c 20260629_JunctionFire_hp-e-mobo-c 20260629_JunctionFire_hp-s-mobo-c
  20260629_JunctionFire_mg-e-mobo-c 20260629_JunctionFire_vo-n-mobo-c 20260629_JunctionFire_vo-w-mobo-c
  # Creelman Fire: three towers
  20260722_CreelmanFire_bm-s-mobo-c 20260722_CreelmanFire_cp-w-mobo-c 20260722_CreelmanFire_mg-s-mobo-c
  # Rainbow Fire from a second tower (the repo's demo set has rm-e)
  20260722_RainbowFire_bh-w-mobo-c
  20260817_Church2Fire_mlo-s-mobo-c 20260715_ThornFire_ws-n-mobo-c 20260618_MissionFire_rm-s-mobo-c
  20260622_BernardoFire_wc-w-mobo-c 20260909_GettyFire_wilson-w-mobo-c
  # the five sequences already used by the demo set
  20260722_RainbowFire_rm-e-mobo-c 20260807_BeaverFire_lp-w-mobo-c 20260816_SorrentoFire_sdsc-e-mobo-c
  20260820_ChiaFire_hp-n-mobo-c 20260906_BrengleFire_rm-s-mobo-c
)
SEQS=("$@"); [ ${#SEQS[@]} -eq 0 ] && SEQS=("${DEFAULT[@]}")
mkdir -p "$ASSETS/figlib"
for s in "${SEQS[@]}"; do
  out="$ASSETS/figlib/$s"
  if [ "$(ls "$out"/*.jpg 2>/dev/null | wc -l)" -ge 60 ]; then echo "have $s"; continue; fi
  tmp="$ASSETS/figlib/.$s.tgz"
  echo "get  $s"
  curl -sfL --limit-rate "$RATE" --retry 3 -o "$tmp" "$BASE/$s.tgz"
  mkdir -p "$out"
  tar xzf "$tmp" -C "$out" --strip-components=4 --wildcards '*.jpg'
  rm -f "$tmp"
  echo "ok   $s $(ls "$out" | wc -l) frames"
done
