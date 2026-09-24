#!/usr/bin/env bash
# One-time download (while online) of everything the incident map needs to run offline.
# Nothing lands in the git repo: it all goes to $ASSETS (default ~/sentinel-assets).
#   maps/sierra-pacific.pmtiles   California OSM vector tiles (Protomaps build, z0-15, ~1.5 GB)
#   web/lib, web/fonts, web/sprites  MapLibre GL 5, pmtiles.js, Protomaps basemap style + glyphs + icons
#   web/data/ca_counties.geojson  California county boundaries (US Census, via plotly/datasets)
#   state/wx_seed.json            one real HPWREN weather-station reading per tower (sensor emulator seed)
set -euo pipefail
ASSETS=${ASSETS:-$HOME/sentinel-assets}
BUILD=${BUILD:-$(date -u -d yesterday +%Y%m%d)}   # daily Protomaps build to extract from
BBOX=-124.48,32.53,-114.13,42.01
PMTILES_VER=1.31.2 MAPLIBRE_VER=5.24.0 PMTILES_JS_VER=4.5.0 BASEMAPS_VER=5.7.2
mkdir -p "$ASSETS"/{bin,maps,web/lib,web/data,state,reference}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

arch=$(uname -m); [ "$arch" = aarch64 ] && arch=arm64
if [ ! -x "$ASSETS/bin/pmtiles" ]; then
  curl -sfL "https://github.com/protomaps/go-pmtiles/releases/download/v$PMTILES_VER/go-pmtiles_${PMTILES_VER}_Linux_$arch.tar.gz" \
    | tar xz -C "$ASSETS/bin" pmtiles
fi
if [ ! -s "$ASSETS/maps/sierra-pacific.pmtiles" ]; then
  "$ASSETS/bin/pmtiles" extract "https://build.protomaps.com/$BUILD.pmtiles" "$ASSETS/maps/sierra-pacific.pmtiles" \
    --bbox=$BBOX --maxzoom=15
fi

(cd "$tmp" && npm pack --silent "maplibre-gl@$MAPLIBRE_VER" "pmtiles@$PMTILES_JS_VER" "@protomaps/basemaps@$BASEMAPS_VER" >/dev/null)
for t in "$tmp"/*.tgz; do d=${t%.tgz}; mkdir -p "$d"; tar xzf "$t" -C "$d" --strip-components=1; done
cp "$tmp/maplibre-gl-$MAPLIBRE_VER/dist/maplibre-gl.js" "$tmp/maplibre-gl-$MAPLIBRE_VER/dist/maplibre-gl.css" \
   "$tmp/pmtiles-$PMTILES_JS_VER/dist/pmtiles.js" "$tmp/protomaps-basemaps-$BASEMAPS_VER/dist/basemaps.js" "$ASSETS/web/lib/"

curl -sfL -o "$tmp/assets.zip" https://github.com/protomaps/basemaps-assets/archive/refs/heads/main.zip
unzip -q -o "$tmp/assets.zip" -d "$tmp"
rm -rf "$ASSETS/web/fonts" "$ASSETS/web/sprites"
mv "$tmp/basemaps-assets-main/fonts" "$tmp/basemaps-assets-main/sprites" "$ASSETS/web/"

curl -sfL -o "$tmp/counties.json" https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json
python3 - "$tmp/counties.json" "$ASSETS/web/data/ca_counties.geojson" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
f = [x for x in d["features"] if x["properties"]["STATE"] == "06"]
for x in f:
    x["properties"] = {"name": x["properties"]["NAME"], "fips": x["id"]}
json.dump({"type": "FeatureCollection", "features": f}, open(sys.argv[2], "w"))
print(f"{len(f)} California counties")
EOF

if [ ! -s "$ASSETS/state/wx_seed.json" ]; then
  curl -sfL -o "$ASSETS/reference/wxtcf.json" https://cdn.hpwren.ucsd.edu/RT/wxtcf.json
  curl -sfL -o "$ASSETS/reference/sites.js" https://www.hpwren.ucsd.edu/cameras/sites.js
  python3 - "$ASSETS/reference/wxtcf.json" "$ASSETS/state/wx_seed.json" <<'EOF'
import datetime, json, sys
wx = json.load(open(sys.argv[1]))
seed = {k: wx[k] for k in ("RM-WXT536", "LP-WXT536", "HP-WXT536") if k in wx}
json.dump({"source": "https://cdn.hpwren.ucsd.edu/RT/wxtcf.json",
           "captured_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
           "readings": seed}, open(sys.argv[2], "w"), indent=1)
print("sensor seed:", ", ".join(seed))
EOF
fi
du -sh "$ASSETS"/maps "$ASSETS"/web
