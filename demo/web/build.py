"""Build the in-browser detector demo (run by .github/workflows/pages.yml).

1. Downloads the published detectors from Hugging Face and exports them to ONNX.
2. Fetches a few public-domain wildfire-smoke photos from Wikimedia Commons with their attribution.
3. Writes models.json / examples.json next to index.html.

    pip install ultralytics==8.4.160 onnx onnxslim huggingface_hub
    python demo/web/build.py
"""
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

WEB = Path(__file__).resolve().parent
REPO = "SavithaVijayarangan/wildfire-edge-sentinel-detector"
MODELS = [  # (weights, imgsz, label)
    ("smoke_yolo.pt", 640, "Field-photo model · 640 px · faster"),
    ("joint_yolo.pt", 960, "Tower model · 960 px · deployed in the system"),
]
EXAMPLES = [
    "File:Adams Fire Smoke Plume 2024 03.jpg",
    "File:Pagami creek smoke plume.jpeg",
    "File:Wildfire spreads in Fort McMurray.jpg",
    "File:Smoke plume (19162389836).jpg",
]
UA = {"User-Agent": "IamSavitha-wildfire-demo/1.0 (https://github.com/IamSavitha/wildfire-edge-sentinel-hp-sjsu-hack)"}


def export_models():
    out = []
    for weights, imgsz, label in MODELS:
        pt = hf_hub_download(REPO, weights)
        model = YOLO(pt)
        onnx_path = Path(model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False))
        dest = WEB / weights.replace(".pt", ".onnx")
        shutil.copy(onnx_path, dest)
        names = [model.names[i] for i in sorted(model.names)]
        out.append({"file": dest.name, "imgsz": imgsz, "label": label, "names": names})
        print(f"exported {dest.name} ({dest.stat().st_size / 1e6:.1f} MB) classes={names}")
    (WEB / "models.json").write_text(json.dumps(out, indent=2))


def strip_html(s):
    import re
    return re.sub(r"<[^>]+>", "", s or "").strip()


def fetch_examples():
    q = urllib.parse.urlencode({"action": "query", "titles": "|".join(EXAMPLES), "prop": "imageinfo",
                                "iiprop": "url|extmetadata", "iiurlwidth": 1280, "format": "json"})
    req = urllib.request.Request(f"https://commons.wikimedia.org/w/api.php?{q}", headers=UA)
    pages = json.loads(urllib.request.urlopen(req, timeout=60).read())["query"]["pages"].values()
    (WEB / "examples").mkdir(exist_ok=True)
    items = []
    for i, p in enumerate(sorted(pages, key=lambda p: EXAMPLES.index(p["title"]) if p["title"] in EXAMPLES else 99)):
        if "imageinfo" not in p:
            continue
        info = p["imageinfo"][0]
        md = info.get("extmetadata", {})
        lic = strip_html(md.get("LicenseShortName", {}).get("value", ""))
        if "public domain" not in lic.lower() and "cc0" not in lic.lower():
            print(f"skipping {p['title']}: license {lic!r}")
            continue
        fname = f"example_{i}.jpg"
        img = urllib.request.urlopen(urllib.request.Request(info["thumburl"], headers=UA), timeout=60).read()
        (WEB / "examples" / fname).write_bytes(img)
        items.append({"file": f"examples/{fname}", "title": p["title"][5:],
                      "author": strip_html(md.get("Artist", {}).get("value", "unknown"))[:120],
                      "license": lic, "source": info["descriptionurl"]})
        print(f"example {fname}: {p['title']} ({lic})")
    (WEB / "examples.json").write_text(json.dumps(items, indent=2))


if __name__ == "__main__":
    export_models()
    fetch_examples()
