# Versions

Each version of Wildfire Edge Sentinel is a **git tag** plus a **frozen set of model weights**, described in its own document. A later version can fail or be abandoned without touching an earlier one.

| Version | Tag | Detector | Context VLM | Headline | Document |
|---|---|---|---|---|---|
| Baseline | `v1.0.0-baseline` | YOLO-World zero-shot (`yolov8s-worldv2.pt`) | Qwen2.5-VL-7B base | D-Fire mAP50 0.002; VLM source-type 45.0%; tower bench recall 0/15 | [v1.0.0-baseline.md](v1.0.0-baseline.md) |
| Fine-tuned | `v1.1.0-finetuned` | YOLO11s fine-tuned on D-Fire | 7B + LoRA distilled from 32B | D-Fire mAP50 0.787; VLM source-type 73.6% | [v1.1.0-finetuned.md](v1.1.0-finetuned.md) |
| Tower | `v1.2.0-tower` | YOLO11s + tower fine-tune (stage 2) | same LoRA | Tower mAP50 0.728, but D-Fire 0.106 (forgetting) | [v1.2.0-tower.md](v1.2.0-tower.md) |
| Joint (recommended models) | `v1.3.0-joint` | YOLO11s joint D-Fire + tower (stage 3) | same LoRA | Tower 0.718 and D-Fire 0.748 — forgetting fixed | [v1.3.0-joint.md](v1.3.0-joint.md) |
| Demo build | `v1.3.1-demo` | as v1.3.0 | same LoRA | start_demo.sh, runbook, measured results | (patch; see v1.3.0) |
| Incident map | `v1.4.0-incident-map` | as v1.3.0 (+ D-Fire model for photos) | same LoRA | Kruthika's offline incident map + Ask Sentinel; 11/11 demo scenarios; Junction triangulated, Beaver ALERT | [v1.4.0-incident-map.md](v1.4.0-incident-map.md) |
| Live camera | `v1.5.0-live-camera` | as v1.3.0 | same LoRA | Webcam / iPhone stream through the full pipeline; ALERT pushed to a team phone (ntfy); verified gate → VLM → ALERT on the Nano | [v1.5.0-live-camera.md](v1.5.0-live-camera.md) |
| Sentinel Console | `v2.0.0-console` | as v1.3.0 | LoRA v1 (v2 selectable) | One server on 8080 with shared services: every camera → one incident, one outbox, one push; Edge vs Cloud page; works with no internet | [v2.0.0-console.md](v2.0.0-console.md) |
| **Officer console (current release)** | `v2.1.0-officer-console` | as v1.3.0 | LoRA v1 (v2 selectable) | Kruthika's officer console is the home page, with Edge vs Cloud, Mobile camera, Models and System tabs; Junction Fire: 11 vs 266 VLM calls, 0 B vs 23 MB | [v2.1.0-officer-console.md](v2.1.0-officer-console.md) |

## How versions are kept safe

1. **Code:** annotated git tags (`git tag -a`) pin each version to one commit. Tags are protected on GitHub by a ruleset (no deletion, no moving).
2. **Weights:** weights are too large for git. Each version's weights are copied to a **read-only** folder on the Nano, `~/sentinel/models/versions/<tag>/`, with a `SHA256SUMS` file. The checksums are also recorded in each version document, so any copy can be verified anywhere.
3. **Base models** (Qwen2.5-VL-7B, the 32B teacher) come from Hugging Face and are pinned by revision in the version documents.
4. **Results:** each version's measured results live in `results/` and are named in its document.

## Restore a version

```bash
git fetch --tags
git checkout v1.1.0-finetuned                         # code exactly as it was
cd ~/sentinel/models/versions/v1.1.0-finetuned
sha256sum -c SHA256SUMS                               # verify weights
```
Then serve and run as described in that version's document.

## Adding a new version

1. Finish and measure the change on a branch; open a pull request (see [CONTRIBUTING.md](../../CONTRIBUTING.md)).
2. After merge, freeze its weights into `models/versions/<tag>/` with `SHA256SUMS` and make the folder read-only.
3. Write `docs/versions/<tag>.md` from an existing one (models, checksums, config, results, known issues, restore).
4. Tag the merge commit: `git tag -a <tag> -m "<one-line summary>" && git push origin <tag>`.
5. Add a row to the table above.

Versioning: MAJOR.MINOR.PATCH — MINOR for a new model/pipeline configuration, PATCH for fixes that don't change model behaviour.
