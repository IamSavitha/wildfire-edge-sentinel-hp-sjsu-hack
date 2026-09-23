"""Simulated cloud dispatch endpoint. Run: python scripts/dispatch_stub.py"""
import argparse

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Dispatch stub (simulated cloud)")
REPORTS: list[dict] = []


@app.post("/dispatch")
def receive(report: dict):
    REPORTS.append(report)
    kind = "UPDATE" if report.get("type") == "forecast_update" else "DISPATCH"
    print(f"[{kind}] {report.get('severity')} {report.get('tower_name')} {report['event_id']} "
          f"forecast={report.get('forecast_status')}", flush=True)
    return {"received": report["event_id"]}


@app.get("/reports")
def reports():
    return [{k: v for k, v in r.items() if k != "thumbnail_jpeg_b64"} for r in REPORTS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)
