"""Simulated cloud dispatch endpoint. Run: python scripts/dispatch_stub.py"""
import argparse

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Dispatch stub (simulated cloud)")
REPORTS: list[dict] = []   # one entry per alert event_id, in arrival order


def receive_report(report: dict) -> str:
    """Dedupe on event_id: an ALERT retried after a lost response is stored once; a forecast_update
    fills in the forecast of the alert it names (or is kept alone if the alert has not arrived)."""
    by_id = {r["event_id"]: r for r in REPORTS}
    event_id = report["event_id"]
    if report.get("type") == "forecast_update":
        target = by_id.get(event_id)
        if target is None:
            REPORTS.append(dict(report))
        else:
            target.update(forecast=report.get("forecast"), forecast_status=report.get("forecast_status"))
        return "update"
    early = by_id.get(event_id)
    if early is not None and early.get("type") == "forecast_update":  # the update overtook its alert
        REPORTS[REPORTS.index(early)] = {**report, "forecast": early.get("forecast"),
                                         "forecast_status": early.get("forecast_status")}
        return "alert"
    if early is not None:
        return "duplicate"
    REPORTS.append(dict(report))
    return "alert"


@app.post("/dispatch")
def receive(report: dict):
    kind = receive_report(report)
    print(f"[{kind.upper()}] {report.get('severity')} {report.get('tower_name')} {report['event_id']} "
          f"forecast={report.get('forecast_status')}", flush=True)
    return {"received": report["event_id"], "kind": kind}


@app.get("/reports")
def reports():
    return [{k: v for k, v in r.items() if k != "thumbnail_jpeg_b64"} for r in REPORTS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)
