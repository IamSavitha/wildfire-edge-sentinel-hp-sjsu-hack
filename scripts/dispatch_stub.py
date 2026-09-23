"""Simulated cloud dispatch endpoint. Run: python scripts/dispatch_stub.py"""
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Dispatch stub (simulated cloud)")
REPORTS: list[dict] = []


@app.post("/dispatch")
def receive(report: dict):
    REPORTS.append(report)
    print(f"[DISPATCH] {report['severity']} {report['tower_name']} {report['event_id']} "
          f"forecast={report.get('forecast_status')}", flush=True)
    return {"received": report["event_id"]}


@app.get("/reports")
def reports():
    return [{k: v for k, v in r.items() if k != "thumbnail_jpeg_b64"} for r in REPORTS]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
