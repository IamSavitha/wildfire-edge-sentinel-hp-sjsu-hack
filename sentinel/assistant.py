"""Ranger assistant: questions about incidents answered by the on-device Qwen2.5-VL, over local data only.

Two model calls per question, both on this box:
  1. plan  - the model picks up to three tools and their arguments (schema-constrained JSON, so it
             works on a vLLM server started without tool-calling flags);
  2. answer - code runs the tools against the local incident store and the model writes the reply
             from those facts only. "Explain" questions also get the incident's latest frame.
Nothing is sent off the device, and the reply lists every tool call so an officer can check it.
"""
import json
import logging
import time
from datetime import datetime
from typing import Callable

from sentinel.geo import compass
from sentinel.incidents import SEVERITY_RANK, haversine_km

log = logging.getLogger(__name__)
MPS_TO_MPH = 2.23694
MAX_CALLS = 3
MAX_LISTED = 25

TOOLS = {
    "attention_now": "What needs the officer's attention right now: active incidents ranked by severity, "
                     "growth, wind and alerts still waiting to be sent.",
    "summarize_period": "Counts and highlights for a time window (hours back from now, e.g. 24 for today).",
    "list_incidents": "Incidents in the last N days, optionally filtered by county or minimum severity.",
    "incident_detail": "Everything about one incident: location, camera, what the VLM saw, the batch-by-batch "
                       "monitoring timeline, wind and escalation. Use for 'explain this incident'.",
    "wind_outlook": "Current wind at an incident or camera and where smoke/fire would head.",
    "draft_dispatch": "Facts needed to draft a short radio/dispatch message for one incident.",
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"calls": {"type": "array", "minItems": 1, "maxItems": MAX_CALLS, "items": {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": list(TOOLS)},
            "hours": {"type": ["number", "null"]},
            "days": {"type": ["number", "null"]},
            "incident": {"type": ["string", "null"]},
            "county": {"type": ["string", "null"]},
            "min_severity": {"type": ["string", "null"], "enum": ["LOG", "MONITOR", "ALERT", None]},
        },
        "required": ["tool", "hours", "days", "incident", "county", "min_severity"]}}},
    "required": ["calls"],
}

PLAN_PROMPT = """You route questions from a forest officer to tools over the lookout-tower incident database.
Tools:
{tools}
Rules: pick 1 to 3 tools. "today" = summarize_period with hours 24. "this week"/"last week" = list_incidents with days 7.
"this incident"/"selected" means incident "{selected}". Use incident names or ids exactly as listed. Unused arguments are null.
Now: {now}. Known incidents (id: name, county, severity, last update): {known}"""

ANSWER_PROMPT = """You are Sentinel, the assistant running on a lookout tower's edge computer, helping a forest officer.
Answer ONLY from the FACTS below (from the tower's own database and sensors). If the facts do not cover the question, say so.
Be brief and practical: short bullets, times in local 24 h format, severities in capitals, distances in km and miles.
Never invent incidents, numbers, places or wind. Mention if wind is from an emulated sensor or a cached forecast.
Now: {now}
FACTS:
{facts}"""


def _ts(t: float | None) -> str | None:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else None


def _wind_brief(w: dict | None) -> dict | None:
    if not w:
        return None
    return {"from": f"{compass(w['dir_from_deg'])} ({w['dir_from_deg']:.0f}°)",
            "speed_mph": round(w["speed_mps"] * MPS_TO_MPH, 1),
            "gust_mph": round(w["gust_mps"] * MPS_TO_MPH, 1) if w.get("gust_mps") else None,
            "smoke_heads": compass((w["dir_from_deg"] + 180) % 360),
            "source": w.get("label") or w.get("source"), "age_min": round((w.get("age_s") or 0) / 60),
            "stale": bool(w.get("stale"))}


def _brief(i: dict) -> dict:
    last = (i.get("updates") or [{}])[-1]
    return {"id": i["id"], "name": i["name"], "severity": i["severity"], "status": i["status"],
            "county": i.get("county"), "source_type": i.get("source_type"), "started": _ts(i["created_at"]),
            "last_update": _ts(i["updated_at"]), "frames_with_smoke": i.get("detections"),
            "trend": last.get("trend"), "camera": i.get("camera_name"),
            "escalation": (i.get("escalation") or {}).get("decision"),
            "lat": round(i["lat"], 4), "lon": round(i["lon"], 4)}


class Assistant:
    def __init__(self, llm_factory: Callable[[], object], model: str, store, wind_for: Callable[[dict], dict | None],
                 clock: Callable[[], float] = time.time, max_tokens: int = 450):
        self.llm_factory = llm_factory
        self.model = model
        self.store = store
        self.wind_for = wind_for
        self.clock = clock
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self.llm_factory()
        return self._client

    # ------------------------------------------------------------ tools (local data only)

    def _find(self, ref: str | None, selected: str | None) -> dict | None:
        incs = self.store.all()
        for key in (ref, selected):
            if not key:
                continue
            k = key.strip().lower()
            for i in incs:
                if i["id"] == key or i["name"].lower() == k:
                    return i
            for i in incs:
                if k and (k in i["name"].lower() or i["name"].lower() in k):
                    return i
        return None

    def tool(self, name: str, args: dict, selected: str | None) -> dict:
        now = self.clock()
        incs = self.store.all()
        if name == "attention_now":
            active = [i for i in incs if i["status"] == "active" and i["severity"] in ("ALERT", "MONITOR")]

            def score(i):
                last = (i.get("updates") or [{}])[-1]
                return (SEVERITY_RANK[i["severity"]], last.get("trend") == "growing",
                        (i.get("escalation") or {}).get("decision") == "queued", i["updated_at"])
            ranked = sorted(active, key=score, reverse=True)[:8]
            out = []
            for i in ranked:
                b = _brief(i)
                b["wind"] = _wind_brief(self.wind_for(i))
                b["latest_update"] = ((i.get("updates") or [{}])[-1]).get("text")
                b["needs"] = ([ "alert waiting in outbox (link down)"] if b["escalation"] == "queued" else []) + \
                             (["plume growing"] if b["trend"] == "growing" else []) + \
                             (["location is an estimate from camera bearing"] if i.get("loc_source") == "camera_bearing" else []) + \
                             ([f"no update for {(now - i['updated_at']) / 86400:.0f} days: confirm status or mark resolved"]
                              if now - i["updated_at"] > 86400 else [])
                out.append(b)
            queued = sum(1 for i in incs if (i.get("escalation") or {}).get("decision") == "queued")
            return {"active_fires": len(active), "alerts_waiting_in_outbox": queued, "ranked": out}
        if name == "summarize_period":
            hours = float(args.get("hours") or 24)
            since = now - hours * 3600
            win = [i for i in incs if i["updated_at"] >= since]
            by_sev = {s: sum(1 for i in win if i["severity"] == s) for s in ("ALERT", "MONITOR", "LOG")}
            by_county: dict[str, int] = {}
            for i in win:
                by_county[i.get("county") or "unknown"] = by_county.get(i.get("county") or "unknown", 0) + 1
            return {"window": f"last {hours:g} h (since {_ts(since)})", "incidents": len(win), "by_severity": by_sev,
                    "by_county": by_county,
                    "alerts_sent": sum(1 for i in win if (i.get("escalation") or {}).get("decision") == "sent"),
                    "alerts_queued": sum(1 for i in win if (i.get("escalation") or {}).get("decision") == "queued"),
                    "incidents_list": [_brief(i) for i in win[:MAX_LISTED]]}
        if name == "list_incidents":
            days = float(args.get("days") or 7)
            since = now - days * 86400
            sel = [i for i in incs if i["updated_at"] >= since]
            if args.get("county"):
                sel = [i for i in sel if (i.get("county") or "").lower() == args["county"].lower().replace(" county", "")]
            if args.get("min_severity") in SEVERITY_RANK:
                sel = [i for i in sel if SEVERITY_RANK[i["severity"]] >= SEVERITY_RANK[args["min_severity"]]]
            return {"window": f"last {days:g} days (since {_ts(since)})", "count": len(sel),
                    "incidents": [_brief(i) for i in sel[:MAX_LISTED]]}
        if name in ("incident_detail", "draft_dispatch", "wind_outlook"):
            i = self._find(args.get("incident"), selected)
            if i is None:
                return {"error": "no such incident; ask the officer to select one on the map"}
            b = _brief(i)
            b.update(location_source=i.get("loc_source"), location_note=i.get("loc_note"),
                     bearing_from_camera=(f"{i['bearing_deg']:.0f}° {compass(i['bearing_deg'])}"
                                          if i.get("bearing_deg") is not None else None),
                     what_the_camera_saw=i.get("description"), wind=_wind_brief(self.wind_for(i)))
            if name == "incident_detail":
                b["monitoring_timeline"] = [{"at": _ts(u["t"]), "update": u.get("text")} for u in (i.get("updates") or [])][-10:]
                b["escalation_note"] = (i.get("escalation") or {}).get("note")
            if name == "wind_outlook" and b["wind"]:
                w = self.wind_for(i)
                b["downwind_km_per_hour_at_10pct_rule"] = round(w["speed_mps"] * 3.6 * 0.1, 2)
                cams = {x.get("camera_name") for x in incs if x["status"] == "active" and x["id"] != i["id"]
                        and haversine_km(i["lat"], i["lon"], x["lat"], x["lon"]) < 25}
                b["other_active_incidents_within_25km"] = sorted(c for c in cams if c)
            return b
        return {"error": f"unknown tool {name}"}

    # ------------------------------------------------------------ the two model calls

    def _chat(self, messages, schema=None, max_tokens=None):
        kw = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens or self.max_tokens}
        if schema:
            kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "plan", "schema": schema}}
        r = self.client.chat.completions.create(**kw)
        u = r.usage
        return r.choices[0].message.content or "", (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)

    def ask(self, question: str, selected: str | None = None) -> dict:
        t0 = time.perf_counter()
        now = self.clock()
        incs = self.store.all()
        sel = self._find(selected, None) if selected else None
        known = "; ".join(f"{i['id']}: {i['name']}, {i.get('county') or '?'}, {i['severity']}, {_ts(i['updated_at'])}"
                          for i in incs[:30]) or "none"
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "model_calls": 0}
        calls: list[dict] = []
        try:
            plan_txt, pi, po = self._chat(
                [{"role": "system", "content": PLAN_PROMPT.format(
                    tools="\n".join(f"- {k}: {v}" for k, v in TOOLS.items()), now=_ts(now),
                    selected=(sel["name"] if sel else "none selected"), known=known)},
                 {"role": "user", "content": question}], schema=PLAN_SCHEMA, max_tokens=200)
            usage.update(prompt_tokens=pi, completion_tokens=po, model_calls=1)
            calls = json.loads(plan_txt).get("calls", [])[:MAX_CALLS]
        except Exception as exc:  # noqa: BLE001 - fall back to a sensible default plan
            log.warning("planning failed: %s", exc)
            calls = [{"tool": "attention_now"}]
            usage["plan_error"] = str(exc)[:200]
        if not calls:
            calls = [{"tool": "attention_now"}]

        results = []
        for c in calls:
            name = c.get("tool")
            args = {k: v for k, v in c.items() if k != "tool" and v not in (None, "")}
            results.append({"tool": name, "args": args, "result": self.tool(name, args, selected)})

        ids = []
        for r in results:
            res = r["result"]
            for key in ("ranked", "incidents_list", "incidents"):
                items = res.get(key)
                if isinstance(items, list):
                    ids += [x["id"] for x in items if isinstance(x, dict) and "id" in x]
            if "id" in res:
                ids.append(res["id"])
        ids = list(dict.fromkeys(ids))

        image = None
        detail = next((r["result"] for r in results if r["tool"] == "incident_detail" and "id" in r["result"]), None)
        if detail:
            inc = self.store.get(detail["id"])
            image = inc.get("thumbnail_b64") if inc else None

        facts = json.dumps(results, default=str)[:12000]
        user = [{"type": "text", "text": question}]
        if image:
            user = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
                    {"type": "text", "text": question + "\n(The image is the tower's latest frame of this incident, "
                                                        "with the detector's box drawn on it.)"}]
        answer, source = None, "model"
        try:
            answer, ai, ao = self._chat([{"role": "system", "content": ANSWER_PROMPT.format(now=_ts(now), facts=facts)},
                                         {"role": "user", "content": user}])
            usage["prompt_tokens"] += ai
            usage["completion_tokens"] += ao
            usage["model_calls"] += 1
        except Exception as exc:  # noqa: BLE001 - the facts are still useful without prose
            log.warning("answer failed: %s", exc)
            answer, source = "The on-device model did not answer; here are the facts it would have used.", "facts_only"
        return {"question": question, "answer": answer.strip(), "source": source, "calls": results,
                "incident_ids": ids, "used_image": bool(image), "model": self.model, "usage": usage,
                "latency_ms": round((time.perf_counter() - t0) * 1000), "bytes_sent_off_device": 0}

