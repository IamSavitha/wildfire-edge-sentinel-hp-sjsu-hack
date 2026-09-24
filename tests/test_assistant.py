import json
from types import SimpleNamespace

import pytest

from sentinel.assistant import Assistant
from sentinel.incidents import IncidentStore

NOW = 1_800_000_000.0
WIND = {"dir_from_deg": 270.0, "speed_mps": 5.0, "gust_mps": 8.0, "label": "emulated on-site sensor", "age_s": 30}


class FakeLLM:
    """OpenAI-shaped client: answers the planning call with `plan`, the answer call with a fixed text."""
    def __init__(self, plan, fail_plan=False, fail_answer=False):
        self.plan, self.fail_plan, self.fail_answer = plan, fail_plan, fail_answer
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        planning = "response_format" in kw
        if (planning and self.fail_plan) or (not planning and self.fail_answer):
            raise RuntimeError("server down")
        text = json.dumps({"calls": self.plan}) if planning else "- Answer from facts."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def call(tool, **kw):
    return {"tool": tool, "hours": None, "days": None, "incident": None, "county": None, "min_severity": None, **kw}


@pytest.fixture
def store(tmp_path):
    s = IncidentStore(str(tmp_path / "i.db"))
    base = {"lat": 33.4, "lon": -117.0, "site_short": "Red Mountain", "name": None, "source_type": "wildland",
            "loc_source": "camera_bearing", "bearing_deg": 95.0, "camera_name": "Red Mountain (east)"}
    a, _ = s.record({**base, "camera_id": "rm-e", "severity": "ALERT", "county": "San Diego",
                     "escalation": {"decision": "queued"}, "thumbnail_b64": "aGVsbG8="}, now=NOW - 3600)
    a["updates"] = [{"t": NOW - 3600, "text": "opened", "trend": None}, {"t": NOW - 600, "text": "growing", "trend": "growing"}]
    s.save(a)
    s.record({**base, "camera_id": "lp-w", "severity": "MONITOR", "county": "San Diego", "lat": 32.7}, now=NOW - 7200)
    s.record({**base, "camera_id": "hp-n", "severity": "LOG", "county": "Riverside"}, now=NOW - 5 * 86400)
    s.record({**base, "camera_id": "rm-s", "severity": "ALERT", "county": "San Diego"}, now=NOW - 20 * 86400)
    return s


def make(store, plan, **kw):
    llm = FakeLLM(plan, **kw)
    return Assistant(lambda: llm, "base7b", store, lambda inc: WIND, clock=lambda: NOW), llm


def test_attention_ranks_alerts_first_and_flags_what_needs_doing(store):
    a, llm = make(store, [call("attention_now")])
    r = a.ask("What needs my attention?")
    ranked = r["calls"][0]["result"]["ranked"]
    assert [x["severity"] for x in ranked] == ["ALERT", "ALERT", "MONITOR"]
    assert "alert waiting in outbox (link down)" in ranked[0]["needs"] and "plume growing" in ranked[0]["needs"]
    assert "no update for 20 days: confirm status or mark resolved" in ranked[1]["needs"]
    assert ranked[0]["wind"]["smoke_heads"] == "E" and ranked[0]["wind"]["speed_mph"] == 11.2
    assert r["answer"] == "- Answer from facts." and r["bytes_sent_off_device"] == 0
    assert r["usage"] == {"prompt_tokens": 200, "completion_tokens": 40, "model_calls": 2}
    assert r["incident_ids"] == [x["id"] for x in ranked]
    plan_req, answer_req = llm.requests
    assert plan_req["response_format"]["type"] == "json_schema" and plan_req["model"] == "base7b"
    assert "FACTS" in answer_req["messages"][0]["content"]


def test_periods_and_filters(store):
    a, _ = make(store, [call("summarize_period", hours=24), call("list_incidents", days=7),
                        call("list_incidents", days=30, county="San Diego County", min_severity="ALERT")])
    r = a.ask("today, this week and alerts this month in San Diego")
    today, week, month = (c["result"] for c in r["calls"])
    assert today["incidents"] == 2 and today["by_severity"] == {"ALERT": 1, "MONITOR": 1, "LOG": 0}
    assert today["alerts_queued"] == 1
    assert week["count"] == 3
    assert month["count"] == 2 and {x["severity"] for x in month["incidents"]} == {"ALERT"}


def test_explain_selected_incident_sends_its_frame_to_the_vlm(store):
    sel = store.all()[0]
    a, llm = make(store, [call("incident_detail", incident="this incident")])
    r = a.ask("Explain this incident", selected=sel["id"])
    d = r["calls"][0]["result"]
    assert d["id"] == sel["id"] and d["monitoring_timeline"][-1]["update"] == "growing"
    assert d["bearing_from_camera"] == "95° E" and r["used_image"]
    content = llm.requests[1]["messages"][1]["content"]
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,aGVsbG8="


def test_wind_and_dispatch_tools_and_unknown_incident(store):
    name = store.all()[0]["name"]
    a, _ = make(store, [call("wind_outlook", incident=name), call("draft_dispatch", incident=name.lower()),
                        call("incident_detail", incident="Nowhere Fire")])
    r = a.ask("wind and a radio message")
    wind, dispatch, missing = (c["result"] for c in r["calls"])
    assert wind["downwind_km_per_hour_at_10pct_rule"] == 1.8 and wind["other_active_incidents_within_25km"]
    assert dispatch["name"] == name and "monitoring_timeline" not in dispatch
    assert "error" in missing


def test_falls_back_when_the_model_is_down(store):
    a, _ = make(store, [], fail_plan=True, fail_answer=True)
    r = a.ask("anything?")
    assert r["calls"][0]["tool"] == "attention_now" and r["source"] == "facts_only"
    assert "plan_error" in r["usage"]


def test_bad_plan_calls_are_skipped_or_reported_not_raised(store):
    a, _ = make(store, ["attention_now", call("summarize_period", hours=1e12), call("list_incidents", county=5),
                        call("list_incidents", days=7)])
    r = a.ask("anything odd?")
    by_tool = [(c["tool"], c["result"]) for c in r["calls"]]
    assert [t for t, _ in by_tool] == ["summarize_period", "list_incidents", "list_incidents"]   # the string is skipped
    assert "error" in by_tool[0][1] and "error" in by_tool[1][1] and len(by_tool[1][1]["error"]) <= 200
    assert by_tool[2][1]["count"] == 3 and r["answer"] == "- Answer from facts."


def test_a_plan_with_only_non_dict_calls_falls_back_to_attention_now(store):
    a, _ = make(store, ["attention_now", 7])
    r = a.ask("what now?")
    assert [c["tool"] for c in r["calls"]] == ["attention_now"] and r["calls"][0]["result"]["ranked"]
