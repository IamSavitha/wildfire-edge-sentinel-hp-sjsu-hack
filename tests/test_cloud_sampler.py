"""CloudSampler: a few real cloud answers next to the modeled cloud-only numbers (fakes only)."""
import numpy as np

from sentinel.cloud_sampler import CloudSampler, read_env_file, sampler_from_env
from sentinel.schema import ContextResult

WILD = ContextResult(source_type="wildland", smoke_color="grey", attended="no", near_structures=False,
                     near_road=False, size_estimate="small", description="Smoke.")
FOG = WILD.model_copy(update={"source_type": "fog_dust_cloud"})


class FakeCloud:
    host = "api.example.com"

    def __init__(self, ctx=WILD, cached=False, boom=None):
        self.ctx, self.last_cached, self.boom, self.calls = ctx, cached, boom, []
        self.last_latency_ms = 900.0

    def classify(self, jpeg):
        self.calls.append(len(jpeg))
        if self.boom:
            raise self.boom
        return self.ctx, 1500


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def now(fn, *args):          # run the "background" call inline
    fn(*args)


def test_once_per_incident_within_the_hourly_cap():
    clock, cloud = Clock(), FakeCloud()
    s = CloudSampler(cloud, cap_per_hour=2, clock=clock, run=now)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert s.maybe_sample("a", frame, {"source_type": "wildland"})
    assert not s.maybe_sample("a", frame, {"source_type": "wildland"})       # once per incident
    assert s.maybe_sample("b", frame, {"source_type": "campfire"})
    assert not s.maybe_sample("c", frame, {})                                 # cap reached
    clock.t += 3601
    assert s.maybe_sample("c", frame, {})                                     # a new hour
    out = s.summary()
    assert out["enabled"] and out["n"] == 3 and out["skipped_cap"] == 1
    assert out["agreement_pct"] == 100 / 3 and out["cloud_ms_p50"] == 900.0 and out["cloud_tokens_p50"] == 1500
    assert len(cloud.calls) == 3


def test_full_frame_is_sent_at_most_1280_px():
    import cv2
    got = []

    class Keep(FakeCloud):
        def classify(self, jpeg):
            got.append(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape)
            return WILD, 1
    CloudSampler(Keep(), run=now).maybe_sample("a", np.zeros((1080, 1920, 3), np.uint8), {})
    assert got == [(720, 1280, 3)]


def test_cached_answers_and_failures_are_flagged_not_fatal():
    s = CloudSampler(FakeCloud(cached=True), run=now)
    s.maybe_sample("a", b"jpeg", {"source_type": "wildland"})
    assert s.summary()["cached"] == 1
    bad = CloudSampler(FakeCloud(boom=SystemExit("5 failed")), run=now)
    bad.maybe_sample("a", b"jpeg", {})
    assert bad.summary()["errors"] == 1 and bad.summary()["n"] == 0
    none = CloudSampler(FakeCloud(ctx=None), run=now)
    none.maybe_sample("a", b"jpeg", {})
    assert none.summary()["errors"] == 1 and none.summary()["agreement_pct"] is None


def test_no_settings_no_sampler(tmp_path):
    assert sampler_from_env(env={}, env_file=tmp_path / "missing.env") is None
    f = tmp_path / "cloud.env"
    f.write_text("export CLOUD_VLM_BASE_URL=https://api.example.com/v1\nCLOUD_VLM_MODEL='qwen'\n# note\n")
    assert read_env_file(f) == {"CLOUD_VLM_BASE_URL": "https://api.example.com/v1", "CLOUD_VLM_MODEL": "qwen"}
    assert sampler_from_env(env={}, env_file=f) is None                       # no API key


def test_settings_from_the_file_build_a_cached_sampler(tmp_path):
    f = tmp_path / "cloud.env"
    f.write_text("CLOUD_VLM_BASE_URL=https://api.example.com/v1\nCLOUD_VLM_MODEL=qwen\nCLOUD_VLM_API_KEY=sk-x\n")
    s = sampler_from_env(env={}, env_file=f, cache_path=str(tmp_path / "cache.jsonl"))
    assert s is not None and s.host == "api.example.com" and s.summary()["n"] == 0
    assert "sk-x" not in repr(s.vlm.vlm)
