"""CloudShadow: edge (measured) vs cloud-only (modeled) for the same frames."""
import threading

import pytest

from sentinel.costmodel import compare
from sentinel.netprofile import CLOUD_REQUEST_RTTS, get_profile, transfer_s
from sentinel.shadow import CLOUD_OUTPUT_TOKENS, CLOUD_PROMPT_TOKENS, CloudShadow
from sentinel.webapp_logic import cloud_frame_tokens

PRICES = {"usd_per_mtok_in": 0.2, "usd_per_mtok_out": 0.6, "usd_per_gb": 5.0}


def frame(sh, source="tower", w=1920, h=1080, nbytes=400_000, *, vlm=False, tokens=0, up=0, ms=40.0,
          online=True, vlm_ms=None):
    sh.record(source, w, h, nbytes, edge_vlm_called=vlm, edge_tokens=tokens, edge_bytes_up=up,
              edge_decide_ms=ms, online=online, vlm_ms=vlm_ms, detect_ms=40.0)


def test_cloud_side_matches_frame_tokens_and_scaled_bytes():
    sh = CloudShadow()
    frame(sh)                                              # 1920 px → 1280 px: area × (2/3)²
    s = sh.summary(PRICES, "lte")
    tin = cloud_frame_tokens(1920, 1080) + CLOUD_PROMPT_TOKENS
    assert s["cloud"]["tokens_in"] == tin and s["cloud"]["tokens_out"] == CLOUD_OUTPUT_TOKENS
    assert s["cloud"]["bytes_up"] == round(400_000 * (1280 / 1920) ** 2)
    assert s["cloud"]["usd"] == pytest.approx(tin / 1e6 * 0.2 + CLOUD_OUTPUT_TOKENS / 1e6 * 0.6
                                              + s["cloud"]["bytes_up"] / 1e9 * 5.0)
    assert s["modeled"] is True and s["method"]["cloud_time"]


def test_small_frames_are_not_scaled_up():
    sh = CloudShadow()
    frame(sh, w=640, h=480, nbytes=50_000)
    assert sh.summary()["cloud"]["bytes_up"] == 50_000


def test_edge_side_is_what_was_measured_and_savings_follow():
    sh = CloudShadow()
    for _ in range(9):
        frame(sh)
    frame(sh, vlm=True, tokens=300, up=2_000, ms=5_640.0, vlm_ms=5_600.0)
    s = sh.summary(PRICES, "lte")
    assert s["edge"]["frames"] == 10 and s["edge"]["vlm_calls"] == 1 and s["edge"]["tokens"] == 300
    assert s["edge"]["bytes_up"] == 2_000 and s["edge"]["decide_ms_p50"] == 5_640.0
    assert s["savings"]["vlm_calls_avoided"] == 9
    assert s["savings"]["bytes_x"] == pytest.approx(s["cloud"]["bytes_up"] / 2_000)
    assert s["savings"]["usd"] == pytest.approx(s["cloud"]["usd"] - s["edge"]["usd"])
    assert s["by_source"]["tower"]["frames"] == 10


def test_cloud_time_is_edge_vlm_time_plus_transfer_over_the_link():
    sh = CloudShadow()
    frame(sh, vlm=True, tokens=300, ms=5_640.0, vlm_ms=5_600.0)
    s = sh.summary(PRICES, "lte")
    nbytes = s["cloud"]["frame_bytes_p50"]
    assert s["cloud"]["decide_ms_p50"] == pytest.approx(
        5_600.0 + transfer_s(nbytes, get_profile("lte"), CLOUD_REQUEST_RTTS) * 1000)


def test_satellite_is_slower_than_fiber_and_outage_has_no_cloud_decision():
    sh = CloudShadow()
    frame(sh, vlm=True, tokens=300, ms=5_640.0, vlm_ms=5_600.0)
    fiber = sh.summary(PRICES, "fiber")["cloud"]["decide_ms_p50"]
    sat = sh.summary(PRICES, "satellite_geo")["cloud"]["decide_ms_p50"]
    assert sat > fiber
    down = sh.summary(PRICES, "outage")
    assert down["link_down"] is True and down["cloud"]["decide_ms_p50"] is None


def test_offline_frames_are_cloud_blind_but_the_edge_still_decides():
    sh = CloudShadow()
    frame(sh, online=False)
    frame(sh, online=False, vlm=True, tokens=300, ms=5_000.0)
    frame(sh)
    s = sh.summary(PRICES)
    assert s["cloud_blind_frames"] == 2 and s["edge_offline_decisions"] == 2
    assert s["cloud"]["frames"] == 1 and s["edge"]["frames"] == 3


def test_zero_prices_are_not_shown_as_zero_dollars():
    sh = CloudShadow()
    frame(sh)
    s = sh.summary({"usd_per_mtok": 0, "usd_per_gb": 0})
    assert s["prices_set"] is False and s["cloud"]["usd"] is None and s["savings"]["usd"] is None
    assert s["savings"]["bytes"] > 0                        # bytes and tokens need no prices


def test_unknown_source_and_profile_are_rejected():
    sh = CloudShadow()
    with pytest.raises(ValueError):
        frame(sh, source="drone")
    with pytest.raises(ValueError):
        sh.summary(PRICES, "carrier_pigeon")


def test_profiles_list_every_link_class():
    names = {p["name"] for p in CloudShadow.profiles()}
    assert {"fiber", "lte", "rural_cellular", "satellite_geo", "outage"} <= names


def test_fleet_matches_compare_with_measured_inputs():
    sh = CloudShadow()
    frame(sh, vlm=True, tokens=300, ms=5_640.0, vlm_ms=5_000.0)
    f = sh.fleet(towers=10, fps=1, days=30, prices=PRICES)
    assert f["rows"][0]["usd_per_day"] == compare(f["inputs"])[0]["usd_per_day"]
    assert f["rows"][0]["usd_per_day_total"] == pytest.approx(f["rows"][0]["usd_per_day"] * 30)
    assert f["inputs"]["local_vlm_s"] == 5.0
    assert {"tokens_per_full_frame", "bytes_per_frame", "local_vlm_s"} <= set(f["measured"])


def test_fleet_without_data_uses_defaults():
    f = CloudShadow().fleet(towers=4, fps=2, days=1)
    assert f["measured"] == [] and len(f["rows"]) == 3


def test_concurrent_records_are_all_counted():
    sh = CloudShadow()

    def work():
        for _ in range(500):
            frame(sh, vlm=True, tokens=1, up=1)
    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s = sh.summary()
    assert s["edge"]["frames"] == 4000 and s["edge"]["tokens"] == 4000 and s["edge"]["bytes_up"] == 4000


def test_reset_clears_everything():
    sh = CloudShadow()
    frame(sh, online=False)
    sh.reset()
    s = sh.summary()
    assert s["edge"]["frames"] == 0 and s["cloud_blind_frames"] == 0 and s["by_source"] == {}
