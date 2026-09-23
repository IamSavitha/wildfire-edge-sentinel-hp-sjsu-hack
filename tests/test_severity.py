from sentinel.schema import ContextResult, Severity
from sentinel.severity import assess, fallback_severity


def ctx(**kw):
    base = dict(source_type="wildland", smoke_color="grey", attended="no",
                near_structures=False, near_road=False, size_estimate="small",
                description="x")
    base.update(kw)
    return ContextResult(**base)


def test_fog_is_ignored_even_if_growing():
    assert assess(ctx(source_type="fog_dust_cloud"), trend="growing") == Severity.IGNORE


def test_growing_wildland_alerts():
    assert assess(ctx(), trend="growing") == Severity.ALERT


def test_static_small_wildland_is_monitor():
    assert assess(ctx(), trend="static") == Severity.MONITOR


def test_wildland_near_structures_alerts_without_trend():
    assert assess(ctx(near_structures=True), trend=None) == Severity.ALERT


def test_black_smoke_alerts_without_trend():
    assert assess(ctx(smoke_color="black"), trend=None) == Severity.ALERT


def test_attended_static_campfire_is_log():
    assert assess(ctx(source_type="campfire", attended="yes"), trend="static") == Severity.LOG


def test_unattended_campfire_is_monitor():
    assert assess(ctx(source_type="campfire", attended="no"), trend="static") == Severity.MONITOR


def test_campfire_out_of_control_alerts():
    c = ctx(source_type="campfire", attended="yes", size_estimate="medium")
    assert assess(c, trend="growing") == Severity.ALERT


def test_stack_in_benign_zone_is_ignored():
    c = ctx(source_type="industrial_stack", smoke_color="white")
    assert assess(c, trend="static", in_benign_zone=True) == Severity.IGNORE


def test_unknown_in_benign_zone_is_log():
    assert assess(ctx(source_type="unknown"), trend="static", in_benign_zone=True) == Severity.LOG


def test_scheduled_burn_downgrades_to_log():
    assert assess(ctx(), trend="growing", burn_scheduled=True) == Severity.LOG


def test_scheduled_burn_near_structures_still_alerts():
    assert assess(ctx(near_structures=True), trend=None, burn_scheduled=True) == Severity.ALERT


def test_unknown_growing_alerts():
    assert assess(ctx(source_type="unknown"), trend="growing") == Severity.ALERT


def test_fallback_without_context_never_ignores():
    assert fallback_severity(None) == Severity.MONITOR
    assert fallback_severity("static") == Severity.MONITOR
    assert fallback_severity("growing") == Severity.ALERT


def test_scheduled_burn_never_hides_structure_fire():
    c = ctx(source_type="structure", size_estimate="large", smoke_color="black")
    assert assess(c, trend="growing", burn_scheduled=True) == Severity.ALERT


def test_scheduled_burn_never_hides_large_black_wildland():
    c = ctx(size_estimate="large", smoke_color="black")
    assert assess(c, trend="growing", burn_scheduled=True) == Severity.ALERT


def test_large_black_benign_source_is_at_least_monitor():
    c = ctx(source_type="controlled_burn", size_estimate="large", smoke_color="black")
    assert assess(c, trend="static") == Severity.MONITOR


def test_unclear_attendance_counts_as_unattended():
    assert assess(ctx(source_type="campfire", attended="unclear"), trend="static") == Severity.MONITOR
