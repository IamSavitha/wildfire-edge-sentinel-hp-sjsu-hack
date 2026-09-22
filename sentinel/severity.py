"""Deterministic, auditable severity rules. Rule order matters; see tests."""
from sentinel.schema import ContextResult, Severity

DANGEROUS = {"wildland", "structure", "vehicle"}
BENIGN = {"campfire", "bbq_chimney", "industrial_stack", "controlled_burn"}
ATTENDABLE = {"campfire", "bbq_chimney"}


def assess(ctx: ContextResult, trend: str | None,
           in_benign_zone: bool = False, burn_scheduled: bool = False) -> Severity:
    growing = trend == "growing"
    if ctx.source_type == "fog_dust_cloud":
        return Severity.IGNORE
    if burn_scheduled and not ctx.near_structures:
        return Severity.LOG
    if ctx.source_type in DANGEROUS:
        if growing or ctx.near_structures or ctx.size_estimate == "large" or ctx.smoke_color == "black":
            return Severity.ALERT
        return Severity.MONITOR
    if ctx.source_type in BENIGN or in_benign_zone:
        if growing and ctx.size_estimate != "small":
            return Severity.ALERT
        if ctx.source_type == "industrial_stack" and in_benign_zone:
            return Severity.IGNORE
        if ctx.source_type in ATTENDABLE and ctx.attended == "no":
            return Severity.MONITOR
        return Severity.LOG
    return Severity.ALERT if growing else Severity.MONITOR


def fallback_severity(trend: str | None) -> Severity:
    """Used when the context VLM is unavailable: never silently ignore."""
    return Severity.ALERT if trend == "growing" else Severity.MONITOR
