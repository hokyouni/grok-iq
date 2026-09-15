from __future__ import annotations

from pathlib import Path

from app.analyzer import Thresholds
from app.persistence.account_repository import AccountRepository
from app.persistence.database import Database
from app.persistence.probe_repository import ProbeRepository

_SHAPES = {
    # Normal reasoning burst: long think, whole answer in one flush.
    "buffered_hard": {
        "output_tokens": 1200, "reasoning_tokens": 1100,
        "first_token_ms": 27000, "duration_ms": 28200, "tps": 100000.0,
    },
    # Genuine sustained speed: real generation window, no buffering.
    "fast_risk": {
        "output_tokens": 3000, "reasoning_tokens": 0,
        "first_token_ms": 200, "duration_ms": 6200, "tps": 500.0,
    },
    # Correct timing but the expected marker did not match.
    "marker_miss": {
        "output_tokens": 900, "reasoning_tokens": 800,
        "first_token_ms": 800, "duration_ms": 15000, "tps": 63.0,
    },
}


def _seed_samples(probes: ProbeRepository, account_id: int, classification: str) -> None:
    run_id = probes.create_run(
        account_id=account_id,
        account_name=f"acct-{account_id}",
        account_email=f"acct-{account_id}@example.test",
        profile_id="quality-marker",
        rounds=1,
        proxy_targets=[{"kind": "current", "id": None}],
        trigger="scheduled",
        priority=0,
        queue_limit=100,
    )
    probes.add_sample(run_id, {
        "round_number": 1,
        "target_key": "current",
        "target_kind": "current",
        "status": "done",
        "status_code": 200,
        **_SHAPES[classification],
        "upstream_tps": _SHAPES[classification]["tps"],
        "expected_matched": classification != "marker_miss",
        "classification": classification,
        "severity": 2 if classification == "buffered_hard" else 4,
    })


def _repositories(tmp_path: Path) -> tuple[AccountRepository, ProbeRepository]:
    database = Database(tmp_path / "grokiq.db")
    database.initialize()
    probes = ProbeRepository(database)
    probes.seed_defaults()
    return AccountRepository(database), probes


def test_buffered_hard_streak_does_not_become_high_risk(tmp_path: Path):
    accounts, probes = _repositories(tmp_path)
    for _ in range(3):
        _seed_samples(probes, 1, "buffered_hard")

    accounts.recalculate(1, Thresholds(), 168)
    stored = accounts.get_assessment(1)
    assert stored is not None
    assert stored["monitor_status"] != "high_risk", (
        "3x buffered_hard (normal burst delivery, markers matched) must not "
        f"escalate to high_risk isolation, got {stored['monitor_status']}"
    )


def test_fast_risk_streak_still_becomes_high_risk(tmp_path: Path):
    accounts, probes = _repositories(tmp_path)
    for _ in range(3):
        _seed_samples(probes, 2, "fast_risk")

    accounts.recalculate(2, Thresholds(), 168)
    stored = accounts.get_assessment(2)
    assert stored is not None
    assert stored["monitor_status"] == "high_risk", (
        "3x fast_risk (sustained impossible speed) must stay high_risk, "
        f"got {stored['monitor_status']}"
    )


def test_marker_miss_streak_still_becomes_high_risk(tmp_path: Path):
    accounts, probes = _repositories(tmp_path)
    for _ in range(3):
        _seed_samples(probes, 3, "marker_miss")

    accounts.recalculate(3, Thresholds(), 168)
    stored = accounts.get_assessment(3)
    assert stored is not None
    assert stored["monitor_status"] == "high_risk"
