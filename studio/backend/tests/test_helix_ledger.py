# SPDX-License-Identifier: AGPL-3.0-only

from core.helix_engine import ledger


def test_recent_records_reads_a_bounded_tail_without_path_read_text(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    monkeypatch.setattr(ledger, "_MAX_RECENT_READ_BYTES", 240)

    for index in range(20):
        assert ledger.append_record(
            "bounded-history",
            {"index": index, "payload": f"row-{index}-" + ("x" * 48)},
        )

    def forbidden_read_text(*args, **kwargs):
        raise AssertionError("recent_records must not read the whole ledger")

    monkeypatch.setattr(ledger.Path, "read_text", forbidden_read_text)
    rows = ledger.recent_records("bounded-history", limit=2_000)

    assert rows
    assert rows[-1]["index"] == 19
    assert rows[0]["index"] > 0
    assert len(rows) < 20

