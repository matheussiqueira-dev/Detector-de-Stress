"""
Testes unitários para stresscam.recorder.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
import csv
import json
import tempfile
import time
from pathlib import Path

import pytest

from stresscam.recorder import SessionRecorder, _std


class TestStd:
    def test_empty_list_returns_zero(self):
        assert _std([]) == 0.0

    def test_single_element_returns_zero(self):
        assert _std([0.5]) == 0.0

    def test_known_std(self):
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = _std(values)
        # std populacional ≈ 2.0
        assert abs(result - 2.0) < 0.01


class TestSessionRecorder:
    def test_initial_n_readings_zero(self):
        rec = SessionRecorder()
        assert rec.n_readings == 0

    def test_record_increases_count(self):
        rec = SessionRecorder()
        rec.record(ts=time.time(), score=0.4, trend=0.01)
        rec.record(ts=time.time(), score=0.5, trend=0.02)
        assert rec.n_readings == 2

    def test_save_empty_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = SessionRecorder(output_dir=tmpdir)
            result = rec.save()
            assert result == []

    def test_save_creates_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = SessionRecorder(output_dir=tmpdir)
            now = time.time()
            for i in range(5):
                rec.record(ts=now + i, score=float(i) / 10, trend=0.01)
            paths = rec.save()
            assert len(paths) == 2
            extensions = {p.suffix for p in paths}
            assert ".json" in extensions
            assert ".csv" in extensions

    def test_json_content_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = SessionRecorder(output_dir=tmpdir)
            now = time.time()
            for i in range(3):
                rec.record(ts=now + i, score=0.4, trend=0.0)
            paths = rec.save()
            json_path = next(p for p in paths if p.suffix == ".json")
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            assert "summary" in data
            assert "readings" in data
            assert len(data["readings"]) == 3

    def test_csv_has_correct_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = SessionRecorder(output_dir=tmpdir)
            rec.record(ts=time.time(), score=0.5, trend=0.0)
            paths = rec.save()
            csv_path = next(p for p in paths if p.suffix == ".csv")
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                assert set(reader.fieldnames or []) == {"ts", "score", "trend", "mode"}

    def test_summary_score_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = SessionRecorder(output_dir=tmpdir)
            now = time.time()
            scores = [0.2, 0.4, 0.6, 0.8]
            for i, s in enumerate(scores):
                rec.record(ts=now + i, score=s, trend=0.0)
            paths = rec.save()
            json_path = next(p for p in paths if p.suffix == ".json")
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            summary = data["summary"]
            assert summary["score_max"] == pytest.approx(0.8, abs=1e-3)
            assert summary["score_min"] == pytest.approx(0.2, abs=1e-3)
            assert summary["score_mean"] == pytest.approx(0.5, abs=1e-3)

    def test_time_above_high_threshold(self):
        """50% das leituras acima de 0.75 → time_above_high_pct ≈ 50."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = SessionRecorder(output_dir=tmpdir, threshold_high=0.75)
            now = time.time()
            for i in range(4):
                score = 0.9 if i % 2 == 0 else 0.3
                rec.record(ts=now + i, score=score, trend=0.0)
            paths = rec.save()
            json_path = next(p for p in paths if p.suffix == ".json")
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            assert data["summary"]["time_above_high_pct"] == pytest.approx(50.0, abs=0.1)
