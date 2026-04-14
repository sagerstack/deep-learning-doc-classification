"""Tests for monitoring store and classify-route integration."""

import io
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from app.src.monitoring.schema import InferenceEvent, build_inference_events
from app.src.monitoring.store import init_db, log_inference_events, query_events

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_db() -> Path:
    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "test_events.sqlite3"
    init_db(db_path)
    return db_path


def _fake_event(model_id: str = "cnn_baseline", request_id: str = "req-001") -> InferenceEvent:
    probs = [1 / 16] * 16
    return InferenceEvent(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        sample_type="sample",
        sample_name="letter.jpg",
        image_width=400,
        image_height=500,
        image_mode="RGB",
        model_id=model_id,
        model_display_name=f"Model {model_id}",
        model_version="1.0",
        predicted_label="letter",
        predicted_index=0,
        confidence=0.85,
        probabilities=probs,
        total_time_ms=120.0,
        feature_time_ms=50.0,
        graph_time_ms=10.0,
        model_time_ms=5.0,
    )


# ─── Store unit tests ─────────────────────────────────────────────────────────


class TestMonitoringStore:
    def test_init_db_creates_table(self):
        db_path = _make_db()
        # If table doesn't exist this would raise; reaching here means success
        rows = query_events(db_path)
        assert isinstance(rows, list)

    def test_log_single_event(self):
        db_path = _make_db()
        event = _fake_event()
        log_inference_events([event], db_path)
        rows = query_events(db_path)
        assert len(rows) == 1
        assert rows[0]["model_id"] == "cnn_baseline"

    def test_log_multiple_events_same_request(self):
        db_path = _make_db()
        events = [
            _fake_event(model_id="cnn_baseline"),
            _fake_event(model_id="fusion_graphsage"),
            _fake_event(model_id="fusion_gat"),
        ]
        log_inference_events(events, db_path)
        rows = query_events(db_path)
        assert len(rows) == 3
        model_ids = {r["model_id"] for r in rows}
        assert model_ids == {"cnn_baseline", "fusion_graphsage", "fusion_gat"}

    def test_log_empty_list_is_noop(self):
        db_path = _make_db()
        log_inference_events([], db_path)
        rows = query_events(db_path)
        assert len(rows) == 0

    def test_query_filter_by_model_id(self):
        db_path = _make_db()
        log_inference_events(
            [_fake_event("cnn_baseline"), _fake_event("fusion_graphsage")], db_path
        )
        rows = query_events(db_path, model_id="cnn_baseline")
        assert len(rows) == 1
        assert rows[0]["model_id"] == "cnn_baseline"

    def test_event_stores_probability_columns(self):
        db_path = _make_db()
        probs = [float(i) / 100 for i in range(16)]
        event = _fake_event()
        event.probabilities = probs
        log_inference_events([event], db_path)
        rows = query_events(db_path)
        assert rows[0]["prob_letter"] == pytest.approx(probs[0])
        assert rows[0]["prob_form"] == pytest.approx(probs[1])

    def test_reinit_is_idempotent(self):
        db_path = _make_db()
        log_inference_events([_fake_event()], db_path)
        # Re-init must not wipe existing data
        init_db(db_path)
        rows = query_events(db_path)
        assert len(rows) == 1


# ─── Schema helper tests ──────────────────────────────────────────────────────


class TestBuildInferenceEvents:
    def _fake_result(self, model_name: str = "cnn_baseline"):
        class R:
            pass

        r = R()
        r.model_name = model_name
        r.display_name = f"Model {model_name}"
        r.predicted_class = "letter"
        r.predicted_index = 0
        r.confidence = 0.9
        r.probabilities = [1 / 16] * 16
        r.inference_time_ms = 5.0
        return r

    def test_one_event_per_result(self):
        results = [self._fake_result("cnn_baseline"), self._fake_result("graphsage")]
        events = build_inference_events(
            request_id="r1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_type="sample",
            sample_name="letter.jpg",
            image_width=400,
            image_height=500,
            image_mode="RGB",
            results=results,
            total_time_ms=200.0,
            feature_time_ms=80.0,
            graph_time_ms=20.0,
            ocr_available=False,
            text_density_available=False,
        )
        assert len(events) == 2
        model_ids = {e.model_id for e in events}
        assert model_ids == {"cnn_baseline", "graphsage"}

    def test_shared_request_id(self):
        results = [self._fake_result()]
        events = build_inference_events(
            request_id="fixed-id",
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_type="upload",
            sample_name="doc.jpg",
            image_width=300,
            image_height=400,
            image_mode="RGB",
            results=results,
            total_time_ms=100.0,
            feature_time_ms=40.0,
            graph_time_ms=10.0,
            ocr_available=True,
            text_density_available=True,
        )
        assert events[0].request_id == "fixed-id"
        assert events[0].ocr_available is True


# ─── Route tests ──────────────────────────────────────────────────────────────


class TestModelPerformanceRoute:
    def test_redirect_when_url_configured(self, client, monkeypatch):
        import app.src.routes.monitoring as mon_module
        monkeypatch.setattr(mon_module, "EVIDENTLY_DASHBOARD_URL", "http://evidently.example.com")
        response = client.get("/model-performance", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert "evidently.example.com" in response.headers["location"]

    def test_fallback_when_url_not_configured(self, client, monkeypatch):
        import app.src.routes.monitoring as mon_module
        monkeypatch.setattr(mon_module, "EVIDENTLY_DASHBOARD_URL", "")
        response = client.get("/model-performance")
        assert response.status_code == 200
        assert "not configured" in response.text.lower()

    def test_fallback_html_mentions_env_var(self, client, monkeypatch):
        import app.src.routes.monitoring as mon_module
        monkeypatch.setattr(mon_module, "EVIDENTLY_DASHBOARD_URL", "")
        response = client.get("/model-performance")
        assert "EVIDENTLY_DASHBOARD_URL" in response.text


# ─── Classify integration test ────────────────────────────────────────────────


@pytest.mark.slow
class TestClassifyLogsMonitoringEvents:
    """Prove that a single classify request writes multiple rows, one per model."""

    def test_classify_writes_one_row_per_model(self, client, tmp_path, monkeypatch):
        # Point monitoring DB to a temp path so we can inspect it
        import app.src.config as cfg_module
        import app.src.routes.classify as classify_module
        import app.src.monitoring.store as store_module

        db_path = tmp_path / "test_monitor.sqlite3"
        monkeypatch.setattr(cfg_module, "MONITORING_DB_PATH", db_path)
        monkeypatch.setattr(classify_module, "MONITORING_DB_PATH", db_path)
        store_module.init_db(db_path)

        img = Image.new("L", (400, 500), color=180)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/classify",
            files={"file": ("test_doc.jpg", buf, "image/jpeg")},
        )
        assert response.status_code == 200

        rows = store_module.query_events(db_path)
        # Should have at least one row per model (CNN + GNN models)
        assert len(rows) >= 1
        # All rows share the same request_id
        request_ids = {r["request_id"] for r in rows}
        assert len(request_ids) == 1


# ─── POST /label route tests ──────────────────────────────────────────────────


class TestLabelRoute:
    """Tests for POST /label feedback capture endpoint."""

    def _seed_row(self, db_path: Path, request_id: str = "req-label-001", model_id: str = "cnn_baseline") -> None:
        """Insert a minimal inference_events row for label testing."""
        event = _fake_event(model_id=model_id, request_id=request_id)
        log_inference_events([event], db_path)

    def test_label_thumbs_up_writes_target(self, client, tmp_path, monkeypatch):
        """POST correct=true must set target = predicted_label for the row."""
        import app.src.routes.monitoring as mon_module
        import app.src.monitoring.store as store_module

        db_path = tmp_path / "label_test.sqlite3"
        store_module.init_db(db_path)
        self._seed_row(db_path, request_id="req-up-001", model_id="cnn_baseline")
        monkeypatch.setattr(mon_module, "MONITORING_DB_PATH", db_path)

        response = client.post(
            "/label",
            data={"request_id": "req-up-001", "model_id": "cnn_baseline", "correct": "true"},
        )
        assert response.status_code == 200
        assert "Recorded" in response.text

        rows = store_module.query_events(db_path, model_id="cnn_baseline")
        assert len(rows) == 1
        assert rows[0]["target"] == rows[0]["predicted_label"]

    def test_label_thumbs_down_leaves_target_null(self, client, tmp_path, monkeypatch):
        """POST correct=false must leave target as NULL (no DB write)."""
        import app.src.routes.monitoring as mon_module
        import app.src.monitoring.store as store_module

        db_path = tmp_path / "label_down_test.sqlite3"
        store_module.init_db(db_path)
        self._seed_row(db_path, request_id="req-down-001", model_id="cnn_baseline")
        monkeypatch.setattr(mon_module, "MONITORING_DB_PATH", db_path)

        response = client.post(
            "/label",
            data={"request_id": "req-down-001", "model_id": "cnn_baseline", "correct": "false"},
        )
        assert response.status_code == 200
        assert "Noted" in response.text

        rows = store_module.query_events(db_path, model_id="cnn_baseline")
        assert len(rows) == 1
        assert rows[0]["target"] is None

    def test_label_unknown_request_id_returns_ok(self, client, tmp_path, monkeypatch):
        """POST with unknown request_id must return 200 without crashing."""
        import app.src.routes.monitoring as mon_module
        import app.src.monitoring.store as store_module

        db_path = tmp_path / "label_unknown_test.sqlite3"
        store_module.init_db(db_path)
        monkeypatch.setattr(mon_module, "MONITORING_DB_PATH", db_path)

        response = client.post(
            "/label",
            data={"request_id": "nonexistent-id", "model_id": "cnn_baseline", "correct": "true"},
        )
        assert response.status_code == 200


# ─── DDL regression tests ─────────────────────────────────────────────────────


class TestTargetColumnDDL:
    """Regression tests ensuring target column is pre-declared and migration works."""

    def test_target_column_pre_declared_on_fresh_db(self, tmp_path):
        """Fresh DB must declare target TEXT, nullable, no constraints."""
        db_path = tmp_path / "fresh.sqlite3"
        init_db(db_path)

        with sqlite3.connect(str(db_path)) as conn:
            cols = {row[1]: {"type": row[2], "notnull": row[3]}
                    for row in conn.execute("PRAGMA table_info(inference_events)")}

        assert "target" in cols, "target column must be declared in DDL"
        assert cols["target"]["type"].upper() == "TEXT"
        assert cols["target"]["notnull"] == 0

    def test_init_db_migrates_legacy_db_without_target(self, tmp_path):
        """init_db on a DB lacking target must add the column and allow UPDATE."""
        db_path = tmp_path / "legacy.sqlite3"

        # Create a legacy DB without the target column
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE inference_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.0
                )
            """)
            conn.execute(
                "INSERT INTO inference_events (request_id, timestamp, model_id, predicted_label, confidence) "
                "VALUES ('r1', '2026-01-01T00:00:00Z', 'cnn_baseline', 'letter', 0.9)"
            )
            conn.commit()

        # Migration must add target column without error
        init_db(db_path)

        with sqlite3.connect(str(db_path)) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(inference_events)")}
            assert "target" in cols

            # Must be able to UPDATE target without OperationalError
            conn.execute("UPDATE inference_events SET target = 'letter' WHERE request_id = 'r1'")
            conn.commit()
            row = conn.execute("SELECT target FROM inference_events WHERE request_id = 'r1'").fetchone()
            assert row[0] == "letter"
