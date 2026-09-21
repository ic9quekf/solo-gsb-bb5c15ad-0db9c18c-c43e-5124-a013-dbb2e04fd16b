"""配额使用分析引擎与 HTTP 层测试：pytest 运行。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import UsageReportRequest
from app.simulation import simulate
from app.usage_report import usage_report

client = TestClient(app)


def _report(buckets, requests, window_seconds):
    payload = UsageReportRequest.model_validate(
        {"buckets": buckets, "requests": requests, "window_seconds": window_seconds}
    )
    return usage_report(payload)


def _by_start(windows):
    return {w.start: w for w in windows}


# ---------- 消耗 / 拒绝归集 ----------

def test_consumed_counts_only_allowed_deductions():
    resp = _report(
        [{"id": "A", "capacity": 10, "refill_rate": 1, "initial": 5}],
        [
            {"id": "r1", "timestamp": 0, "cost": 10, "chain": ["A"]},  # 拒绝
            {"id": "r2", "timestamp": 5, "cost": 10, "chain": ["A"]},  # 放行扣 10
        ],
        100,
    )
    bucket = resp.buckets[0]
    assert bucket.consumed_total == 10
    assert bucket.rejected_total == 1
    win = _by_start(resp.windows)[0.0]
    assert win.buckets["A"].consumed == 10
    assert win.buckets["A"].rejected == 1
    assert win.total_consumed == 10
    assert win.allowed_requests == 1 and win.rejected_requests == 1


def test_rejected_counts_every_bucket_on_rejected_chain():
    # 链上 A、B；B 不足导致原子拒绝，但 A、B 都应各计一次拒绝
    resp = _report(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0},
            {"id": "B", "capacity": 2, "refill_rate": 0},
        ],
        [{"id": "x", "timestamp": 0, "cost": 5, "chain": ["A", "B"]}],
        10,
    )
    win = resp.windows[0]
    assert win.rejected_requests == 1
    assert win.buckets["A"].rejected == 1
    assert win.buckets["B"].rejected == 1
    assert win.buckets["A"].consumed == 0 and win.buckets["B"].consumed == 0


# ---------- 峰值占用 ----------

def test_peak_occupancy_is_capacity_minus_after_balance():
    # t=0 扣 8 -> 占用 8；补充不产生采样；t=10 扣 1（补满后余 9）-> 占用 1
    resp = _report(
        [{"id": "A", "capacity": 10, "refill_rate": 1}],
        [
            {"id": "r1", "timestamp": 0, "cost": 8, "chain": ["A"]},
            {"id": "r2", "timestamp": 10, "cost": 1, "chain": ["A"]},
        ],
        100,
    )
    assert resp.buckets[0].peak_occupancy == 8
    assert resp.buckets[0].peak_window_start == 0


def test_peak_occupancy_on_rejection_uses_unchanged_balance():
    # 满桶 10，请求被拒（链上 B 不足），A 余额不变 -> 占用 0
    resp = _report(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0},
            {"id": "B", "capacity": 1, "refill_rate": 0},
        ],
        [{"id": "x", "timestamp": 0, "cost": 5, "chain": ["A", "B"]}],
        10,
    )
    win = resp.windows[0]
    assert win.buckets["A"].peak_occupancy == 0
    assert win.buckets["B"].peak_occupancy == 0


def test_peak_is_scoped_per_window():
    # 窗口 [0,10)：占用 8；窗口 [10,20)：占用 1
    resp = _report(
        [{"id": "A", "capacity": 10, "refill_rate": 1}],
        [
            {"id": "r1", "timestamp": 0, "cost": 8, "chain": ["A"]},
            {"id": "r2", "timestamp": 10, "cost": 1, "chain": ["A"]},
        ],
        10,
    )
    wins = _by_start(resp.windows)
    assert wins[0.0].buckets["A"].peak_occupancy == 8
    assert wins[10.0].buckets["A"].peak_occupancy == 1
    bucket = resp.buckets[0]
    assert bucket.peak_occupancy == 8
    assert (bucket.peak_window_start, bucket.peak_window_end) == (0.0, 10.0)


def test_peak_window_tie_keeps_earliest():
    # 速率 5/s：t=0 扣 5 -> 占用 5；t=2 时补回复满再扣 5 -> 占用仍 5，并列取最早窗
    resp = _report(
        [{"id": "A", "capacity": 100, "refill_rate": 5}],
        [
            {"id": "a", "timestamp": 0, "cost": 5, "chain": ["A"]},
            {"id": "b", "timestamp": 2, "cost": 5, "chain": ["A"]},
        ],
        1,
    )
    bucket = resp.buckets[0]
    assert bucket.peak_occupancy == 5
    assert bucket.peak_window_start == 0  # 并列取最早


# ---------- 窗口划分 ----------

def test_windows_half_open_boundary_goes_to_right_window():
    resp = _report(
        [{"id": "A", "capacity": 100, "refill_rate": 0}],
        [
            {"id": "r0", "timestamp": 1.999, "cost": 1, "chain": ["A"]},
            {"id": "r1", "timestamp": 2, "cost": 1, "chain": ["A"]},
            {"id": "r2", "timestamp": 4, "cost": 1, "chain": ["A"]},
        ],
        2,
    )
    assert [(w.start, w.end) for w in resp.windows] == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    wins = _by_start(resp.windows)
    assert wins[0.0].allowed_requests == 1
    assert wins[2.0].allowed_requests == 1
    assert wins[4.0].allowed_requests == 1


def test_negative_timestamps_align_against_origin():
    # t=-1 落入 [-2,0)，t=0 落入 [0,2)：半开区间相对时间原点对齐
    resp = _report(
        [{"id": "A", "capacity": 100, "refill_rate": 0}],
        [
            {"id": "neg", "timestamp": -1, "cost": 1, "chain": ["A"]},
            {"id": "zero", "timestamp": 0, "cost": 1, "chain": ["A"]},
        ],
        2,
    )
    assert [(w.start, w.end) for w in resp.windows] == [(-2.0, 0.0), (0.0, 2.0)]


def test_empty_windows_are_skipped():
    resp = _report(
        [{"id": "A", "capacity": 100, "refill_rate": 0}],
        [
            {"id": "a", "timestamp": 0, "cost": 1, "chain": ["A"]},
            {"id": "b", "timestamp": 5, "cost": 1, "chain": ["A"]},
        ],
        1,
    )
    # 窗口 1/2/3/4 无请求，直接跳过
    assert [w.start for w in resp.windows] == [0.0, 5.0]


def test_out_of_order_requests_windowed_chronologically():
    resp = _report(
        [{"id": "A", "capacity": 10, "refill_rate": 1, "initial": 0}],
        [
            {"id": "late", "timestamp": 5, "cost": 5, "chain": ["A"]},
            {"id": "early", "timestamp": 0, "cost": 1, "chain": ["A"]},
        ],
        10,
    )
    win = resp.windows[0]
    # 与 /simulate 一致：early 拒绝、late 放行
    assert win.allowed_requests == 1 and win.rejected_requests == 1
    assert win.buckets["A"].consumed == 5


# ---------- 最忙窗口 / 汇总 ----------

def test_busiest_window_is_max_total_consumed():
    resp = _report(
        [
            {"id": "A", "capacity": 100, "refill_rate": 0},
            {"id": "B", "capacity": 100, "refill_rate": 0},
        ],
        [
            {"id": "a", "timestamp": 0, "cost": 4, "chain": ["A", "B"]},  # 8
            {"id": "b", "timestamp": 2, "cost": 4, "chain": ["A"]},      # 4
        ],
        1,
    )
    assert resp.busiest_window.start == 0
    assert resp.busiest_window.total_consumed == 8


def test_busiest_window_tie_keeps_earliest():
    resp = _report(
        [{"id": "A", "capacity": 100, "refill_rate": 0}],
        [
            {"id": "a", "timestamp": 0, "cost": 5, "chain": ["A"]},
            {"id": "b", "timestamp": 2, "cost": 5, "chain": ["A"]},
        ],
        1,
    )
    assert resp.busiest_window.start == 0


def test_totals_match_simulate():
    buckets = [
        {"id": "A", "capacity": 10, "refill_rate": 1},
        {"id": "B", "capacity": 5, "refill_rate": 0},
    ]
    requests = [
        {"id": "r1", "timestamp": 0, "cost": 6, "chain": ["A", "B"]},
        {"id": "r2", "timestamp": 1, "cost": 1, "chain": ["A", "B"]},
        {"id": "r3", "timestamp": 3, "cost": 6, "chain": ["A", "B"]},
    ]
    sim = simulate(UsageReportRequest.model_validate(
        {"buckets": buckets, "requests": requests, "window_seconds": 2}
    ))
    report = _report(buckets, requests, 2)
    assert (report.allowed_count, report.rejected_count) == (
        sim.allowed_count,
        sim.rejected_count,
    )
    assert report.allowed_count + report.rejected_count == len(requests)
    # 每桶总消耗 = 放行次数 × cost（本用例 cost 恒定）
    for bucket in report.buckets:
        assert bucket.rejected_total + 1 <= len(requests)  # 至少 r2 放行


def test_bucket_windows_mapping_only_contains_touched_windows():
    # B 只在窗口 0 被触及；A 在两个窗口都被触及
    resp = _report(
        [
            {"id": "A", "capacity": 100, "refill_rate": 0},
            {"id": "B", "capacity": 100, "refill_rate": 0},
        ],
        [
            {"id": "x", "timestamp": 0, "cost": 1, "chain": ["A", "B"]},
            {"id": "y", "timestamp": 2, "cost": 1, "chain": ["A"]},
        ],
        1,
    )
    by_id = {b.id: b for b in resp.buckets}
    assert set(by_id["A"].windows) == {"0.0", "2.0"}
    assert set(by_id["B"].windows) == {"0.0"}


def test_unreferenced_bucket_reports_zero_and_first_peak_window():
    resp = _report(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0},
            {"id": "IDLE", "capacity": 3, "refill_rate": 0},
        ],
        [{"id": "r", "timestamp": 0, "cost": 2, "chain": ["A"]}],
        1,
    )
    by_id = {b.id: b for b in resp.buckets}
    idle = by_id["IDLE"]
    assert idle.consumed_total == 0
    assert idle.rejected_total == 0
    assert idle.peak_occupancy == 0
    assert (idle.peak_window_start, idle.peak_window_end) == (0.0, 1.0)
    assert idle.windows == {}
    assert [b.id for b in resp.buckets] == ["A", "IDLE"]


# ---------- 校验 ----------

def _validate(payload):
    from app.errors import RequestValidationErrors
    from pydantic import ValidationError

    with pytest.raises((RequestValidationErrors, ValidationError)):
        UsageReportRequest.model_validate(payload)


@pytest.mark.parametrize("bad", [0, -1, -0.5, 1e400, float("nan"), float("inf")])
def test_non_positive_or_non_finite_window_rejected(bad):
    _validate(
        {
            "window_seconds": bad,
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["A"]}],
        }
    )


@pytest.mark.parametrize("bad", ["1", None, True, [1]])
def test_wrong_type_window_rejected(bad):
    _validate(
        {
            "window_seconds": bad,
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["A"]}],
        }
    )


def test_inherited_simulate_validation_still_applies():
    # 未知桶引用等 /simulate 的跨字段校验必须原样生效
    from app.errors import RequestValidationErrors

    with pytest.raises(RequestValidationErrors) as exc:
        UsageReportRequest.model_validate(
            {
                "window_seconds": 1,
                "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
                "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["X"]}],
            }
        )
    assert any(e["type"] == "unknown_bucket_ref" for e in exc.value.errors)


def test_extra_field_rejected():
    _validate(
        {
            "window_seconds": 1,
            "unexpected": 1,
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["A"]}],
        }
    )


# ---------- HTTP ----------

def test_http_usage_report_ok():
    response = client.post(
        "/usage-report",
        json={
            "window_seconds": 2,
            "buckets": [
                {"id": "tenant_quota", "capacity": 100, "refill_rate": 10, "initial": 100},
                {"id": "api_qps", "capacity": 5, "refill_rate": 5},
                {"id": "user_quota", "capacity": 20, "refill_rate": 1, "initial": 3},
            ],
            "requests": [
                {"id": "req-1", "timestamp": 0, "cost": 4,
                 "chain": ["tenant_quota", "api_qps", "user_quota"]},
                {"id": "req-2", "timestamp": 0, "cost": 4,
                 "chain": ["tenant_quota", "api_qps"]},
                {"id": "req-3", "timestamp": 1, "cost": 4,
                 "chain": ["tenant_quota", "api_qps"]},
                {"id": "req-4", "timestamp": 2, "cost": 4,
                 "chain": ["tenant_quota", "api_qps", "user_quota"]},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert [w["start"] for w in data["windows"]] == [0.0, 2.0]
    w0, w1 = data["windows"]
    assert (w0["allowed_requests"], w0["rejected_requests"], w0["total_consumed"]) == (2, 1, 16.0)
    assert (w1["allowed_requests"], w1["rejected_requests"], w1["total_consumed"]) == (1, 0, 12.0)
    # user_quota 峰值占用 19 出现在窗口 [2,4)
    user = next(b for b in data["buckets"] if b["id"] == "user_quota")
    assert user["peak_occupancy"] == 19
    assert (user["peak_window_start"], user["peak_window_end"]) == (2.0, 4.0)
    # 全局最忙为窗口 0
    assert data["busiest_window"]["start"] == 0
    assert (data["allowed_count"], data["rejected_count"]) == (3, 1)


def test_http_window_zero_is_422_structured():
    response = client.post(
        "/usage-report",
        json={
            "window_seconds": 0,
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["A"]}],
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail
    assert any(e["loc"][-1] == "window_seconds" for e in detail)


def test_http_unknown_bucket_is_422_structured():
    response = client.post(
        "/usage-report",
        json={
            "window_seconds": 1,
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "chain": ["GHOST"]}],
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(e["type"] == "unknown_bucket_ref" for e in detail)


def test_http_root_lists_usage_report():
    body = client.get("/").json()
    assert body["usage_report"] == "/usage-report"
