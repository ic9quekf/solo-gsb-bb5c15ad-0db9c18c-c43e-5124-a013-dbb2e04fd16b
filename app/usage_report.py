"""配额使用分析：复用 /simulate 的连续补充与链式原子判定，再按窗口归集。"""
from __future__ import annotations

from app.models import (
    EPS,
    BucketUsage,
    BucketWindowUsage,
    UsageReportRequest,
    UsageReportResponse,
    WindowSummary,
)
from app.simulation import _round, run_engine


def usage_report(payload: UsageReportRequest) -> UsageReportResponse:
    """对请求序列执行与 /simulate 完全一致的模拟，并按定长半开窗口统计用量。

    窗口规则：
    - 窗口相对时间原点对齐：包含时间戳 t 的窗口为
      [k·window_seconds, (k+1)·window_seconds)，k = floor(t / window_seconds)；
    - 半开区间意味着恰好落在窗口边界的请求计入**右侧**（较晚）窗口，
      跨窗口的请求只计入其发生时刻所在窗口，不向其他窗口分摊；
    - 空窗口（无任何请求落入）不在 windows 中输出，直接跳过。

    每个桶在窗口内统计：
    - consumed：被放行请求从该桶扣减的令牌总量；
    - rejected：引用该桶且被原子拒绝的请求次数（即使限制桶是链上其他桶，
      该请求触及的每个桶都各计一次拒绝）；
    - peak_occupancy：窗口内各请求判定时刻的峰值占用，
      占用 = capacity − 判定后余额（拒绝时余额不变），即配额被占住的比例量。

    跨窗口不连续采样：桶在窗口边界之间的连续补充不产生事件，占用因此
    只按该窗口内实际发生的请求时刻统计。
    """
    run = run_engine(payload)

    window_seconds = float(payload.window_seconds)
    capacity = {b.id: float(b.capacity) for b in payload.buckets}

    # ---- 第一遍：按处理顺序（即时间顺序）逐事件归集到 (窗口 -> 桶) ----
    # 窗口以其起点标识；窗口首次出现时按处理顺序登记，天然时间升序。
    window_order: list[float] = []
    window_seen: set[float] = set()

    # win_start -> bucket_id -> 累加器 [consumed, rejected, peak_occupancy]
    per_window: dict[float, dict[str, list[float]]] = {}
    # win_start -> 窗口内 [allowed_requests, rejected_requests]
    win_requests: dict[float, list[int]] = {}

    for event in run.events:
        k = event.timestamp // window_seconds
        win_start = k * window_seconds
        if win_start not in window_seen:
            window_seen.add(win_start)
            window_order.append(win_start)
            win_requests[win_start] = [0, 0]
            per_window[win_start] = {}

        if event.allowed:
            win_requests[win_start][0] += 1
        else:
            win_requests[win_start][1] += 1

        bucket_stats = per_window[win_start]
        for be in event.buckets:
            stats = bucket_stats.get(be.bucket_id)
            if stats is None:
                stats = [0.0, 0, 0.0]
                bucket_stats[be.bucket_id] = stats

            if event.allowed:
                stats[0] += be.before - be.after  # 放行扣减量（内部精度即 cost）
            else:
                stats[1] += 1

            occupancy = max(0.0, capacity[be.bucket_id] - be.after)
            if occupancy > stats[2]:
                stats[2] = occupancy

    # ---- 组装窗口汇总 ----
    windows: list[WindowSummary] = []
    for win_start in window_order:
        win_end = win_start + window_seconds
        buckets_view: dict[str, BucketWindowUsage] = {}
        total_consumed = 0.0
        for bucket_id, (consumed, rejected, peak) in per_window[win_start].items():
            buckets_view[bucket_id] = BucketWindowUsage(
                consumed=_round(consumed),
                rejected=rejected,
                peak_occupancy=_round(peak),
            )
            total_consumed += consumed

        allowed_requests, rejected_requests = win_requests[win_start]
        windows.append(
            WindowSummary(
                start=_round(win_start),
                end=_round(win_end),
                allowed_requests=allowed_requests,
                rejected_requests=rejected_requests,
                total_consumed=_round(total_consumed),
                buckets=buckets_view,
            )
        )

    # 全局最忙窗口：放行扣减总量最大；并列取最早窗口（window_order 已升序）
    busiest = max(
        windows,
        key=lambda w: (
            w.total_consumed,
            # 负起点作为并列时的最早优先（max 需要反向）
            -w.start,
        ),
    )

    # ---- 第二遍：按桶跨窗口汇总 ----
    bucket_results: list[BucketUsage] = []
    for bucket in payload.buckets:
        bucket_id = bucket.id
        consumed_total = 0.0
        rejected_total = 0
        peak_total = 0.0
        peak_start: float | None = None
        peak_end: float | None = None
        bucket_windows: dict[str, BucketWindowUsage] = {}

        for window in windows:
            usage = window.buckets.get(bucket_id)
            if usage is None:
                continue
            bucket_windows[str(window.start)] = usage
            consumed_total += usage.consumed
            rejected_total += usage.rejected
            # 峰值并列时保留最早窗口：windows 已升序，仅在严格更大时更新
            if usage.peak_occupancy > peak_total + EPS:
                peak_total = usage.peak_occupancy
                peak_start = window.start
                peak_end = window.end

        # 从未被任何链引用的桶没有事件，峰值窗口取首个窗口（占用 0）
        if peak_start is None:
            peak_start = windows[0].start
            peak_end = windows[0].end

        bucket_results.append(
            BucketUsage(
                id=bucket_id,
                consumed_total=_round(consumed_total),
                rejected_total=rejected_total,
                peak_occupancy=_round(peak_total),
                peak_window_start=peak_start,
                peak_window_end=peak_end,
                windows=bucket_windows,
            )
        )

    allowed_count = sum(w.allowed_requests for w in windows)
    rejected_count = sum(w.rejected_requests for w in windows)

    return UsageReportResponse(
        window_seconds=_round(window_seconds),
        windows=windows,
        buckets=bucket_results,
        busiest_window=busiest,
        allowed_count=allowed_count,
        rejected_count=rejected_count,
    )
