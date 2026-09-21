"""核心模拟引擎：连续补充 + 链上原子扣减。"""
from __future__ import annotations

from dataclasses import dataclass

from app.models import EPS, BucketBalance, SimulateRequest, SimulateResponse, SimResult

# 输出精度，避免二进制浮点产生的长尾数
_ROUND_DIGITS = 9


def _round(value: float) -> float:
    rounded = round(value, _ROUND_DIGITS)
    # 规范化 -0.0
    return rounded if rounded != 0 else 0.0


@dataclass(frozen=True)
class BucketEvent:
    """单次请求判定时，链上某个桶的原始状态。

    before/after 均为未舍入的内部精度：拒绝时 after == before（原子性）。
    """

    bucket_id: str
    before: float
    after: float


@dataclass(frozen=True)
class RequestEvent:
    """单次请求按时间顺序处理后的判定结果（原始下标，非处理顺序）。"""

    index: int
    request_id: str
    timestamp: float
    allowed: bool
    rejected_by: str | None
    buckets: tuple[BucketEvent, ...]


@dataclass(frozen=True)
class EngineRun:
    """模拟引擎一次完整运行的原始轨迹。"""

    events: tuple[RequestEvent, ...]  # 按处理顺序（时间升序、同刻按输入顺序）
    final_balances: dict[str, float]
    t_start: float
    t_end: float


def run_engine(payload: SimulateRequest) -> EngineRun:
    """执行连续补充 + 链上原子扣减，返回未舍入的逐事件轨迹。

    /simulate 与 /usage-report 共用此核心，保证两者对同一请求序列的
    连续补充与原子判定完全一致。
    """
    if not payload.buckets:
        raise ValueError("buckets 不能为空")
    if not payload.requests:
        raise ValueError("requests 不能为空")

    # 模拟开始时刻 = 最早请求时间戳：所有桶在该时刻持有初始余额，
    # 之后按 (now - last) * rate 连续补充，上限为容量。时间戳因此可以
    # 使用任意一致的时间原点（Unix 时间、相对秒数均可）。
    t_start = min(request.timestamp for request in payload.requests)
    t_end = max(request.timestamp for request in payload.requests)

    capacity = {b.id: float(b.capacity) for b in payload.buckets}
    refill_rate = {b.id: float(b.refill_rate) for b in payload.buckets}
    balance = {
        b.id: float(b.capacity if b.initial is None else b.initial) for b in payload.buckets
    }
    last_time = {b.id: t_start for b in payload.buckets}

    def refill(bucket_id: str, now: float) -> float:
        """把某个桶补充到时刻 now，返回补充后的余额。"""
        elapsed = now - last_time[bucket_id]
        if elapsed > 0:
            balance[bucket_id] = min(
                capacity[bucket_id], balance[bucket_id] + refill_rate[bucket_id] * elapsed
            )
        last_time[bucket_id] = now
        return balance[bucket_id]

    # 内部按时间顺序处理；同一时刻按原始输入顺序处理。
    order = sorted(
        range(len(payload.requests)),
        key=lambda i: (payload.requests[i].timestamp, i),
    )

    events: list[RequestEvent] = []

    for i in order:
        request = payload.requests[i]
        now = request.timestamp

        # 先对链上所有桶做连续补充（无论最终是否放行，时间都会推进）
        refilled = {bucket_id: refill(bucket_id, now) for bucket_id in request.chain}

        # 链上第一个余额不足的桶即限制桶；必须全部充足才放行
        rejected_by: str | None = None
        for bucket_id in request.chain:
            if refilled[bucket_id] + EPS < request.cost:
                rejected_by = bucket_id
                break

        allowed = rejected_by is None
        bucket_events: list[BucketEvent] = []

        for bucket_id in request.chain:
            before = refilled[bucket_id]
            if allowed:
                balance[bucket_id] = before - request.cost
                after = balance[bucket_id]
            else:
                # 原子性：拒绝时链上所有桶保持不变
                after = before
            bucket_events.append(BucketEvent(bucket_id=bucket_id, before=before, after=after))

        events.append(
            RequestEvent(
                index=i,
                request_id=request.id,
                timestamp=now,
                allowed=allowed,
                rejected_by=rejected_by,
                buckets=tuple(bucket_events),
            )
        )

    # 把所有未再参与请求的桶补充到最后时刻，给出终态余额
    final_balances = {
        bucket_id: min(
            capacity[bucket_id],
            balance[bucket_id]
            + refill_rate[bucket_id] * max(0.0, t_end - last_time[bucket_id]),
        )
        for bucket_id in balance
    }

    return EngineRun(
        events=tuple(events),
        final_balances=final_balances,
        t_start=t_start,
        t_end=t_end,
    )


def simulate(payload: SimulateRequest) -> SimulateResponse:
    """按时间顺序模拟，返回逐项扣减视图（results 按原始数组顺序）。"""
    run = run_engine(payload)

    results_by_index: dict[int, SimResult] = {}
    for event in run.events:
        balances_view = {
            be.bucket_id: BucketBalance(before=_round(be.before), after=_round(be.after))
            for be in event.buckets
        }
        results_by_index[event.index] = SimResult(
            request_id=event.request_id,
            index=event.index,
            timestamp=_round(event.timestamp),
            allowed=event.allowed,
            rejected_by=event.rejected_by,
            balances=balances_view,
        )

    results = [results_by_index[i] for i in range(len(payload.requests))]
    allowed_count = sum(1 for r in results if r.allowed)

    return SimulateResponse(
        allowed_count=allowed_count,
        rejected_count=len(results) - allowed_count,
        results=results,
        final_balances={
            bucket_id: _round(value) for bucket_id, value in run.final_balances.items()
        },
    )
