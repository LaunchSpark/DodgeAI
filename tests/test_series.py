from dodge.series import BASELINE_SAMPLES, RunSeries


def test_capacity_is_never_exceeded():
    series = RunSeries(capacity=16)
    for i in range(5000):
        series.append(i)
    assert len(series) <= 16


def test_series_always_spans_every_sample():
    """interval * points (+ the open bucket) must account for every append."""
    series = RunSeries(capacity=8)
    for i in range(1000):
        series.append(i)
        covered = len(series) * series.interval + len(series._bucket)
        assert covered == series.count


def test_interval_doubles_on_compaction():
    series = RunSeries(capacity=8)
    series.extend(range(8))
    assert series.interval == 2
    series.extend(range(8))
    assert series.interval == 4


def test_merging_preserves_order_and_approximate_values():
    series = RunSeries(capacity=8)
    series.extend(range(64))
    values = series.as_list()
    assert values == sorted(values), "a rising input must stay rising after merging"
    assert values[0] < 8 and values[-1] > 55


def test_trend_is_monotonic_for_monotonic_input():
    series = RunSeries(capacity=32)
    series.extend(range(300))
    assert series.trend == sorted(series.trend)


def test_trend_smooths_noise_while_points_are_still_raw():
    """Before any compaction, the trend must swing less than the raw values."""
    noisy = RunSeries(capacity=512)
    for i in range(200):
        noisy.append(50 + (30 if i % 2 else -30))
    assert noisy.interval == 1, "this test is about the pre-compaction case"
    raw_swing = max(noisy.values) - min(noisy.values)
    trend_swing = max(noisy.trend) - min(noisy.trend)
    assert trend_swing < raw_swing


def test_compaction_averages_within_buckets():
    """Stored points are bucket means, so compaction smooths the raw line too.

    This is inherent to downsampling and is why the chart keeps a separate
    trend line: after a long run the 'raw' polyline is no longer spiky.
    """
    series = RunSeries(capacity=8)
    for i in range(200):
        series.append(50 + (30 if i % 2 else -30))
    assert series.interval > 1
    assert all(abs(value - 50) < 1e-9 for value in series.values)


def test_baseline_uses_the_opening_samples_only():
    series = RunSeries(capacity=8)
    series.extend([10.0] * BASELINE_SAMPLES)
    series.extend([500.0] * 200)
    assert series.baseline == 10.0


def test_delta_reports_improvement_against_the_baseline():
    series = RunSeries(capacity=32)
    series.extend([0.0] * BASELINE_SAMPLES)
    series.extend([100.0] * 400)
    assert series.baseline == 0.0
    assert series.delta > 90


def test_bounds_cover_every_drawn_point():
    series = RunSeries(capacity=32)
    series.extend(range(500))
    low, high = series.bounds()
    assert low <= min(series.values) and high >= max(series.values)
    assert low <= min(series.trend) and high >= max(series.trend)


def test_bounds_are_never_zero_width():
    series = RunSeries(capacity=8)
    series.extend([7.0] * 40)
    low, high = series.bounds()
    assert high > low


def test_empty_series_is_safe_to_read():
    series = RunSeries()
    assert len(series) == 0
    assert not series
    assert series.latest == 0.0
    assert series.baseline == 0.0
    assert series.delta == 0.0
    low, high = series.bounds()
    assert high > low


def test_latest_reports_the_newest_raw_sample():
    series = RunSeries(capacity=8)
    series.extend(range(100))
    assert series.latest == 99.0


def test_odd_capacity_is_rejected():
    """Pair-merging requires an even capacity or a point would be dropped."""
    try:
        RunSeries(capacity=15)
    except ValueError:
        return
    raise AssertionError("expected ValueError for an odd capacity")


def test_copy_is_detached_from_the_original():
    """The dashboard renders from a copy so it need not hold the trainer's lock."""
    from dodge.series import RunSeries

    series = RunSeries()
    series.extend([1.0, 2.0, 3.0])
    clone = series.copy()
    series.extend([100.0] * 20)

    assert clone.as_list() == [1.0, 2.0, 3.0]
    assert clone.count == 3
    assert clone.latest == 3.0
    # A snapshot of the run so far, not a live view: the copy's baseline is
    # the mean of the three samples it actually holds.
    assert clone.baseline == 2.0
    assert len(series) > len(clone), "the original kept growing"


def test_copy_survives_compaction_of_the_original():
    from dodge.series import CAPACITY, RunSeries

    series = RunSeries()
    series.extend(range(CAPACITY))
    clone = series.copy()
    before, interval = clone.as_list(), clone.interval
    series.extend(range(CAPACITY * 4))

    assert clone.as_list() == before
    assert series.interval > interval, "the original compacted further"
    assert clone.interval == interval, "compaction must not reach the copy"
