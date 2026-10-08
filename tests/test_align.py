from app.models import Segment
from app.pipeline.align import plan
from app.pipeline.output import srt


def seg(i, start, end):
    return Segment(id=i, start=start, end=end, fa="x", en=f"line {i}")


def test_short_clip_keeps_original_start_and_speed():
    p = plan([seg(0, 1.0, 3.0), seg(1, 5.0, 7.0)], [1.5, 1.5], total=10, max_speedup=1.35)
    assert [(x.start, x.speed) for x in p] == [(1.0, 1.0), (5.0, 1.0)]


def test_clip_may_use_pause_before_next_line():
    # Slot is 2s but there are 4s until the next line, so a 3.5s clip needs no speedup.
    p = plan([seg(0, 1.0, 3.0), seg(1, 5.0, 7.0)], [3.5, 1.0], total=10, max_speedup=1.35)
    assert p[0].speed == 1.0


def test_long_clip_is_sped_up_within_limit():
    p = plan([seg(0, 0.0, 2.0), seg(1, 2.0, 4.0)], [2.4, 1.0], total=10, max_speedup=1.35)
    assert p[0].speed == 1.2
    assert p[1].start == 2.0


def test_overflow_pushes_next_line_instead_of_overlapping():
    p = plan([seg(0, 0.0, 2.0), seg(1, 2.0, 4.0), seg(2, 8.0, 9.0)], [4.0, 1.0, 1.0], total=10, max_speedup=1.35)
    assert p[0].speed == 1.35
    end0 = p[0].start + 4.0 / p[0].speed
    assert p[1].start >= end0 - 1e-3
    assert p[2].start == 8.0  # back in sync after the pause


def test_srt_format():
    text = srt([seg(0, 0.0, 1.5), seg(1, 3661.25, 3662.0)])
    assert "1\n00:00:00,000 --> 00:00:01,500\nline 0\n" in text
    assert "01:01:01,250 --> 01:01:02,000" in text
