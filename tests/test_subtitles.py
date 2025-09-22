from breadscribe.types import Segment
from breadscribe.subtitles import segments_to_srt, segments_to_vtt


def test_srt_formatting():
	segments = [
		Segment(index=1, start=0.0, end=1.234, text="Hello"),
		Segment(index=2, start=1.234, end=3.5, text="World"),
	]
	srt = segments_to_srt(segments)
	assert "00:00:00,000 --> 00:00:01,234" in srt
	assert "00:00:01,234 --> 00:00:03,500" in srt
	assert srt.strip().endswith("World")


def test_vtt_formatting():
	segments = [
		Segment(index=1, start=0.0, end=1.234, text="Hello"),
		Segment(index=2, start=1.234, end=3.5, text="World"),
	]
	vtt = segments_to_vtt(segments)
	assert vtt.startswith("WEBVTT\n\n")
	assert "00:00:00.000 --> 00:00:01.234" in vtt
	assert "00:00:01.234 --> 00:00:03.500" in vtt
	assert vtt.strip().endswith("World")



