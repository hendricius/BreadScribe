from breadscribe.audio_utils import choose_best_audio_stream


def test_choose_best_audio_stream_prefers_eng_and_higher_bitrate():
	streams = [
		{"index": 1, "channels": 1, "bit_rate": "64000", "tags": {"language": "deu"}},
		{"index": 2, "channels": 2, "bit_rate": "128000", "tags": {"language": "eng"}},
		{"index": 0, "channels": 2, "bit_rate": "96000"},
	]
	assert choose_best_audio_stream(streams) == 2


