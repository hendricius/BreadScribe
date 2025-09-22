from breadscribe.postprocess import remove_filler_words


def test_removes_bracketed_annotations():
	text = "[music] We will (laughs) now fold the dough."
	clean = remove_filler_words(text)
	assert "[music]" not in clean
	assert "(laughs)" not in clean
	assert "fold the dough" in clean


