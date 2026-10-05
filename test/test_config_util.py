from config import flatten


def test_flatten():
    nested = {"a": {"b": 1, "c": {"d": [2, 3]}}, "e": "x"}
    assert flatten(nested) == {"a.b": 1, "a.c.d": [2, 3], "e": "x"}
