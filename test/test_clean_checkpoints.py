from clean_checkpoints import checkpoints_to_delete


def test_keeps_new_records_unevaluated_and_named_checkpoints():
    # 100, 300, 400 and 500 clear more levels than all before them, though 400 was deleted before. 450 is named by a
    # tag and 600 not evaluated yet.
    levels_cleared = {100: 2.0, 200: 2.0, 300: 5.0, 400: 9.0, 450: 3.0, 500: 12.0, 550: 12.0}
    iters = [100, 200, 300, 450, 500, 550, 600]
    assert checkpoints_to_delete(iters, levels_cleared, keep={450}) == [200, 550]
