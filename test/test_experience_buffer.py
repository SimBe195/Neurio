import torch

from agents.experience_buffer import ExperienceBuffer

CPU = torch.device("cpu")


def _step(buffer: ExperienceBuffer, actions, dones) -> None:
    buffer.buffer_states(torch.zeros(len(actions), 1, 2, 2))
    buffer.buffer_actions(torch.tensor(actions, dtype=torch.int64))
    buffer.buffer_values(torch.zeros(len(actions)))
    buffer.buffer_log_probs(torch.zeros(len(actions)))
    buffer.buffer_rewards(torch.zeros(len(actions)))
    buffer.buffer_dones(torch.tensor(dones))


def test_prev_actions_are_shifted_actions():
    buffer = ExperienceBuffer(num_workers=2, device=CPU, size=10)
    _step(buffer, [1, 2], [False, False])
    _step(buffer, [3, 4], [False, False])

    assert buffer.get_prev_action_buffer().tolist() == [[0, 0], [1, 2]]
    assert buffer.get_action_buffer().tolist() == [[1, 2], [3, 4]]
    assert buffer.get_last_actions().tolist() == [3, 4]


def test_prev_action_is_reset_after_done():
    buffer = ExperienceBuffer(num_workers=2, device=CPU, size=10)
    _step(buffer, [1, 2], [True, False])
    _step(buffer, [3, 4], [False, True])

    assert buffer.get_prev_action_buffer().tolist() == [[0, 0], [0, 2]]
    assert buffer.get_last_actions().tolist() == [3, 0]


def test_buffers_stay_aligned_when_full():
    buffer = ExperienceBuffer(num_workers=1, device=CPU, size=3)
    for action in range(1, 6):
        _step(buffer, [action], [False])

    assert buffer.get_action_buffer().flatten().tolist() == [3, 4, 5]
    assert buffer.get_prev_action_buffer().flatten().tolist() == [2, 3, 4]
    assert buffer.get_state_buffer().shape[0] == 3
    assert buffer.get_reward_buffer().shape[0] == 3


def test_unbounded_buffer():
    buffer = ExperienceBuffer(num_workers=1, device=CPU, size=None)
    for action in range(1, 6):
        _step(buffer, [action], [False])

    assert buffer.get_action_buffer().flatten().tolist() == [1, 2, 3, 4, 5]
    assert buffer.get_prev_action_buffer().flatten().tolist() == [0, 1, 2, 3, 4]


def test_reset_keeps_last_action():
    buffer = ExperienceBuffer(num_workers=1, device=CPU, size=10)
    _step(buffer, [7], [False])
    buffer.reset()

    assert buffer.get_last_actions().tolist() == [7]
    assert len(buffer.actions) == 0

    buffer.reset(forget_prev_action=True)
    assert buffer.get_last_actions().tolist() == [0]
