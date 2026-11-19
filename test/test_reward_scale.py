import torch

from agents.ppo.reward_scale import RewardScale


def test_scaled_returns_have_unit_std():
    torch.manual_seed(0)
    scale = RewardScale(num_workers=8, gamma=0.9, device=torch.device("cpu"))
    no_dones = torch.zeros(8, dtype=torch.bool)
    for _ in range(2000):
        scale(50 + 20 * torch.randn(8), no_dones)
    # Returns of 500 ± 46 per worker, with the transient of the first steps
    assert 50 < scale.std < 250
    assert 0.2 < 50 / scale.std < 1.0


def test_returns_restart_after_episode_end():
    scale = RewardScale(num_workers=2, gamma=0.5, device=torch.device("cpu"))
    scale(torch.tensor([2.0, 2.0]), torch.tensor([True, False]))
    assert scale.returns.tolist() == [0.0, 2.0]
    scaled = scale(torch.tensor([4.0, 4.0]), torch.tensor([False, False]))
    assert scale.returns.tolist() == [4.0, 5.0]
    assert torch.allclose(scaled, torch.tensor([4.0, 4.0]) / scale.std)


def test_state_is_restored():
    scale = RewardScale(num_workers=2, gamma=0.9, device=torch.device("cpu"))
    for reward in range(10):
        scale(torch.tensor([reward, -reward], dtype=torch.float32), torch.tensor([False, False]))
    restored = RewardScale(num_workers=2, gamma=0.9, device=torch.device("cpu"))
    restored.load_state_dict(scale.state_dict())
    assert restored.std == scale.std
