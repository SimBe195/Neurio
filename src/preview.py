"""
Videos of what a trained DreamerV3 agent's world model predicts. The agent plays episodes as in training, and from a
random point of each, the model starts from the state the agent had there, sees a quarter of a training sequence and
predicts the rest:

- open loop: from the actions the agent really took, next to the true frames and their difference
- dream: from actions of its own policy, without any true frames after the start

Usage, with the same run selection as test.py:
    uv run src/preview.py test_run_id=<run id> levels=single levels.level=1-2

The videos are written to previews/<run name>-<iteration>-<levels>-{open-loop,dream}.gif.
"""

import logging
from pathlib import Path

import hydra
import mlflow
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from omegaconf import DictConfig

from agents.dreamer import DreamerAgent
from agents.dreamer.agent import PREVIEW_SEQUENCES
from agents.dreamer.best_episodes import sequence
from agents.dreamer.preview import render_preview, save_gif
from evaluation import load_evaluation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Episodes to try for sequences of full length
MAX_EPISODES = 50


def play_episode(env: AsyncVectorEnv, agent: DreamerAgent) -> dict[str, np.ndarray]:
    """:return: Per step of one episode of the single worker: the frame, the action taken and the agent's state"""
    observations, _ = env.reset()
    agent.is_first[:] = True
    steps: dict[str, list] = {"frames": [], "actions": [], "deter": [], "stoch": []}
    while True:
        (action,) = agent.act(observations)
        steps["frames"].append(np.array(observations[0, -1]))
        steps["actions"].append(action)
        steps["deter"].append(agent.state.deter[0].half().cpu().numpy())
        steps["stoch"].append(agent.state.stoch[0].argmax(-1).byte().cpu().numpy())
        observations, _, terminated, truncated, _ = env.step(np.array([action]))
        if terminated[0] or truncated[0]:
            return {name: np.stack(values) for name, values in steps.items()}


def sample_sequences(
    env: AsyncVectorEnv, agent: DreamerAgent, length: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """:return: Sequences of the given length like those of Replay.sample, one from each long enough episode"""
    sequences = []
    for _ in range(MAX_EPISODES):
        episode = play_episode(env, agent)
        if len(episode["frames"]) >= length:
            start = int(rng.integers(len(episode["frames"]) - length + 1))
            sequences.append(sequence(episode, start, length))
        if len(sequences) == PREVIEW_SEQUENCES:
            break
    assert sequences, f"No episode lasted {length} steps."
    return {name: np.stack([sequence[name] for sequence in sequences]) for name in sequences[0]}


@hydra.main(version_base="1.3", config_path="../configs", config_name="neurio_config")
def main(cfg: DictConfig) -> None:
    with load_evaluation(cfg) as evaluation:
        agent = evaluation.agent
        assert isinstance(agent, DreamerAgent), "Previews need a run of the DreamerV3 agent (agent=dreamer)."
        length = agent.config.batch_length
        batch = sample_sequences(evaluation.env, agent, length, np.random.default_rng())
    context = length // 4

    run_name = mlflow.get_run(evaluation.run_id).info.run_name
    prefix = Path("previews") / f"{run_name}-{evaluation.iteration}-{evaluation.config.levels.name}"
    prefix.parent.mkdir(exist_ok=True)
    for name, dream in (("open-loop", False), ("dream", True)):
        predicted = agent.predict(batch, context, dream=dream)
        path = prefix.with_name(f"{prefix.name}-{name}.gif")
        save_gif(render_preview(None if dream else batch["frames"], predicted, context), path)
        log.info(f"Wrote {path}.")


if __name__ == "__main__":
    main()
