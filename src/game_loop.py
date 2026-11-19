from dataclasses import dataclass

import mlflow
import numpy as np
from gymnasium.vector import AsyncVectorEnv

from agents import Agent, StepOutcome
from environment import Powerup
from exploration import NoveltyBonus, Position, UntriedActions
from level_schedulers import LevelScheduler
from powerups import StartPowerups


@dataclass
class EpisodeStats:
    reward: float
    length: int
    level: str
    # Where the episode ended and how far it got, see environment.ProgressShaping
    x_pos: int
    distance: int
    # The furthest it got, e.g. before 8-4's loop sent Mario back
    max_distance: int
    flag_get: bool
    # Mario's power-up at the start and the end of the episode
    start_powerup: Powerup = Powerup.SMALL
    powerup: Powerup = Powerup.SMALL


class GameLoop:
    def __init__(
        self,
        environment: AsyncVectorEnv,
        agent: Agent,
        level_scheduler: LevelScheduler | None = None,
        greedy: bool = False,
        novelty_bonus: NoveltyBonus | None = None,
        start_powerups: StartPowerups | None = None,
        untried_actions: UntriedActions | None = None,
    ) -> None:
        """
        :param greedy: Whether the agent takes its most likely actions, for evaluation
        :param novelty_bonus: Exploration bonus for the agent to learn from; episode rewards stay the game's own
        :param start_powerups: Statistics from which the levels' start power-ups are sampled, else Mario starts small
        :param untried_actions: Which actions have been taken in each place, to try each one before walls while training
        """
        self.env = environment
        self.agent = agent
        self.level_scheduler = level_scheduler
        self.greedy = greedy
        self.novelty_bonus = novelty_bonus
        self.start_powerups = start_powerups
        self.untried_actions = untried_actions if not greedy else None

        self.iter_episodes: list[EpisodeStats] = []
        self.iter_novelty_bonus = 0.0
        self.iter_untried_actions = 0
        self.total_iters_finished = 0
        self.total_env_steps = 0
        self.restart()

    def restart(self, seed: int | None = None) -> None:
        """
        Start new episodes on all workers, e.g. after changing the level distribution.

        :param seed: Seed of the environments' randomness, e.g. of the random start frames
        """
        self.env.reset_async(seed=seed)
        self.current_states, self.current_infos = self.env.reset_wait()
        self.max_distances = np.zeros(self.env.num_envs, dtype=np.int64)

    @staticmethod
    def _positions(infos: list[dict]) -> list[Position]:
        """Where each worker's Mario is, from the vector environment's infos of each worker."""
        keys = ("world", "stage", "area", "x_pos", "y_pos")
        return [tuple(int(info[key][w]) for key in keys) for w, info in enumerate(infos)]

    def run_single_step(self) -> None:
        untried = None
        if self.untried_actions is not None:
            positions = self._positions([self.current_infos] * self.env.num_envs)
            on_ground = [bool(flag) for flag in self.current_infos["on_ground"]]
            untried = self.untried_actions.untried(
                [str(level) for level in self.current_infos["level"]],
                [int(distance) for distance in self.current_infos["distance"]],
                positions,
                on_ground,
            )
            self.iter_untried_actions += int(untried.any(-1).sum())
        actions = self.agent.act(self.current_states, greedy=self.greedy, untried=untried)
        if self.untried_actions is not None:
            self.untried_actions.record(positions, on_ground, actions)
        self.env.step_async(np.array(actions))
        if not self.greedy:
            self.agent.train_while_playing()
        self.current_states, rewards, terminated, truncated, infos = self.env.step_wait()
        # For finished episodes, the first step of the next one
        self.current_infos = infos
        self.total_env_steps += len(actions)
        dones = np.logical_or(terminated, truncated)

        # Finished episodes report their last step in final_info, the regular info already belongs to the next episode
        step_infos = [infos["final_info"] if done else infos for done in dones]
        levels = [str(info["level"][w]) for w, info in enumerate(step_infos)]
        distances = [int(info["distance"][w]) for w, info in enumerate(step_infos)]
        self.max_distances = np.maximum(self.max_distances, distances)
        rarity = None
        if self.novelty_bonus is not None:
            visits = self.novelty_bonus.visit(self._positions(step_infos))
            bonus = self.novelty_bonus.bonus(visits)
            rewards = rewards + bonus
            rarity = 1 / visits
            self.iter_novelty_bonus += float(bonus.sum())

        self.agent.observe(
            StepOutcome(
                self.current_states,
                rewards,
                terminated,
                truncated,
                levels,
                infos.get("final_obs"),
                game_rewards=np.array([float(info["game_reward"][w]) for w, info in enumerate(step_infos)]),
                rarity=rarity,
                start_powerups=np.array([int(info["start_powerup"][w]) for w, info in enumerate(step_infos)]),
            )
        )
        for worker in np.flatnonzero(dones):
            self.record_episode(infos["final_info"], int(worker), levels[worker])
        self.max_distances[dones] = 0

    def record_episode(self, info: dict, worker: int, level: str) -> None:
        episode = EpisodeStats(
            reward=float(info["episode"]["r"][worker]),
            length=int(info["episode"]["l"][worker]),
            level=level,
            x_pos=int(info["x_pos"][worker]),
            distance=int(info["distance"][worker]),
            max_distance=int(self.max_distances[worker]),
            flag_get=bool(info["flag_get"][worker]),
            start_powerup=Powerup(int(info["start_powerup"][worker])),
            powerup=Powerup(int(info["powerup_level"][worker])),
        )
        if self.level_scheduler is not None:
            self.level_scheduler.record_episode(level, episode.flag_get, episode.distance)
        if self.start_powerups is not None:
            self.start_powerups.record_episode(level, episode.flag_get, episode.powerup)
        if self.untried_actions is not None:
            self.untried_actions.record_episode(level, episode.flag_get, episode.max_distance)
        self.iter_episodes.append(episode)

    def update_levels(self) -> None:
        if self.start_powerups is not None and self.start_powerups.update():
            self.env.call("set_start_powerups", self.start_powerups.distributions)
        if self.level_scheduler is None or not self.level_scheduler.update():
            return
        distribution = self.level_scheduler.distribution()
        assert distribution is not None
        self.env.call("set_level_distribution", *distribution)

    def run_train_iter(self, steps: int) -> dict[str, float]:
        """
        Collect experience for `steps` steps per worker, update the agent and log all metrics of the iteration in one
        batch, with the iteration, starting at 1, as step.

        :return: The logged metrics
        """
        self.iter_episodes = []
        self.iter_novelty_bonus = 0.0
        self.iter_untried_actions = 0
        for _ in range(steps):
            self.run_single_step()

        metrics = self.agent.update()
        self.update_levels()
        self.total_iters_finished += 1
        if self.level_scheduler is not None:
            metrics.update(self.level_scheduler.get_metrics())
        if self.start_powerups is not None:
            metrics.update(self.start_powerups.get_metrics())
        metrics["progress/env_steps"] = self.total_env_steps
        if self.novelty_bonus is not None:
            metrics["exploration/novelty_bonus"] = self.iter_novelty_bonus / (steps * self.env.num_envs)
            metrics["exploration/visited_cells"] = self.novelty_bonus.visited_cells
        if self.untried_actions is not None:
            self.untried_actions.end_iteration()
            metrics["exploration/untried_actions"] = self.iter_untried_actions / (steps * self.env.num_envs)
            metrics.update(self.untried_actions.get_metrics())
        mlflow.log_metrics(metrics, step=self.total_iters_finished)
        return metrics

    def state_dict(self) -> dict:
        """The progress of training, to resume it."""
        return {
            "agent": self.agent.state_dict(),
            "level_scheduler": self.level_scheduler.state_dict() if self.level_scheduler is not None else {},
            "novelty_bonus": self.novelty_bonus.state_dict() if self.novelty_bonus is not None else {},
            "start_powerups": self.start_powerups.state_dict() if self.start_powerups is not None else {},
            "untried_actions": self.untried_actions.state_dict() if self.untried_actions is not None else {},
            "total_iters_finished": self.total_iters_finished,
            "total_env_steps": self.total_env_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the progress of training and restart the episodes on the restored level distribution."""
        self.agent.load_state_dict(state["agent"])
        if self.level_scheduler is not None:
            self.level_scheduler.load_state_dict(state["level_scheduler"])
            if (distribution := self.level_scheduler.distribution()) is not None:
                self.env.call("set_level_distribution", *distribution)
        if self.novelty_bonus is not None:
            self.novelty_bonus.load_state_dict(state["novelty_bonus"])
        if self.start_powerups is not None:
            self.start_powerups.load_state_dict(state["start_powerups"])
            self.env.call("set_start_powerups", self.start_powerups.distributions)
        if self.untried_actions is not None:
            self.untried_actions.load_state_dict(state["untried_actions"])
        self.total_iters_finished = state["total_iters_finished"]
        self.total_env_steps = state["total_env_steps"]
        self.restart()

    def run_test_episode(self) -> EpisodeStats:
        """Play one episode without training. With render mode "human", the environment shows it in real time."""
        assert self.agent.num_workers == 1
        self.iter_episodes = []
        while not self.iter_episodes:
            self.run_single_step()
        return self.iter_episodes[0]
