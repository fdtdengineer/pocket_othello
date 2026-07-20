"""Mixed-opponent Double-DQN training for Othello."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .agent import optimize_double_dqn, soft_update
from .model import DuelingQNetwork
from .replay_buffer import ReplayBuffer
from .self_play import OpponentMode, play_episode


@dataclass(frozen=True, slots=True)
class ReinforcementConfig:
    episodes: int = 10_000
    batch_size: int = 128
    replay_capacity: int = 200_000
    warmup_transitions: int = 2_000
    updates_per_episode: int = 8
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_tau: float = 0.005
    gradient_clip_norm: float = 10.0
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 5_000
    self_play_weight: float = 0.50
    heuristic_weight: float = 0.30
    random_weight: float = 0.20
    checkpoint_every: int = 500
    seed: int = 20260720

    def validate(self) -> None:
        if self.episodes <= 0 or self.batch_size <= 0:
            raise ValueError("Episodes and batch size must be positive.")
        if self.replay_capacity < self.batch_size:
            raise ValueError("Replay capacity must be at least one batch.")
        if self.warmup_transitions < 0 or self.updates_per_episode < 0:
            raise ValueError("Warmup and update counts must be non-negative.")
        if not 0 <= self.gamma <= 1 or not 0 <= self.target_tau <= 1:
            raise ValueError("Gamma and target tau must be between 0 and 1.")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("Learning rate and gradient clipping must be positive.")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1.")
        if self.epsilon_decay_episodes <= 0:
            raise ValueError("Epsilon decay episodes must be positive.")
        if self.checkpoint_every <= 0:
            raise ValueError("Checkpoint interval must be positive.")
        weights = (
            self.self_play_weight,
            self.heuristic_weight,
            self.random_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Opponent weights must be non-negative with a positive sum.")


def epsilon_for_episode(config: ReinforcementConfig, episode: int) -> float:
    progress = min(max(episode - 1, 0) / config.epsilon_decay_episodes, 1.0)
    return config.epsilon_start + progress * (
        config.epsilon_end - config.epsilon_start
    )


def choose_opponent_mode(
    config: ReinforcementConfig,
    rng: random.Random,
) -> OpponentMode:
    modes: tuple[OpponentMode, ...] = ("self", "heuristic", "random")
    weights = (
        config.self_play_weight,
        config.heuristic_weight,
        config.random_weight,
    )
    return rng.choices(modes, weights=weights, k=1)[0]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_initial_network(
    checkpoint_path: str | Path | None,
    model_config: dict[str, int] | None,
) -> tuple[DuelingQNetwork, dict[str, int], dict[str, Any] | None]:
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        resolved_config = dict(checkpoint["model_config"])
    else:
        resolved_config = {
            "input_channels": 4,
            "channels": 32,
            "residual_blocks": 3,
            "value_channels": 8,
            **(model_config or {}),
        }

    model = DuelingQNetwork(**resolved_config)
    if checkpoint is not None:
        state = checkpoint.get("model_state_dict")
        if state is None:
            state = checkpoint.get("online_state_dict")
        if state is None:
            raise ValueError("Checkpoint does not contain model weights.")
        model.load_state_dict(state)
    return model, resolved_config, checkpoint


def _save_checkpoint(
    output_path: Path,
    *,
    online_network: DuelingQNetwork,
    target_network: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, int],
    config: ReinforcementConfig,
    episode: int,
    replay_size: int,
    updates: int,
    history: list[dict[str, Any]],
    initialized_from: str | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "stage": "reinforcement",
            "model_config": model_config,
            "online_state_dict": {
                name: value.detach().cpu()
                for name, value in online_network.state_dict().items()
            },
            "target_state_dict": {
                name: value.detach().cpu()
                for name, value in target_network.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": asdict(config),
            "episode": episode,
            "replay_size": replay_size,
            "updates": updates,
            "history": history,
            "initialized_from": initialized_from,
        },
        output_path,
    )


def train_reinforcement(
    *,
    output_path: str | Path,
    config: ReinforcementConfig = ReinforcementConfig(),
    init_checkpoint: str | Path | None = None,
    model_config: dict[str, int] | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    config.validate()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    rng = random.Random(config.seed)
    resolved_device = _resolve_device(device)

    online_network, resolved_model_config, _ = _load_initial_network(
        init_checkpoint,
        model_config,
    )
    online_network = online_network.to(resolved_device)
    target_network = copy.deepcopy(online_network).to(resolved_device)
    target_network.eval()
    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=config.learning_rate,
    )
    replay = ReplayBuffer(config.replay_capacity)
    output = Path(output_path)

    history: list[dict[str, Any]] = []
    updates = 0
    minimum_replay = max(config.batch_size, config.warmup_transitions)

    for episode in range(1, config.episodes + 1):
        epsilon = epsilon_for_episode(config, episode)
        mode = choose_opponent_mode(config, rng)
        result = play_episode(
            online_network,
            mode=mode,
            epsilon=epsilon,
            rng=rng,
        )
        replay.extend(result.transitions)

        losses: list[float] = []
        if len(replay) >= minimum_replay:
            for _ in range(config.updates_per_episode):
                batch = replay.sample(
                    config.batch_size,
                    device=resolved_device,
                    rng=rng,
                )
                loss = optimize_double_dqn(
                    online_network,
                    target_network,
                    optimizer,
                    batch,
                    gamma=config.gamma,
                    gradient_clip_norm=config.gradient_clip_norm,
                )
                soft_update(
                    target_network,
                    online_network,
                    tau=config.target_tau,
                )
                losses.append(loss)
                updates += 1

        history.append(
            {
                "episode": episode,
                "mode": mode,
                "epsilon": epsilon,
                "winner": result.winner,
                "black_discs": result.black_discs,
                "white_discs": result.white_discs,
                "moves": result.moves,
                "replay_size": len(replay),
                "mean_loss": sum(losses) / len(losses) if losses else None,
            }
        )

        if episode % config.checkpoint_every == 0 or episode == config.episodes:
            _save_checkpoint(
                output,
                online_network=online_network,
                target_network=target_network,
                optimizer=optimizer,
                model_config=resolved_model_config,
                config=config,
                episode=episode,
                replay_size=len(replay),
                updates=updates,
                history=history,
                initialized_from=str(init_checkpoint) if init_checkpoint else None,
            )

    mode_counts = {
        mode: sum(item["mode"] == mode for item in history)
        for mode in ("self", "heuristic", "random")
    }
    return {
        "checkpoint": str(output),
        "episodes": config.episodes,
        "updates": updates,
        "replay_size": len(replay),
        "mode_counts": mode_counts,
        "last_episode": history[-1],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="engine/python/checkpoints/reinforcement.pt",
    )
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--warmup-transitions", type=int, default=2_000)
    parser.add_argument("--updates-per-episode", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=5_000)
    parser.add_argument("--self-play-weight", type=float, default=0.50)
    parser.add_argument("--heuristic-weight", type=float, default=0.30)
    parser.add_argument("--random-weight", type=float, default=0.20)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--value-channels", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = train_reinforcement(
        output_path=arguments.output,
        init_checkpoint=arguments.init_checkpoint,
        config=ReinforcementConfig(
            episodes=arguments.episodes,
            batch_size=arguments.batch_size,
            replay_capacity=arguments.replay_capacity,
            warmup_transitions=arguments.warmup_transitions,
            updates_per_episode=arguments.updates_per_episode,
            gamma=arguments.gamma,
            learning_rate=arguments.learning_rate,
            target_tau=arguments.target_tau,
            gradient_clip_norm=arguments.gradient_clip_norm,
            epsilon_start=arguments.epsilon_start,
            epsilon_end=arguments.epsilon_end,
            epsilon_decay_episodes=arguments.epsilon_decay_episodes,
            self_play_weight=arguments.self_play_weight,
            heuristic_weight=arguments.heuristic_weight,
            random_weight=arguments.random_weight,
            checkpoint_every=arguments.checkpoint_every,
            seed=arguments.seed,
        ),
        model_config={
            "channels": arguments.channels,
            "residual_blocks": arguments.residual_blocks,
            "value_channels": arguments.value_channels,
        },
        device=arguments.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
