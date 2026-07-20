"""Action selection and Double-DQN optimization utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .replay_buffer import ReplayBatch


@dataclass(frozen=True, slots=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 128
    replay_capacity: int = 200_000
    target_tau: float = 0.005
    gradient_clip_norm: float = 10.0


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def mask_illegal_actions(q_values: Tensor, legal_masks: Tensor) -> Tensor:
    if q_values.shape != legal_masks.shape:
        raise ValueError("Q-values and legal masks must have the same shape.")
    if q_values.ndim != 2 or q_values.shape[1] != 64:
        raise ValueError("Expected Q-values with shape [batch, 64].")
    return q_values.masked_fill(
        ~legal_masks.bool(),
        torch.finfo(q_values.dtype).min,
    )


def select_action(
    q_network: nn.Module,
    state: Tensor,
    legal_mask: Tensor,
    *,
    epsilon: float = 0.0,
    rng: random.Random | None = None,
) -> int:
    """Select an epsilon-greedy legal action for one encoded state."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon must be between 0 and 1.")

    legal = legal_mask.detach().to("cpu").bool().flatten()
    legal_indices = torch.nonzero(legal, as_tuple=False).flatten().tolist()
    if not legal_indices:
        raise ValueError("No legal action is available; the environment must pass.")

    chooser = rng if rng is not None else random
    if chooser.random() < epsilon:
        return int(chooser.choice(legal_indices))

    device = _module_device(q_network)
    observation = state.to(device)
    if observation.ndim == 3:
        observation = observation.unsqueeze(0)
    if observation.shape[0] != 1:
        raise ValueError("select_action expects exactly one state.")

    was_training = q_network.training
    q_network.eval()
    with torch.no_grad():
        q_values = q_network(observation)
        masked = mask_illegal_actions(
            q_values,
            legal.to(device).unsqueeze(0),
        )
        action = int(masked.argmax(dim=1).item())
    q_network.train(was_training)
    return action


@torch.no_grad()
def compute_double_dqn_targets(
    online_network: nn.Module,
    target_network: nn.Module,
    batch: ReplayBatch,
    *,
    gamma: float,
) -> Tensor:
    """Compute signed two-player Double-DQN targets.

    ``bootstrap_sign`` is -1 when the next state belongs to the opponent and
    +1 when an opponent pass gives the same player another move.
    """

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Gamma must be between 0 and 1.")

    nonterminal = ~batch.terminated
    has_legal_action = batch.next_legal_masks.bool().any(dim=1)
    if torch.any(nonterminal & ~has_legal_action):
        raise ValueError("Every non-terminal next state must contain a legal action.")

    online_q = online_network(batch.next_states)
    safe_masks = batch.next_legal_masks.bool().clone()
    safe_masks[batch.terminated, 0] = True
    next_actions = mask_illegal_actions(online_q, safe_masks).argmax(
        dim=1,
        keepdim=True,
    )

    target_q = target_network(batch.next_states)
    next_values = target_q.gather(1, next_actions).squeeze(1)
    next_values = torch.where(
        nonterminal,
        next_values,
        torch.zeros_like(next_values),
    )

    return batch.rewards + gamma * batch.bootstrap_signs * next_values


def optimize_double_dqn(
    online_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: ReplayBatch,
    *,
    gamma: float,
    gradient_clip_norm: float = 10.0,
) -> float:
    predicted = online_network(batch.states).gather(
        1,
        batch.actions.unsqueeze(1),
    ).squeeze(1)
    targets = compute_double_dqn_targets(
        online_network,
        target_network,
        batch,
        gamma=gamma,
    )

    loss = F.smooth_l1_loss(predicted, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        online_network.parameters(),
        gradient_clip_norm,
    )
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def hard_update(target_network: nn.Module, online_network: nn.Module) -> None:
    target_network.load_state_dict(online_network.state_dict())


@torch.no_grad()
def soft_update(
    target_network: nn.Module,
    online_network: nn.Module,
    *,
    tau: float,
) -> None:
    if not 0.0 <= tau <= 1.0:
        raise ValueError("Tau must be between 0 and 1.")
    for target_parameter, online_parameter in zip(
        target_network.parameters(),
        online_network.parameters(),
        strict=True,
    ):
        target_parameter.lerp_(online_parameter, tau)
