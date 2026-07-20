"""A small tensor replay buffer for off-policy DQN training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Transition:
    state: Tensor
    action: int
    reward: float
    next_state: Tensor
    next_legal_mask: Tensor
    terminated: bool
    bootstrap_sign: float


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    next_legal_masks: Tensor
    terminated: Tensor
    bootstrap_signs: Tensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = capacity
        self._storage: list[Transition] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._storage)

    def push(self, transition: Transition) -> None:
        stored = Transition(
            state=transition.state.detach().to("cpu").clone(),
            action=int(transition.action),
            reward=float(transition.reward),
            next_state=transition.next_state.detach().to("cpu").clone(),
            next_legal_mask=transition.next_legal_mask.detach().to("cpu").bool().clone(),
            terminated=bool(transition.terminated),
            bootstrap_sign=float(transition.bootstrap_sign),
        )
        if len(self._storage) < self.capacity:
            self._storage.append(stored)
        else:
            self._storage[self._next_index] = stored
        self._next_index = (self._next_index + 1) % self.capacity

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.push(transition)

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        rng: random.Random | None = None,
    ) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if batch_size > len(self._storage):
            raise ValueError("Cannot sample more transitions than are stored.")

        chooser = rng if rng is not None else random
        transitions = chooser.sample(self._storage, batch_size)
        return ReplayBatch(
            states=torch.stack([item.state for item in transitions]).to(device),
            actions=torch.tensor(
                [item.action for item in transitions],
                dtype=torch.long,
                device=device,
            ),
            rewards=torch.tensor(
                [item.reward for item in transitions],
                dtype=torch.float32,
                device=device,
            ),
            next_states=torch.stack([item.next_state for item in transitions]).to(device),
            next_legal_masks=torch.stack(
                [item.next_legal_mask for item in transitions]
            ).to(device),
            terminated=torch.tensor(
                [item.terminated for item in transitions],
                dtype=torch.bool,
                device=device,
            ),
            bootstrap_signs=torch.tensor(
                [item.bootstrap_sign for item in transitions],
                dtype=torch.float32,
                device=device,
            ),
        )
