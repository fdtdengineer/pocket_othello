"""Compact Dueling Double-DQN building blocks."""

from .agent import (
    DQNConfig,
    compute_double_dqn_targets,
    hard_update,
    mask_illegal_actions,
    optimize_double_dqn,
    select_action,
    soft_update,
)
from .encoding import OBSERVATION_CHANNELS, encode_batch, encode_observation
from .model import DuelingQNetwork, ResidualBlock, count_trainable_parameters
from .replay_buffer import ReplayBatch, ReplayBuffer, Transition
from .symmetry import (
    AllSymmetriesDataset,
    SYMMETRY_COUNT,
    transform_action,
    transform_batch,
    transform_observation,
)
from .teacher_data import TeacherDataset, TeacherExample, load_teacher_examples

__all__ = [
    "AllSymmetriesDataset",
    "DQNConfig",
    "DuelingQNetwork",
    "OBSERVATION_CHANNELS",
    "ReplayBatch",
    "ReplayBuffer",
    "ResidualBlock",
    "SYMMETRY_COUNT",
    "TeacherDataset",
    "TeacherExample",
    "Transition",
    "compute_double_dqn_targets",
    "count_trainable_parameters",
    "encode_batch",
    "encode_observation",
    "hard_update",
    "load_teacher_examples",
    "mask_illegal_actions",
    "optimize_double_dqn",
    "select_action",
    "soft_update",
    "transform_action",
    "transform_batch",
    "transform_observation",
]
