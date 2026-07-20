"""Small export-friendly dueling Q-network for 8x8 Othello."""

from __future__ import annotations

from torch import Tensor, nn

from .encoding import OBSERVATION_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.activation(self.conv1(inputs))
        residual = self.conv2(residual)
        return self.activation(inputs + residual)


class DuelingQNetwork(nn.Module):
    """Compact CNN with one Q-value per board square."""

    def __init__(
        self,
        *,
        input_channels: int = OBSERVATION_CHANNELS,
        channels: int = 32,
        residual_blocks: int = 3,
        value_channels: int = 8,
    ) -> None:
        super().__init__()
        if channels <= 0 or residual_blocks < 0 or value_channels <= 0:
            raise ValueError("Network dimensions must be positive.")

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.body = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(residual_blocks))
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_channels, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(value_channels, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Flatten(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        if observations.ndim != 4:
            raise ValueError("Expected observations with shape [batch, channels, 8, 8].")
        features = self.body(self.stem(observations))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
