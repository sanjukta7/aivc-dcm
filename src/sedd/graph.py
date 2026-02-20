import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple

Tensor = torch.Tensor


class Graph(ABC):
    def __init__(self, num_states: int):
        self.num_states = num_states

    @abstractmethod
    def rate(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        pass

    @abstractmethod
    def sample_transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        pass

    def sample_limiting(self, shape: Tuple[int, ...], device: torch.device) -> Tensor:
        pass


class AbsorbingGraph(Graph):
    def __init__(self, num_states: int):
        super().__init__(num_states)
        self.mask_index = num_states - 1

    def rate(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.shape
        device = x.device

        rate = torch.zeros(batch_size, seq_len, self.num_states, device=device)

        is_not_mask = (x != self.mask_index).float()
        rate[..., self.mask_index] = is_not_mask

        return rate

    def transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1)

        batch_size, seq_len = x.shape
        device = x.device

        p_stay = torch.exp(-sigma)  # [batch, 1, 1]

        probs = torch.zeros(batch_size, seq_len, self.num_states, device=device)

        probs.scatter_(2, x.unsqueeze(-1), p_stay.expand(batch_size, seq_len, 1))

        is_not_mask = (x != self.mask_index).float().unsqueeze(-1)
        p_to_mask = (1 - p_stay) * is_not_mask
        probs[..., self.mask_index:self.mask_index + 1] += p_to_mask

        is_mask = (x == self.mask_index).float().unsqueeze(-1)
        probs[..., self.mask_index:self.mask_index + 1] += is_mask * (1 - p_stay)

        return probs

    def sample_transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1)

        device = x.device
        batch_size, seq_len = x.shape

        p_stay = torch.exp(-sigma)  # [batch, 1]

        uniform = torch.rand(batch_size, seq_len, device=device)
        transition_to_mask = uniform > p_stay

        result = x.clone()
        result[transition_to_mask] = self.mask_index

        return result

    def sample_limiting(self, shape: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.full(shape, self.mask_index, dtype=torch.long, device=device)


class UniformGraph(Graph):
    def __init__(self, num_states: int):
        super().__init__(num_states)

    def rate(self, x: Tensor) -> Tensor:

        batch_size, seq_len = x.shape
        device = x.device

        rate_val = 1.0 / (self.num_states - 1)
        rate = torch.full(
            (batch_size, seq_len, self.num_states),
            rate_val,
            device=device
        )

        rate.scatter_(2, x.unsqueeze(-1), 0.0)

        return rate

    def transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1)

        batch_size, seq_len = x.shape
        device = x.device
        S = self.num_states

        decay = torch.exp(-sigma * S / (S - 1))

        p_uniform = 1.0 / S

        p_other = p_uniform * (1 - decay)

        p_stay = p_uniform + (1 - p_uniform) * decay

        probs = torch.full(
            (batch_size, seq_len, self.num_states),
            0.0,
            device=device
        )

        probs = probs + p_other

        extra_stay = p_stay - p_other
        probs.scatter_add_(2, x.unsqueeze(-1), extra_stay.expand(batch_size, seq_len, 1))

        return probs

    def sample_transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        probs = self.transition(x, sigma)
        return torch.multinomial(
            probs.view(-1, self.num_states),
            num_samples=1
        ).view(x.shape)

    def sample_limiting(self, shape: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randint(0, self.num_states, shape, device=device)


def get_graph(name: str, num_states: int) -> Graph:
    graphs = {
        "absorbing": AbsorbingGraph,
        "uniform": UniformGraph,
    }

    if name not in graphs:
        raise ValueError(f"Unknown graph: {name}. Choose from {list(graphs.keys())}")

    return graphs[name](num_states)