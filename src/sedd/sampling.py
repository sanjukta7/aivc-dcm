import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Callable
from tqdm import tqdm

from .graph import Graph, AbsorbingGraph
from .noise import NoiseSchedule

Tensor = torch.Tensor


class Sampler(ABC):
    def __init__(
        self,
        model,
        graph: Graph,
        noise: NoiseSchedule,
        num_steps: int = 100,
        device: torch.device = None,
    ):

        self.model = model
        self.graph = graph
        self.noise = noise
        self.num_steps = num_steps
        self.device = device or next(model.parameters()).device

    @abstractmethod
    def step(
        self,
        x: Tensor,
        t: float,
        dt: float,
    ) -> Tensor:

        pass

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        mask_positions: Optional[Tensor] = None,
        show_progress: bool = True,
    ) -> Tensor:

        self.model.eval()
        x = x_init.clone().to(self.device)

        times = torch.linspace(1, 0, self.num_steps + 1, device=self.device)
        dt = -1.0 / self.num_steps

        iterator = range(self.num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling")

        for i in iterator:
            t = times[i]
            x = self.step(x, t, dt)

            if mask_positions is not None:
                x = torch.where(mask_positions, x, x_init)

        x = self.denoise(x)

        return x

    @torch.no_grad()
    def denoise(self, x: Tensor) -> Tensor:

        sigma = torch.tensor([0.01], device=self.device)
        score = self.model.score(x, sigma)

        mask_idx = self.graph.mask_index if hasattr(self.graph, 'mask_index') else -1
        is_masked = (x == mask_idx)

        if is_masked.any():
            probs = F.softmax(score[..., :-1], dim=-1)
            sampled = probs.argmax(dim=-1)
            x = torch.where(is_masked, sampled, x)

        return x


class EulerSampler(Sampler):
    def __init__(
        self,
        model,
        graph: Graph,
        noise: NoiseSchedule,
        num_steps: int = 100,
        device: torch.device = None,
        temperature: float = 1.0,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(model, graph, noise, num_steps, device)
        self.temperature = temperature
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

    def step(self, x: Tensor, t: float, dt: float) -> Tensor:
        t_tensor = torch.tensor([t], device=self.device)
        sigma = self.noise.total(t_tensor)
        dsigma = self.noise.rate(t_tensor) * (-dt)  # dt is negative

        # Use autocast for faster inference
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            score = self.model.score(x, sigma)  # [batch, seq, vocab]

        if isinstance(self.graph, AbsorbingGraph):
            return self._euler_step_absorbing(x, score, sigma, dsigma)
        else:
            return self._euler_step_general(x, score, sigma, dsigma)

    def _euler_step_absorbing(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:

        batch_size, seq_len = x.shape
        mask_idx = self.graph.mask_index

        is_masked = (x == mask_idx)

        if not is_masked.any():
            return x

        probs = F.softmax(score / self.temperature, dim=-1)

        p_stay = torch.exp(-dsigma)

        unmask_prob = (1 - p_stay) * is_masked.float()
        do_unmask = torch.rand_like(unmask_prob) < unmask_prob

        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(batch_size, seq_len)

        new_tokens = new_tokens.clamp(max=mask_idx - 1)

        x_new = torch.where(do_unmask, new_tokens, x)

        return x_new

    def _euler_step_general(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:

        probs = F.softmax(score / self.temperature, dim=-1)

        p_stay = torch.exp(-dsigma).item()

        stay_mask = torch.rand(x.shape, device=x.device) < p_stay

        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(x.shape)

        return torch.where(stay_mask, x, new_tokens)


class PerturbationEulerSampler(Sampler):
    def __init__(
        self,
        model,
        graph: Graph,
        noise: NoiseSchedule,
        num_steps: int = 100,
        device: torch.device = None,
        temperature: float = 1.0,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(model, graph, noise, num_steps, device)
        self.temperature = temperature
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        pert_labels: Tensor,
        mask_positions: Optional[Tensor] = None,
        show_progress: bool = True,
        cell_type_labels: Optional[Tensor] = None,
    ) -> Tensor:

        self.model.eval()
        x = x_init.clone().to(self.device)
        pert_labels = pert_labels.to(self.device)
        if cell_type_labels is not None:
            cell_type_labels = cell_type_labels.to(self.device)

        times = torch.linspace(1, 0, self.num_steps + 1, device=self.device)
        dt = -1.0 / self.num_steps

        iterator = range(self.num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling")

        for i in iterator:
            t = times[i]
            x = self.step(x, t, dt, pert_labels, cell_type_labels)

            if mask_positions is not None:
                x = torch.where(mask_positions, x, x_init)

        x = self.denoise(x, pert_labels, cell_type_labels)

        return x

    def step(
        self,
        x: Tensor,
        t: float,
        dt: float,
        pert_labels: Tensor,
        cell_type_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Single denoising step with perturbation and cell type conditioning."""
        t_tensor = torch.tensor([t], device=self.device)
        sigma = self.noise.total(t_tensor)
        dsigma = self.noise.rate(t_tensor) * (-dt)

        # Get score WITH perturbation and cell type conditioning (use autocast for faster inference)
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            score = self.model.score(x, sigma, pert_labels, cell_type_labels=cell_type_labels)

        if isinstance(self.graph, AbsorbingGraph):
            return self._euler_step_absorbing(x, score, sigma, dsigma)
        else:
            return self._euler_step_general(x, score, sigma, dsigma)

    def _euler_step_absorbing(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:
        batch_size, seq_len = x.shape
        mask_idx = self.graph.mask_index

        is_masked = (x == mask_idx)

        if not is_masked.any():
            return x

        probs = F.softmax(score / self.temperature, dim=-1)

        p_stay = torch.exp(-dsigma)

        unmask_prob = (1 - p_stay) * is_masked.float()
        do_unmask = torch.rand_like(unmask_prob) < unmask_prob

        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(batch_size, seq_len)

        new_tokens = new_tokens.clamp(max=mask_idx - 1)

        x_new = torch.where(do_unmask, new_tokens, x)

        return x_new

    def _euler_step_general(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:
        probs = F.softmax(score / self.temperature, dim=-1)

        p_stay = torch.exp(-dsigma).item()

        stay_mask = torch.rand(x.shape, device=x.device) < p_stay

        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(x.shape)

        return torch.where(stay_mask, x, new_tokens)

    @torch.no_grad()
    def denoise(self, x: Tensor, pert_labels: Tensor, cell_type_labels: Optional[Tensor] = None) -> Tensor:
        sigma = torch.tensor([0.01], device=self.device)
        
        # Use autocast for faster inference
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            score = self.model.score(x, sigma, pert_labels, cell_type_labels=cell_type_labels)

        mask_idx = self.graph.mask_index if hasattr(self.graph, 'mask_index') else -1
        is_masked = (x == mask_idx)

        if is_masked.any():
            probs = F.softmax(score[..., :-1], dim=-1)
            sampled = probs.argmax(dim=-1)
            x = torch.where(is_masked, sampled, x)

        return x


class AnalyticSampler(Sampler):
    def __init__(
        self,
        model,
        graph: Graph,
        noise: NoiseSchedule,
        num_steps: int = 100,
        device: torch.device = None,
        temperature: float = 1.0,
    ):
        super().__init__(model, graph, noise, num_steps, device)
        self.temperature = temperature

    def step(self, x: Tensor, t: float, dt: float) -> Tensor:
        t_tensor = torch.tensor([t], device=self.device)
        t_next = torch.tensor([t + dt], device=self.device)

        sigma = self.noise.total(t_tensor)
        sigma_next = self.noise.total(t_next)
        dsigma = sigma - sigma_next

        score = self.model.score(x, sigma)

        if isinstance(self.graph, AbsorbingGraph):
            return self._analytic_step_absorbing(x, score, sigma, dsigma)
        else:
            return self._analytic_step_general(x, score, sigma, dsigma)

    def _analytic_step_absorbing(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:

        batch_size, seq_len = x.shape
        mask_idx = self.graph.mask_index

        is_masked = (x == mask_idx)

        if not is_masked.any():
            return x

        staggered = score * torch.exp(-dsigma)
        probs = F.softmax(staggered / self.temperature, dim=-1)

        p_stay_masked = torch.exp(-dsigma).item()

        do_unmask = torch.rand_like(is_masked.float()) > p_stay_masked

        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(batch_size, seq_len)

        new_tokens = new_tokens.clamp(max=mask_idx - 1)

        should_unmask = is_masked & do_unmask
        x_new = torch.where(should_unmask, new_tokens, x)

        return x_new

    def _analytic_step_general(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:

        staggered = score * torch.exp(-dsigma)
        probs = F.softmax(staggered / self.temperature, dim=-1)

        p_stay = torch.exp(-dsigma).item()
        stay_mask = torch.rand(x.shape, device=x.device) < p_stay

        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(x.shape)

        return torch.where(stay_mask, x, new_tokens)


def get_sampler(
    name: str,
    model,
    graph: Graph,
    noise: NoiseSchedule,
    **kwargs
) -> Sampler:
    samplers = {
        "euler": EulerSampler,
        "analytic": AnalyticSampler,
        "perturbation_euler": PerturbationEulerSampler,
    }

    if name not in samplers:
        raise ValueError(f"Unknown sampler: {name}. Choose from {list(samplers.keys())}")

    return samplers[name](model, graph, noise, **kwargs)


@torch.no_grad()
def impute_masked(
    model,
    graph: Graph,
    noise: NoiseSchedule,
    x: Tensor,
    mask: Tensor,
    sampler: str = "euler",
    num_steps: int = 100,
    temperature: float = 1.0,
    show_progress: bool = True,
) -> Tensor:

    device = next(model.parameters()).device
    x = x.clone().to(device)
    mask = mask.to(device)

    mask_idx = graph.mask_index if hasattr(graph, 'mask_index') else graph.num_states - 1
    x[mask] = mask_idx

    s = get_sampler(
        sampler,
        model,
        graph,
        noise,
        num_steps=num_steps,
        temperature=temperature,
        device=device,
    )

    result = s.sample(x, mask_positions=mask, show_progress=show_progress)

    return result