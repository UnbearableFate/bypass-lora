"""LoRA adapters for the frozen, revision-pinned Qwen3 base model."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
           'gate_proj', 'up_proj', 'down_proj')


@dataclass(frozen=True)
class LoRAConfig:
    layers: tuple = (6, 13, 20, 27)
    rank: int = 10
    alpha: float = None
    model_revision: str = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'

    def __post_init__(self):
        if self.rank < 1 or not self.layers:
            raise ValueError('rank, alpha and layers must be positive')
        if self.alpha is None:
            object.__setattr__(self, 'alpha', float(self.rank))
        if self.alpha <= 0:
            raise ValueError('rank, alpha and layers must be positive')
        if len(set(self.layers)) != len(self.layers) or min(self.layers) < 0:
            raise ValueError('layers must be distinct nonnegative indices')


class LoRALinear(nn.Module):
    """A frozen linear layer plus a FP32, zero-exit LoRA update."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('LoRA targets must be nn.Linear modules')
        self.base = base
        self.base.requires_grad_(False)
        self.scale = float(alpha) / rank
        device = base.weight.device
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5 ** .5)

    def forward(self, x):
        output = self.base(x)
        update = F.linear(F.linear(x.float(), self.A), self.B) * self.scale
        return output + update.to(output.dtype)


def _get_parent(root, path):
    parts = path.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class LoRAModel(nn.Module):
    """Attach LoRA to Q/K/V/O and gate/up/down projections in selected layers."""
    target_names = TARGETS

    def __init__(self, model, config=None):
        super().__init__()
        self.model = model
        self.config = config or LoRAConfig()
        layers = self.model.model.layers
        if max(self.config.layers) >= len(layers):
            raise ValueError('configured layer index exceeds model depth')
        self.model.requires_grad_(False)
        self._targets = {}
        for index in self.config.layers:
            layer = layers[index]
            paths = [f'{"self_attn" if name in TARGETS[:4] else "mlp"}.{name}'
                     for name in self.target_names]
            for path in paths:
                parent, name = _get_parent(layer, path)
                base = getattr(parent, name)
                wrapped = LoRALinear(base, self.config.rank, self.config.alpha)
                setattr(parent, name, wrapped)
                self._targets[(index, path)] = wrapped

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def adapter_parameters(self):
        for adapter in self._targets.values():
            yield adapter.A
            yield adapter.B

    def export_adapter(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'config.json').write_text(json.dumps(asdict(self.config), indent=2) + '\n')
        state = {f'{i}.{path}.{name}': getattr(adapter, name).detach().cpu()
                 for (i, path), adapter in self._targets.items() for name in ('A', 'B')}
        torch.save(state, directory / 'adapter.pt')

    @classmethod
    def load_adapter(cls, model, directory):
        directory = Path(directory)
        data = json.loads((directory / 'config.json').read_text())
        data['layers'] = tuple(data['layers'])
        result = cls(model, LoRAConfig(**data))
        state = torch.load(directory / 'adapter.pt', map_location='cpu', weights_only=True)
        expected = {f'{i}.{path}.{name}' for i, path in result._targets
                    for name in ('A', 'B')}
        if set(state) != expected:
            raise ValueError('adapter checkpoint targets do not match its config')
        with torch.no_grad():
            for (i, path), adapter in result._targets.items():
                for name in ('A', 'B'):
                    getattr(adapter, name).copy_(state[f'{i}.{path}.{name}'])
        return result


class UpLoRAModel(LoRAModel):
    """Same zero-exit LoRA, restricted to up_proj in the configured layers."""
    target_names = ('up_proj',)

