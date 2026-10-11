"""Sensor-only, target-equivariant graph policy for non-contact observation.

The ten feature channels have fixed physical scales, independent of IDs or the
current population: relative position /100 m, relative velocity /20 m/s,
closing speed /20 m/s, time to reach the observation band /30 s, ZEM /100 m,
and reachability unchanged. Signed normalized channels are clipped to [-4,4];
time/ZEM to [0,4]. Unavailable targets retain their column but have zero P.
"""
from dataclasses import replace
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn

from estimator import Observation, estimate_pairs


FEATURE_NAMES = ('relative_x', 'relative_y', 'relative_z',
                 'relative_vx', 'relative_vy', 'relative_vz',
                 'closing_speed', 'time_to_band', 'zero_effort_miss', 'reachability')
FEATURE_SCALES = (100., 100., 100., 20., 20., 20., 20., 30., 100., 1.)
FORMAT_VERSION = 1
ARCHITECTURE = 'targetwise-edgeconv-2'


def sensor_pair_mask(obs: Observation) -> torch.Tensor:
    """Return (N,M) validity without changing the observed target ID order."""
    available = getattr(obs, 'target_available', None)
    if available is None:
        available = np.ones(len(obs.target_ids), dtype=bool)
    available = np.asarray(available)
    if available.shape != (len(obs.target_ids),) or available.dtype != np.bool_:
        raise ValueError('target_available must be a boolean vector aligned with target_ids')
    return torch.from_numpy(np.broadcast_to(available, (len(obs.chaser_ids), len(available))).copy())


def build_sensor_graph(obs: Observation, cfg):
    """Build (features[N,M,10], adjacency[N,N], estimates) from local tracks.

    The target columns exactly match obs.target_ids, including unavailable
    physical tracks. Pass sensor_pair_mask(obs) to mask those assignment logits,
    or explicitly filter an Observation copy before calling this function.
    Edges use only the noisy observed craft positions and broadcast_range;
    every sensed craft has a self-loop. No plant state or track ID is a feature.
    """
    estimates = estimate_pairs(obs.chaser_pos, obs.chaser_vel,
        obs.target_pos, obs.target_vel, r_dock=cfg.r_dock,
        sigma_base=cfg.sigma_base, zem_scale=cfg.zem_scale,
        ground_z=cfg.ground_z, min_separation=cfg.min_separation,
        perimeter_x=cfg.width)
    shape = (len(obs.chaser_ids), len(obs.target_ids))
    if estimates.distance.shape != shape:
        raise ValueError('sensor state rows and track IDs must agree')
    valid = sensor_pair_mask(obs).numpy()
    estimates = replace(estimates, probability=np.where(valid, estimates.probability, 0.))
    raw = np.concatenate((estimates.relative_pos, estimates.relative_vel,
        estimates.closing[..., None], estimates.t_go[..., None],
        estimates.zem[..., None], estimates.probability[..., None]), axis=-1)
    features = raw/np.asarray(FEATURE_SCALES)
    features[..., :7] = np.clip(features[..., :7], -4., 4.)
    features[..., 7:9] = np.clip(features[..., 7:9], 0., 4.)
    features[..., 9] = np.clip(features[..., 9], 0., 1.)
    if not np.isfinite(features).all():
        raise ValueError('sensor-derived graph features must be finite')
    positions = np.asarray(obs.chaser_pos, dtype=float)
    adjacency = np.linalg.norm(positions[:, None]-positions[None, :], axis=-1) <= cfg.broadcast_range
    np.fill_diagonal(adjacency, True)
    return torch.from_numpy(features.astype(np.float32)), torch.from_numpy(adjacency), estimates


def _valid_mask(logits: torch.Tensor, mask) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(logits, dtype=torch.bool)
    mask = torch.as_tensor(mask, device=logits.device)
    if mask.dtype != torch.bool or mask.shape != logits.shape:
        raise ValueError('pair mask must be boolean and match the pair matrix shape')
    return mask


def masked_softmax(logits: torch.Tensor, mask=None) -> torch.Tensor:
    """Softmax over targets; an entirely masked craft produces an all-zero row."""
    if not torch.is_tensor(logits) or logits.ndim < 2 or not torch.is_floating_point(logits):
        raise ValueError('logits must be a floating (...,N,M) tensor')
    if not torch.isfinite(logits).all():
        raise ValueError('logits must be finite')
    valid = _valid_mask(logits, mask)
    if logits.shape[-1] == 0:
        return torch.zeros_like(logits)
    values = logits.masked_fill(~valid, -torch.inf)
    maximum = values.amax(dim=-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    weights = torch.exp(values-maximum)
    return weights/weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)


def service_leakage_loss(logits: torch.Tensor, probability, mask=None) -> torch.Tensor:
    """Mean over batches of sum_j prod_i (1-softmax(logits)_ij * P_ij).

    This soft independent-event surrogate does not guarantee unique hard
    argmax assignments. A supplied mask excludes fully padded target columns
    from the sum; valid but unassigned targets still contribute one.
    """
    x = masked_softmax(logits, mask)
    probability = torch.as_tensor(probability, dtype=logits.dtype, device=logits.device)
    if probability.shape != logits.shape or not torch.isfinite(probability).all():
        raise ValueError('probability must be finite and match logits')
    if torch.any((probability < 0.) | (probability > 1.)):
        raise ValueError('probability must be in [0,1]')
    leakage = torch.prod(1.-x*probability, dim=-2)
    if mask is not None:
        leakage = leakage*_valid_mask(logits, mask).any(dim=-2)
    return leakage.sum(dim=-1).mean()


def _craft_context(pair_state, valid):
    weights = valid[..., None].to(pair_state.dtype)
    return (pair_state*weights).sum(dim=-2)/weights.sum(dim=-2).clamp_min(1.)


class _TargetEdgeConv(nn.Module):
    """Same-target messages from neighboring craft, with per-craft context."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.update = nn.Sequential(nn.Linear(3*hidden_dim, hidden_dim), nn.SiLU(),
                                    nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, state, adjacency, valid):
        # Axes are (..., receiver craft, sender craft, target, hidden).
        receiver = state.unsqueeze(-3).expand(*state.shape[:-3], state.shape[-3],
                                             state.shape[-3], state.shape[-2], state.shape[-1])
        sender = state.unsqueeze(-4).expand_as(receiver)
        edge_valid = (adjacency[..., None] & valid.unsqueeze(-2) & valid.unsqueeze(-3))
        messages = self.message(torch.cat((receiver, sender-receiver), dim=-1))
        weights = edge_valid[..., None].to(state.dtype)
        aggregate = (messages*weights).sum(dim=-3)/weights.sum(dim=-3).clamp_min(1.)
        context = _craft_context(state, valid).unsqueeze(-2).expand_as(state)
        updated = self.norm(state+self.update(torch.cat((state, aggregate, context), dim=-1)))
        return updated*valid[..., None]


class SwarmGNN(nn.Module):
    """Two distinct masked message-passing layers producing (...,N,M) logits.

    Shared pair encoders, same-target craft-edge messages and masked mean
    contexts preserve craft/target permutation equivariance. IDs are absent.
    The output is a learned local policy, not a one-to-one matching algorithm.
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim < 4:
            raise ValueError('hidden_dim must be an integer >=4')
        self.hidden_dim = hidden_dim
        self.pair_embedding = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden_dim), nn.SiLU())
        self.message_layers = nn.ModuleList([_TargetEdgeConv(hidden_dim), _TargetEdgeConv(hidden_dim)])
        self.head = nn.Sequential(nn.Linear(2*hidden_dim+len(FEATURE_NAMES), hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.checkpoint_metadata = None

    def forward(self, features, adjacency, mask=None):
        parameter = next(self.parameters())
        features = torch.as_tensor(features, dtype=parameter.dtype, device=parameter.device)
        adjacency = torch.as_tensor(adjacency, device=parameter.device)
        if features.ndim < 3 or features.shape[-1] != len(FEATURE_NAMES) or not torch.isfinite(features).all():
            raise ValueError('features must be finite (...,N,M,10)')
        n = features.shape[-3]
        if adjacency.shape != features.shape[:-3]+(n, n) or adjacency.dtype != torch.bool:
            raise ValueError('adjacency must be boolean (...,N,N) matching feature batches')
        valid = _valid_mask(features[..., 0], mask)
        state = self.pair_embedding(features)*valid[..., None]
        for layer in self.message_layers:
            state = layer(state, adjacency, valid)
        context = _craft_context(state, valid).unsqueeze(-2).expand_as(state)
        logits = self.head(torch.cat((features, state, context), dim=-1)).squeeze(-1)
        # Keep outputs finite for checkpoint/inference checks. masked_softmax
        # gives invalid rows zero weight; direct argmax still needs a mask.
        return logits.masked_fill(~valid, 0.)


def _plain_metadata(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain_metadata(item) for key, item in value.items()}
    raise ValueError('checkpoint metadata must contain only finite JSON-compatible values')


def save_checkpoint(model: SwarmGNN, path, *, trained_epochs: int = 0,
                    metrics: dict | None = None, metadata: dict | None = None) -> Path:
    """Save versioned weights and plain training metadata atomically on CPU."""
    if not isinstance(model, SwarmGNN):
        raise ValueError('model must be SwarmGNN')
    if not isinstance(trained_epochs, int) or isinstance(trained_epochs, bool) or trained_epochs < 0:
        raise ValueError('trained_epochs must be a nonnegative integer')
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('refusing to save nonfinite model weights')
    checkpoint = dict(format_version=FORMAT_VERSION, architecture=ARCHITECTURE,
        feature_names=list(FEATURE_NAMES), feature_scales=list(FEATURE_SCALES),
        hidden_dim=model.hidden_dim, trained_epochs=trained_epochs,
        metrics=_plain_metadata(metrics or {}), metadata=_plain_metadata(metadata or {}), state_dict=state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_policy(path) -> SwarmGNN:
    """Load validated CPU weights in eval mode, never silently invent a policy."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'GNN checkpoint missing: {path}. Run python train_gnn.py --output {path}')
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise ValueError('GNN checkpoint must be a versioned dictionary')
    expected = {'format_version': FORMAT_VERSION, 'architecture': ARCHITECTURE,
                'feature_names': list(FEATURE_NAMES), 'feature_scales': list(FEATURE_SCALES)}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f'incompatible GNN checkpoint {key}')
    epochs = checkpoint.get('trained_epochs')
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 0:
        raise ValueError('invalid checkpoint trained_epochs')
    model = SwarmGNN(checkpoint.get('hidden_dim'))
    state = checkpoint.get('state_dict')
    template = model.state_dict()
    if not isinstance(state, dict) or state.keys() != template.keys():
        raise ValueError('checkpoint state_dict keys do not match architecture')
    for key, expected_tensor in template.items():
        value = state[key]
        if (not torch.is_tensor(value) or value.shape != expected_tensor.shape or
                value.dtype != expected_tensor.dtype or not torch.isfinite(value).all()):
            raise ValueError(f'invalid checkpoint tensor: {key}')
    model.load_state_dict(state, strict=True)
    model.checkpoint_metadata = _plain_metadata({key: value for key, value in checkpoint.items()
                                               if key != 'state_dict'})
    model.eval()
    return model
