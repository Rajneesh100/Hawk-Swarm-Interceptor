"""Train the graph policy on randomized non-contact sensor snapshots.

Each epoch is one fresh randomized minibatch and one Adam update. A separate,
fixed-seed holdout is evaluated before and after training; it never supplies
training gradients or selects checkpoints. The objective is the soft service
leakage surrogate, not collisions, capture counts, or hard matching labels.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from estimator import Observation
from gnn_model import (SwarmGNN, build_sensor_graph, masked_softmax,
                       save_checkpoint, service_leakage_loss)
from sim_engine import Config


PROJECT_ROOT = Path(__file__).resolve().parent


def make_training_batch(rng: np.random.Generator, cfg: Config, batch_size: int):
    """Randomized local sensor graphs, padded to the configured population.

    Targets move forward with lateral/vertical variation at observation
    altitudes. Craft sample near and outside the safe observation annulus with
    matching or approaching velocity bounded by the configured speed limits.
    Random sensor/radio ranges and observation noise produce partial local
    graphs, including disconnected peers. No plant, collision events, or ID-based
    allocation labels are used. Padded rows/columns are marked invalid.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    n, m = cfg.n_chasers, cfg.n_targets
    features = torch.zeros((batch_size, n, m, 10), dtype=torch.float32)
    adjacency = torch.zeros((batch_size, n, n), dtype=torch.bool)
    probability = torch.zeros((batch_size, n, m), dtype=torch.float32)
    mask = torch.zeros((batch_size, n, m), dtype=torch.bool)
    for index in range(batch_size):
        target_pos = np.column_stack((rng.uniform(.15, .8, m)*cfg.width,
            rng.uniform(.15, .85, m)*cfg.depth,
            rng.uniform(cfg.altitude_min, cfg.altitude_max, m)))
        target_vel = np.column_stack((rng.uniform(cfg.target_speed_min, cfg.target_speed_max, m), rng.normal(0., .8, m),
                                      rng.normal(0., .3, m)))
        target_speed = np.linalg.norm(target_vel, axis=1, keepdims=True)
        target_vel *= np.clip(target_speed, cfg.target_speed_min, cfg.target_speed_max)/target_speed
        anchors = rng.integers(0, m, size=n)
        directions = rng.normal(size=(n, 3))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
        in_band = rng.random(n) < .5
        radii = np.where(in_band,
            rng.uniform(cfg.min_separation+.5, cfg.r_dock-.5, n),
            rng.uniform(cfg.r_dock+1., cfg.r_dock+45., n))
        chaser_pos = target_pos[anchors]+directions*radii[:, None]
        chaser_pos[:, 2] = np.maximum(cfg.ground_z+3., chaser_pos[:, 2])
        approach = np.where(in_band, rng.uniform(-.3, .3, n), rng.uniform(2., 8., n))
        chaser_vel = target_vel[anchors]-directions*approach[:, None]+rng.normal(0., .6, (n, 3))
        chaser_vel *= np.minimum(1., cfg.v_max/np.maximum(
            np.linalg.norm(chaser_vel, axis=1, keepdims=True), 1e-9))
        # Select a local observer before adding independent sensor noise. IDs
        # carry alignment only; the model receives no ID feature channel.
        observer = int(rng.integers(n))
        local_range = cfg.sensor_range*rng.uniform(.35, 1.)
        ci = np.flatnonzero(np.linalg.norm(chaser_pos-chaser_pos[observer], axis=1) <= local_range)
        ti = np.flatnonzero(np.linalg.norm(target_pos-chaser_pos[observer], axis=1) <= local_range)
        packet = Observation(0., observer, ci,
            chaser_pos[ci]+rng.normal(0., cfg.position_noise, (len(ci), 3)),
            chaser_vel[ci]+rng.normal(0., cfg.velocity_noise, (len(ci), 3)), ti,
            target_pos[ti]+rng.normal(0., cfg.position_noise, (len(ti), 3)),
            target_vel[ti]+rng.normal(0., cfg.velocity_noise, (len(ti), 3)))
        # Log-uniform radio range spends meaningful training time on sparse
        # and disconnected graphs instead of almost always using a full clique.
        graph_cfg = replace(cfg, broadcast_range=cfg.broadcast_range*
                            np.exp(rng.uniform(np.log(.05), 0.)))
        f, a, estimates = build_sensor_graph(packet, graph_cfg)
        ni, mi = len(ci), len(ti)
        features[index, :ni, :mi] = f
        adjacency[index, :ni, :ni] = a
        probability[index, :ni, :mi] = torch.as_tensor(estimates.probability, dtype=torch.float32)
        mask[index, :ni, :mi] = True
    return features, adjacency, probability, mask


def _hard_loss(choices, probability, mask):
    leakage = torch.prod(1.-choices*probability, dim=-2)
    return float((leakage*mask.any(dim=-2)).sum(dim=-1).mean())


def evaluate_policy(model, batch):
    """Report soft objective and hard argmax diagnostics on a fixed batch."""
    features, adjacency, probability, mask = batch
    model.eval()
    with torch.inference_mode():
        logits = model(features, adjacency, mask)
        soft_loss = float(service_leakage_loss(logits, probability, mask))
        soft = masked_softmax(logits, mask)
        hard = torch.zeros_like(soft)
        if soft.shape[-1]:
            indices = soft.argmax(dim=-1, keepdim=True)
            hard.scatter_(-1, indices, 1.)
            hard *= mask.any(dim=-1, keepdim=True)
        duplicates = (hard.sum(dim=-2)-1.).clamp_min(0.).sum(dim=-1).mean()
        return dict(soft_leakage_loss=soft_loss, hard_leakage_loss=_hard_loss(hard, probability, mask),
                    mean_hard_duplicate_choices=float(duplicates))


def holdout_baselines(batch, seed):
    """Nearest and random choices are diagnostics on exactly the same holdout."""
    features, _, probability, mask = batch
    with torch.inference_mode():
        distance = torch.linalg.vector_norm(features[..., :3], dim=-1)
        nearest = torch.zeros_like(probability)
        nearest.scatter_(-1, distance.masked_fill(~mask, torch.inf).argmin(-1, keepdim=True), 1.)
        nearest *= mask.any(dim=-1, keepdim=True)
        generator = torch.Generator().manual_seed(seed)
        random_scores = torch.rand(probability.shape, generator=generator)
        random = torch.zeros_like(probability)
        random.scatter_(-1, random_scores.masked_fill(~mask, -1.).argmax(-1, keepdim=True), 1.)
        random *= mask.any(dim=-1, keepdim=True)
        uniform_loss = float(service_leakage_loss(torch.zeros_like(probability), probability, mask))
        return dict(nearest_hard_leakage_loss=_hard_loss(nearest, probability, mask),
                    random_hard_leakage_loss=_hard_loss(random, probability, mask),
                    uniform_soft_leakage_loss=uniform_loss)


def train_policy(*, epochs: int = 300, seed: int = 7, batch_size: int = 32,
                 holdout_size: int = 128, hidden_dim: int = 32, learning_rate: float = .001,
                 cfg: Config | None = None, progress: bool = False):
    """Return genuinely updated weights plus fixed-holdout evidence, without saving."""
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (epochs, batch_size, holdout_size, hidden_dim)):
        raise ValueError('epochs, batch_size, holdout_size and hidden_dim must be positive integers')
    if not isinstance(seed, int) or seed < 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('seed and learning_rate are invalid')
    cfg = Config() if cfg is None else cfg
    if cfg.r_dock-cfg.min_separation <= 1.:
        raise ValueError('training annulus must be wider than one metre')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    holdout_seed = seed+1_000_003
    holdout = make_training_batch(np.random.default_rng(holdout_seed), cfg, holdout_size)
    model = SwarmGNN(hidden_dim)
    initial_weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    initial = evaluate_policy(model, holdout)
    baselines = holdout_baselines(holdout, seed+2_000_003)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        features, adjacency, probability, mask = make_training_batch(rng, cfg, batch_size)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, adjacency, mask)
        loss = service_leakage_loss(logits, probability, mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite training loss at epoch {epoch+1}')
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise RuntimeError(f'nonfinite gradient at epoch {epoch+1}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        history.append(float(loss.detach()))
        if progress and (epoch == 0 or (epoch+1) % 50 == 0 or epoch+1 == epochs):
            print(f'epoch {epoch+1}/{epochs}: training soft leakage={history[-1]:.6f}', flush=True)
    final = evaluate_policy(model, holdout)
    finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
    delta = sum(float(torch.sum((parameter.detach()-initial_weights[name])**2))
                for name, parameter in model.named_parameters())**.5
    changed = sum(not torch.equal(parameter.detach(), initial_weights[name])
                  for name, parameter in model.named_parameters())
    if not finite or not np.isfinite(delta) or delta <= 0:
        raise RuntimeError('training did not produce finite updated weights')
    report = dict(format_version=1, task='non-contact observation-band service leakage',
        seed=seed, holdout_seed=holdout_seed, trained_epochs=epochs, optimizer_steps=epochs,
        epoch_definition='one fresh randomized minibatch and one Adam update',
        batch_size=batch_size, training_snapshots=epochs*batch_size, holdout_snapshots=holdout_size,
        hidden_dim=hidden_dim, learning_rate=learning_rate, config=asdict(cfg),
        snapshot_distribution=dict(target_speed_norm=[cfg.target_speed_min, cfg.target_speed_max],
            target_altitude=[cfg.altitude_min, cfg.altitude_max], chaser_speed_norm_max=cfg.v_max,
            broadcast_range_multiplier=[.05, 1.], broadcast_range_sampling='log-uniform',
            sensor_range_multiplier=[.35, 1.]),
        initial_holdout=initial, final_holdout=final, holdout_baselines=baselines,
        holdout_soft_loss_improvement=initial['soft_leakage_loss']-final['soft_leakage_loss'],
        weights_finite=finite, weight_delta_l2=delta, changed_parameter_tensors=changed,
        total_parameter_tensors=len(initial_weights), parameter_count=sum(p.numel() for p in model.parameters()),
        training_loss_history=history, training_wall_seconds=round(time.perf_counter()-start, 4),
        limitations=['Soft leakage does not guarantee unique hard choices or rollout success.',
                     'Synthetic snapshot holdout is not an independent real-world evaluation.',
                     'No collision or capture reward is used; service targets the safe observation band.'])
    model.eval()
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--holdout-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'output'/'swarm_gnn_model.pt')
    parser.add_argument('--json', type=Path, default=PROJECT_ROOT/'output'/'gnn-training.json')
    args = parser.parse_args()
    model, report = train_policy(epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
        holdout_size=args.holdout_size, hidden_dim=args.hidden_dim, learning_rate=args.learning_rate, progress=True)
    metrics = {key: report[key] for key in ('initial_holdout', 'final_holdout', 'holdout_baselines',
        'holdout_soft_loss_improvement', 'weights_finite', 'weight_delta_l2', 'changed_parameter_tensors')}
    save_checkpoint(model, args.output, trained_epochs=args.epochs, metrics=metrics,
        metadata=dict(seed=args.seed, holdout_seed=report['holdout_seed'],
                      optimizer='Adam', optimizer_steps=args.epochs,
                      training_snapshots=report['training_snapshots'],
                      snapshot_distribution=report['snapshot_distribution'],
                      task=report['task'], config=report['config']))
    report['checkpoint_path'] = str(args.output.resolve())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(checkpoint=str(args.output), report=str(args.json),
        initial_holdout=report['initial_holdout'], final_holdout=report['final_holdout'],
        weight_delta_l2=report['weight_delta_l2'], weights_finite=report['weights_finite']), sort_keys=True))


if __name__ == '__main__':
    main()
