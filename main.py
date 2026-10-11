"""Continuous noncontact observation world, with optional three-policy benchmark.

Default: fullscreen live world. Use --benchmark for finite seeded comparisons,
or --headless --duration 60 for a timed continuous simulation without a window.
Control is 50 Hz; MuJoCo force integration is 500 Hz, independent of rendering.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from sim_engine import Config, Simulation
from swarm_controller import SwarmController
from reservations import ReservationManager

DEFAULT_MODES = ('greedy', 'auction', 'gnn')


def project_isometric(point, *, ground_z=0., scale=1., origin=(0., 0.)):
    """Project metre-valued (..., 3) world positions to (..., 2) screen pixels.

    x/y span the ground plane; positive z moves up the screen. This pure helper
    deliberately has no Pygame dependency so geometry can be checked headlessly.
    """
    point = np.asarray(point, dtype=float)
    if point.ndim == 0 or point.shape[-1] != 3:
        raise ValueError('isometric projection requires (..., 3) positions')
    return np.stack(((point[..., 0]-point[..., 1])*np.sqrt(3.)/2.,
                     (point[..., 0]+point[..., 1])*.5-(point[..., 2]-ground_z)),
                    axis=-1)*scale + np.asarray(origin)


class OrbitCamera:
    """Orthographic orbit camera; all state belongs to the view, not the plant.

    Screen right/down and the direction toward the camera form an orthonormal
    basis. A point projects as ``viewport_center + scale * (point-center) @
    [right, down].T``; depth is its component toward the camera. Elevation stays
    above the ground and below the pole, while azimuth can circle the arena.
    """
    def __init__(self, cfg, viewport=(38, 128, 870, 585)):
        values = np.asarray(viewport, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all() or np.any(values[2:] <= 0):
            raise ValueError('viewport must contain finite x, y, positive width and height')
        self.viewport = tuple(values)
        self._screen_center = values[:2]+values[2:]/2.
        self._initial_center = np.array([cfg.width/2., cfg.depth/2., cfg.ground_z+cfg.height/2.])
        self.reset()
        corners = np.array([[x, y, z] for x in (0., cfg.width)
                            for y in (0., cfg.depth)
                            for z in (cfg.ground_z, cfg.ground_z+cfg.height)])
        right, down, _ = self._basis()
        projected = (corners-self.center) @ np.stack((right, down)).T
        # Leave room for ground/axis annotations at the default orientation.
        usable = values[2:]-2.*np.minimum([50., 30.], values[2:]*.1)
        self.base_scale = float(np.min(usable/np.ptp(projected, axis=0)))

    def reset(self):
        """Restore the volume center, isometric angles, and unit zoom."""
        self.center = self._initial_center.copy()
        self.azimuth = np.pi/4.
        self.elevation = float(np.arctan(1./np.sqrt(2.)))
        self.zoom = 1.

    @property
    def scale(self):
        return self.base_scale*self.zoom

    def _basis(self):
        sa, ca = np.sin(self.azimuth), np.cos(self.azimuth)
        se, ce = np.sin(self.elevation), np.cos(self.elevation)
        return (np.array([ca, -sa, 0.]), np.array([se*sa, se*ca, -ce]),
                np.array([ce*sa, ce*ca, se]))

    @staticmethod
    def _vectors(values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 0 or values.shape[-1] != 3 or not np.isfinite(values).all():
            raise ValueError('camera projection requires finite (..., 3) vectors')
        return values

    def project(self, points):
        """Map world points to pixels, including orbit center and viewport."""
        return self.project_vector(self._vectors(points)-self.center)+self._screen_center

    def project_vector(self, vectors):
        """Map directions to pixel deltas without translating them."""
        right, down, _ = self._basis()
        return self._vectors(vectors) @ np.stack((right, down)).T*self.scale

    def depth(self, points):
        """Return signed view depth; larger values are nearer the camera."""
        _, _, view = self._basis()
        return (self._vectors(points)-self.center) @ view

    def orbit(self, dx, dy):
        """Rotate by mouse-pixel deltas at 0.006 radians per pixel."""
        if not np.isfinite([dx, dy]).all():
            raise ValueError('orbit deltas must be finite')
        self.azimuth = float((self.azimuth+dx*.006) % (2.*np.pi))
        self.elevation = float(np.clip(self.elevation+dy*.006,
                                       np.deg2rad(5.), np.deg2rad(85.)))

    def pan(self, dx, dy):
        """Translate the view so fixed world points follow the dragged pixels."""
        if not np.isfinite([dx, dy]).all():
            raise ValueError('pan deltas must be finite')
        right, down, _ = self._basis()
        self.center -= (dx*right+dy*down)/self.scale

    def zoom_by(self, steps):
        """Apply exponential wheel zoom, clamped to 0.2–5 times fitted scale."""
        if not np.isfinite(steps):
            raise ValueError('zoom steps must be finite')
        factor = np.exp(np.clip(steps*np.log(1.12), -50., 50.))
        self.zoom = float(np.clip(self.zoom*factor, .2, 5.))


class SimulatedMesh:
    """Reliable zero-latency radio flooding over range-limited 3-D links.

    World positions are used only for network connectivity. The transport reads
    packet addressing/lifecycle metadata, never capabilities or target choices.
    Every live receiver in the sender's connected component gets the immutable
    packet once, including through intermediate peers. Disconnected components
    cannot negotiate with one another. Controllers own and expire their buffers.
    """
    def __init__(self, cfg: Config):
        self.range = cfg.broadcast_range
        self.ttl = cfg.broadcast_ttl
        self.sent_packets = 0
        self.deliveries = 0
        self.rounds = 0
        self.sender_packets = {i: 0 for i in range(cfg.n_chasers)}
        self.withdrawn_senders = set()
        self._link_positions = None
        self._link_adjacency = None

    def _links(self, positions, active):
        positions = np.asarray(positions, dtype=float)
        active = np.asarray(active, dtype=bool)
        n = len(self.sender_packets)
        if positions.shape != (n, 3) or active.shape != (n,) or not np.isfinite(positions).all():
            raise ValueError('mesh requires finite (N,3) positions and (N,) live mask')
        if self._link_positions is None or not np.array_equal(positions, self._link_positions):
            self._link_positions = positions.copy()
            self._link_adjacency = np.linalg.norm(
                positions[:, None, :]-positions[None, :, :], axis=-1) <= self.range
        return self._link_adjacency, active

    @staticmethod
    def _reachable(sender, links, allowed):
        seen, frontier = {int(sender)}, [int(sender)]
        while frontier:
            node = frontier.pop()
            for neighbor in np.flatnonzero(links[node] & allowed):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        return seen

    def connected_components(self, positions, active):
        """Return sorted live member groups as transport topology metadata.

        The signature includes every active ID and no target or bid data. A
        component merge requires fresh agreement before the next actuation.
        """
        links, active = self._links(positions, active)
        remaining, groups = set(np.flatnonzero(active)), []
        while remaining:
            members = self._reachable(min(remaining), links, active)
            groups.append(tuple(sorted(members)))
            remaining -= members
        return tuple(groups)

    def exchange(self, messages, controllers, positions, active, time, *, retiring=()):
        """Flood one synchronous packet batch; return its receiver deliveries.

        ``retiring`` authorizes one final inactive packet per departing sender.
        For that packet only, its contact pose joins the otherwise-live graph.
        Retired craft never relay other packets or send subsequent messages.
        """
        n = len(controllers)
        if n != len(self.sender_packets):
            raise ValueError('mesh controller roster must match its configured sender roster')
        links, active = self._links(positions, active)
        if not np.isfinite(time):
            raise ValueError('mesh time must be finite')
        messages = tuple(messages)
        departing = set(retiring)
        deliveries = 0
        self.rounds += 1
        for message in messages:
            sender = message.sender_id
            if not isinstance(sender, (int, np.integer)) or not 0 <= sender < n:
                continue
            if sender in self.withdrawn_senders:
                continue
            age = time-message.time
            if not np.isfinite(age) or age < -1e-9 or age > self.ttl+1e-9:
                continue
            final_withdrawal = sender in departing and not active[sender] and not message.active
            if not (active[sender] and message.active) and not final_withdrawal:
                continue
            allowed = active.copy()
            allowed[sender] = True
            seen = self._reachable(sender, links, allowed)
            self.sent_packets += 1
            self.sender_packets[int(sender)] += 1
            if final_withdrawal:
                self.withdrawn_senders.add(int(sender))
            for receiver in sorted(seen-{int(sender)}):
                controllers[receiver].receive_broadcast(message)
                deliveries += 1
        self.deliveries += deliveries
        return deliveries


class Scenario:
    """SITL wiring, mesh transport, and explicit mission reservations.

    Policy scores come from local telemetry. A shared mission ledger enforces
    the configured capacity after policy allocation, including across radio
    partitions. This global reservation guarantee is an additional centralized
    simulation service, not a property of the learned network or local mesh.
    """
    def __init__(self, cfg: Config, seed: int, mode: str, model_path=None):
        self.sim = Simulation(cfg, seed)
        self.mode = 'auction' if mode == 'coordinated' else mode
        self.controllers = [SwarmController(i, cfg, self.mode, model_path=model_path) for i in range(cfg.n_chasers)]
        self.mesh = SimulatedMesh(cfg)
        self.reservations = ReservationManager(cfg.chasers_per_target)
        self.allocation_epoch = 0
        self.allocation_rounds = 0
        self.max_allocation_rounds = 0
        self._last_round_index = -1
        self._last_topology = None
        self._generations = self.sim.chaser_generation.copy()
        self.loss_sum = 0.
        self.steps = 0
        self.control_seconds = 0.
        self.duplicate_sum = 0.
        self.max_duplicate_commitments = 0
        self.max_capacity_violations = 0

    def _sync_incarnations(self):
        """Re-admit a fresh craft after the plant confirms a safe base launch.

        Bay IDs are reused, but old commitments and withdrawal tombstones
        belong to the previous craft. Transport counters and learned weights
        remain on the controller, avoiding stale packets and repeated loads.
        """
        changed = np.flatnonzero(self.sim.chaser_generation != self._generations)
        for bay in changed:
            bay = int(bay)
            self.controllers[bay].reset_for_launch(epoch=self.allocation_epoch)
            self.reservations.forget_agent(bay)
            self.mesh.withdrawn_senders.discard(bay)
            for controller in self.controllers:
                controller.forget_peer_incarnation(bay)
        if len(changed):
            self._last_topology = None
            self._generations = self.sim.chaser_generation.copy()

    def step(self):
        start = time.perf_counter()
        self._sync_incarnations()
        for i, controller in enumerate(self.controllers):
            controller.update_telemetry(self.sim.observe(i))
        active_controllers = [c for i, c in enumerate(self.controllers) if self.sim.chaser_active[i]]
        topology = self.mesh.connected_components(self.sim.chaser_pos, self.sim.chaser_active)
        topology_changed = topology != self._last_topology
        self._last_topology = topology
        if (self.mode == 'auction' and active_controllers
                and (topology_changed or any(c.allocation_due for c in active_controllers))):
            self.allocation_epoch += 1
            for controller in active_controllers:
                controller.begin_allocation(self.allocation_epoch)
            # All proposals exist before delivery; all receipts precede local
            # resolution. This is the network's round clock, not an allocator.
            limit = self.sim.config.n_chasers*len(self.sim.target_ids)+1
            for round_index in range(limit):
                messages = tuple(c.make_broadcast(round_index) for c in active_controllers)
                self.mesh.exchange(messages, self.controllers, self.sim.chaser_pos,
                                   self.sim.chaser_active, self.sim.time)
                changed = [c.resolve_round() for c in active_controllers]
                self._last_round_index = round_index
                self.allocation_rounds += 1
                if not any(changed):
                    break
            else:
                raise RuntimeError('mesh allocation did not converge before the conservative round bound')
            self.max_allocation_rounds = max(self.max_allocation_rounds, round_index+1)
            for controller in active_controllers:
                controller.finalize_allocation()
        assignments = self.reservations.assign(self.controllers)
        self.sim.set_observation_assignments(assignments)
        commands = np.stack([c.get_control_commands() for c in self.controllers])
        self.control_seconds += time.perf_counter()-start
        duplicates = self.duplicate_commitments
        self.duplicate_sum += duplicates
        self.max_duplicate_commitments = max(self.max_duplicate_commitments, duplicates)
        self.max_capacity_violations = max(self.max_capacity_violations, self.capacity_violations)
        self.steps += 1
        previously_active = self.sim.chaser_active.copy()
        self.sim.step(commands)
        self._sync_incarnations()
        # Refresh onboard time before delivering final withdrawals, including
        # the terminal tick. Survivors clear lost tracks locally and can request
        # a new epoch on the next tick; no post-step actuation happens here.
        start = time.perf_counter()
        for i, controller in enumerate(self.controllers):
            controller.update_telemetry(self.sim.observe(i))
        # Live telemetry can invalidate or replace a raw policy choice. Apply
        # the ledger again before exposing the state to the renderer/caller.
        self.sim.set_observation_assignments(self.reservations.assign(self.controllers))
        # Publish accepted commitments, not pre-reservation preferences. Round
        # counters restart only with a new auction epoch; all other updates use
        # fresh increasing rounds, preserving stale-packet rejection.
        self._last_round_index += 1
        messages = tuple(c.make_broadcast(self._last_round_index) for i, c in enumerate(self.controllers)
                         if self.sim.chaser_active[i])
        self.mesh.exchange(messages, self.controllers, self.sim.chaser_pos,
                           self.sim.chaser_active, self.sim.time)
        self.loss_sum += self.local_loss
        retiring = np.flatnonzero(previously_active & ~self.sim.chaser_active)
        if len(retiring):
            self._last_round_index += 1
            messages = tuple(self.controllers[i].make_broadcast(self._last_round_index) for i in retiring)
            self.mesh.exchange(messages, self.controllers, self.sim.chaser_pos,
                               self.sim.chaser_active, self.sim.time, retiring=retiring)
        self.control_seconds += time.perf_counter()-start

    @property
    def duplicate_commitments(self):
        chosen = self.active_commitments
        return len(chosen)-len(set(chosen))

    @property
    def active_commitments(self):
        return [c.target_id for i, c in enumerate(self.controllers)
                if self.sim.chaser_active[i] and c.target_id is not None
                and (slot := self.sim.target_index(c.target_id)) is not None
                and self.sim.status[slot] == 0]

    @property
    def capacity_violations(self):
        from collections import Counter
        return sum(max(0, count-self.sim.config.chasers_per_target)
                   for count in Counter(self.active_commitments).values())

    @property
    def local_loss(self):
        values = [c.allocation_loss for i, c in enumerate(self.controllers)
                  if self.sim.chaser_active[i]]
        return float(np.mean(values)) if values else 0.

    def result(self):
        forward_calls = sum(c.gnn_forward_calls for c in self.controllers)
        forward_seconds = sum(c.gnn_forward_seconds for c in self.controllers)
        compute_seconds = sum(c.gnn_compute_seconds for c in self.controllers)
        return dict(mode=self.mode, seed=self.sim.seed, config=asdict(self.sim.config), **self.sim.stats(),
                    mean_local_allocation_loss=self.loss_sum/max(1, self.steps),
                    mean_duplicate_commitments=self.duplicate_sum/max(1, self.steps),
                    max_duplicate_commitments=self.max_duplicate_commitments,
                    max_capacity_violations=self.max_capacity_violations,
                    chasers_per_target=self.sim.config.chasers_per_target,
                    reservation_model='shared mission ledger over local policy candidates',
                    broadcast_packets=self.mesh.sent_packets,
                    broadcast_deliveries=self.mesh.deliveries,
                    broadcast_rounds=self.mesh.rounds,
                    packets_by_sender=dict(self.mesh.sender_packets),
                    allocation_epochs=self.allocation_epoch,
                    allocation_rounds=self.allocation_rounds,
                    max_allocation_rounds=self.max_allocation_rounds,
                    auction_rounds=self.allocation_rounds,
                    gnn_forward_calls=forward_calls,
                    gnn_forward_wall_seconds=round(forward_seconds, 6),
                    gnn_compute_wall_seconds=round(compute_seconds, 6),
                    mean_gnn_forward_ms=1000.*forward_seconds/forward_calls if forward_calls else None,
                    communication_model='reliable zero-latency simulated mesh; wall compute measured separately',
                    controller_wall_seconds=round(self.control_seconds, 4))


def run_headless(cfg: Config, seed: int, modes=DEFAULT_MODES, model_path=None) -> list[dict]:
    if cfg.duration is None:
        raise ValueError('headless runs require a finite duration')
    results = []
    for mode in modes:
        scenario = Scenario(cfg, seed, mode, model_path=model_path)
        while not scenario.sim.done:
            scenario.step()
        results.append(scenario.result())
    return results


def run_visual(cfg, seed, speed=1., auto_exit=False, screenshot=None,
               modes=DEFAULT_MODES, model_path=None, fullscreen=True):
    # Headless runs never initialize Pygame or an OpenGL rendering context.
    from world_renderer import Renderer
    modes = tuple(modes)
    if not modes:
        raise ValueError('at least one benchmark mode is required')
    ui = Renderer(cfg, fullscreen=fullscreen)
    pg, clock = ui.pg, ui.pg.time.Clock()
    scenario = Scenario(cfg, seed, modes[0], model_path=model_path)
    results, accumulator, paused, running = [], 0., False, True
    profiles = ('mixed', 'random', 'constant-height', 'weave', 'parabolic', 'orbit')
    completed = False

    def save_screenshot():
        if screenshot:
            path = Path(screenshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            pg.image.save(ui.screen, str(path))

    try:
        while running:
            wall_dt = clock.tick(60)/1000.
            settings = None
            for event in pg.event.get():
                if ui.handle_event(event):
                    continue
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    running = False
                elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    paused = not paused
                elif event.type == pg.KEYDOWN and event.key == pg.K_r:
                    scenario = Scenario(cfg, seed, modes[0], model_path=model_path)
                    results, accumulator, paused, completed = [], 0., False, False
                elif cfg.continuous and event.type == pg.KEYDOWN:
                    if event.key in (pg.K_1, pg.K_2, pg.K_3):
                        modes = (DEFAULT_MODES[event.key-pg.K_1],)
                        scenario = Scenario(cfg, seed, modes[0], model_path=model_path)
                        results, accumulator, paused, completed = [], 0., False, False
                    elif event.key == pg.K_t:
                        settings = {'trajectory': profiles[(profiles.index(cfg.trajectory)+1) % len(profiles)]}
                    elif event.key in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
                        if not cfg.recycle_on_service:
                            settings = {'chasers_per_target': min(cfg.n_chasers, cfg.chasers_per_target+1)}
                    elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
                        settings = {'chasers_per_target': max(1, cfg.chasers_per_target-1)}
                    elif event.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                        settings = {'arrival_interval': max(.5, cfg.arrival_interval+(-.5 if event.key == pg.K_LEFTBRACKET else .5))}
            requested_settings = ui.consume_settings()
            if requested_settings is not None:
                settings = requested_settings
            if settings:
                cfg = replace(cfg, **settings)
                was_fullscreen = ui.fullscreen
                ui.close()
                ui = Renderer(cfg, fullscreen=was_fullscreen)
                pg, clock = ui.pg, ui.pg.time.Clock()
                scenario = Scenario(cfg, seed, modes[0], model_path=model_path)
                results, accumulator, paused, completed = [], 0., False, False
            if not paused and len(results) < len(modes):
                accumulator += wall_dt*speed
                # Limit work per frame for event responsiveness; keep the backlog.
                for _ in range(min(25, int((accumulator+1e-12)/cfg.dt))):
                    scenario.step()
                    accumulator = max(0., accumulator-cfg.dt)
                    if scenario.sim.done:
                        results.append(scenario.result())
                        completed = len(results) == len(modes)
                        accumulator = 0.
                        if len(results) < len(modes):
                            scenario = Scenario(cfg, seed, modes[len(results)], model_path=model_path)
                        break
            alpha = 1. if scenario.sim.done or paused else min(1., accumulator/cfg.dt)
            ui.draw(scenario, alpha, results, paused, expected_results=len(modes))
            if len(results) == len(modes):
                if screenshot:
                    save_screenshot()
                    screenshot = None
                if auto_exit:
                    break
    finally:
        save_screenshot()
        ui.close()
    if cfg.continuous and not completed:
        results = [dict(scenario.result(), session_stopped=True)]
    return results


def write_results(path, launch_config, results):
    """Preserve launch settings and the effective restarted-world configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    initial = asdict(launch_config)
    effective = results[-1].get('config', initial) if results else initial
    path.write_text(json.dumps(dict(config=effective, launch_config=initial, results=results), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='Run a timed simulation without opening a window')
    parser.add_argument('--benchmark', action='store_true', help='Finite seeded Greedy / Auction / GNN comparison')
    parser.add_argument('--windowed', action='store_true', help='Open a resizable window instead of fullscreen')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--duration', type=float, help='Simulation seconds; live default runs until quit, headless/benchmark default 35')
    parser.add_argument('--chasers', type=int, default=6)
    parser.add_argument('--targets', type=int, default=6)
    parser.add_argument('--chasers-per-object', '--chasers-per-target', dest='chasers_per_target', type=int, default=1,
                        help='Maximum observers per object; values above 1 require --keep-observers')
    parser.add_argument('--arrival-interval', type=float, default=4., help='Seconds between new tracks in continuous mode')
    parser.add_argument('--max-tracks', type=int, default=24, help='Bounded number of physical tracks in the live world')
    parser.add_argument('--trajectory', choices=('mixed', 'random', 'constant-height', 'weave', 'parabolic', 'orbit'), default='mixed')
    parser.add_argument('--target-speed-min', type=float, default=3.)
    parser.add_argument('--target-speed-max', type=float, default=7.)
    parser.add_argument('--chaser-speed', type=float, default=12., metavar='MPS',
                        help='Observer cruise and commanded speed limit in m/s')
    parser.add_argument('--safe-distance', '--min-separation', dest='min_separation',
                        type=float, default=5., metavar='METRES',
                        help='Minimum non-contact separation for observation, avoidance and launch clearance (default: 5)')
    parser.add_argument('--stand-off-distance', type=float, metavar='METRES',
                        help='Preferred tracking range; default max(10, safe distance + 5)')
    parser.add_argument('--observation-radius', type=float, metavar='METRES',
                        help='Outer observation range; default max(16, stand-off + 6)')
    parser.add_argument('--profile-speed', nargs=3, action='append', default=[],
                        metavar=('PROFILE', 'MIN', 'MAX'),
                        help='Override one trajectory speed range in m/s; repeat for multiple profiles')
    parser.add_argument('--relaunch-delay', type=float, default=1.5, metavar='SECONDS',
                        help='Minimum delay before replacing a completed observer at a clear base pad')
    parser.add_argument('--keep-observers', action='store_true',
                        help='Keep observers after safe observation; enables multi-observer capacity experiments')
    parser.add_argument('--altitude-min', type=float, default=20.)
    parser.add_argument('--altitude-max', type=float, default=60.)
    parser.add_argument('--wind', nargs=3, type=float, default=(0., 0., 0.), metavar=('X', 'Y', 'Z'), help='Wind vector in m/s')
    parser.add_argument('--mass', type=float, default=1.5, help='Observer mass in kg')
    parser.add_argument('--gravity', type=float, default=9.81, help='Gravity in m/s²')
    parser.add_argument('--mode', choices=DEFAULT_MODES, default='gnn', help='Continuous world policy')
    parser.add_argument('--sensor-range', type=float, default=150.)
    parser.add_argument('--broadcast-range', type=float, default=150., help='Local radio link range in metres')
    parser.add_argument('--model', type=Path, help='GNN checkpoint; default output/swarm_gnn_model.pt')
    parser.add_argument('--modes', nargs='+', choices=('greedy', 'auction', 'gnn', 'coordinated'),
                        help='Select finite benchmark modes; implies --benchmark')
    parser.add_argument('--speed', type=float, default=1., help='Visual playback rate; does not alter dt')
    parser.add_argument('--auto-exit', action='store_true', help='Close window after selected modes complete')
    parser.add_argument('--screenshot', type=Path, help='Save the final visual comparison as PNG')
    parser.add_argument('--json', type=Path, help='Write full configuration and benchmark results')
    args = parser.parse_args()
    args.benchmark = args.benchmark or args.modes is not None
    args.modes = ['auction' if mode == 'coordinated' else mode for mode in
                  (args.modes or (DEFAULT_MODES if args.benchmark else (args.mode,)))]
    if len(set(args.modes)) != len(args.modes):
        parser.error('choose each mode once; coordinated is a backward-compatible alias for auction')
    if args.seed < 0 or not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('seed must be nonnegative and speed must be finite and positive')
    if args.headless and args.screenshot:
        parser.error('--screenshot requires visual mode (SDL dummy driver also works)')
    if args.auto_exit and not args.benchmark and args.duration is None:
        parser.error('--auto-exit requires --duration in the continuous world, or --benchmark')
    try:
        stand_off = (args.stand_off_distance if args.stand_off_distance is not None
                     else max(10., args.min_separation+5.))
        observation_radius = (args.observation_radius if args.observation_radius is not None
                              else max(16., stand_off+6.))
        cfg = Config(n_chasers=args.chasers, n_targets=args.targets,
                     continuous=not args.benchmark,
                     duration=args.duration if args.duration is not None else (35. if args.benchmark or args.headless else None),
                     sensor_range=args.sensor_range, broadcast_range=args.broadcast_range,
                     chasers_per_target=args.chasers_per_target, arrival_interval=args.arrival_interval,
                     max_tracks=args.max_tracks, trajectory=args.trajectory,
                     target_speed_min=args.target_speed_min, target_speed_max=args.target_speed_max,
                     v_max=args.chaser_speed, cruise_speed=args.chaser_speed,
                     min_separation=args.min_separation, stand_off_distance=stand_off,
                     r_dock=observation_radius,
                     profile_speeds=tuple((profile, float(low), float(high))
                                          for profile, low, high in args.profile_speed),
                     recycle_on_service=not args.keep_observers, relaunch_delay=args.relaunch_delay,
                     altitude_min=args.altitude_min, altitude_max=args.altitude_max,
                     wind=tuple(args.wind), mass=args.mass, gravity=args.gravity)
    except ValueError as error:
        parser.error(str(error))
    # Small tensor optimizations are much faster without threaded BLAS overhead.
    torch.set_num_threads(1)
    try:
        if 'gnn' in args.modes:
            from gnn_model import load_policy
            path = args.model or Path(__file__).parent/'output'/'swarm_gnn_model.pt'
            if not path.is_file():
                parser.error(f'GNN checkpoint missing: {path}. Run .venv/bin/python train_gnn.py first, '
                             'or choose --modes greedy auction.')
            load_policy(path)  # Fail before starting playback on incompatible checkpoints.
        results = run_headless(cfg, args.seed, args.modes, args.model) if args.headless else run_visual(
            cfg, args.seed, args.speed, args.auto_exit, args.screenshot, args.modes, args.model,
            fullscreen=not args.windowed)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    for result in results:
        print(json.dumps(result, sort_keys=True))
    if len(results) != len(args.modes):
        print('Benchmark interrupted; only completed scenarios are reported.')
    if args.json:
        write_results(args.json, cfg, results)


if __name__ == '__main__':
    main()
