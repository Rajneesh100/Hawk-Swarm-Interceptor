"""MuJoCo plant for reusable, non-contact aerial observation.

The command-only plant imports no policy. Service requires continuous
observation inside a safe spherical annulus. Optional completion recycling
withdraws the serviced track and observer, then launches a new bay incarnation.
"""
from dataclasses import dataclass
from collections import deque
from physics_constants import GRAVITY, AIR_DENSITY, TRAJECTORIES
from physics_world import PhysicsWorld
import numpy as np
from estimator import Observation


ACTIVE, SERVICED, ESCAPED = 0, 1, 2
RETIRED = SERVICED  # Deprecated target-status alias.


def limit_norm(v: np.ndarray, maximum: float) -> np.ndarray:
    """Limit Euclidean vector norms while preserving direction, also in batches."""
    return v * np.minimum(1., maximum / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


@dataclass(frozen=True)
class Config:
    dt: float = .02
    tau: float = .1  # Deprecated response-model compatibility; not used by physics.
    drag_coeff: float = 1.  # Deprecated; drag_area is the physical Cd*A.
    physics_dt: float = .002
    mass: float = 1.5
    gravity: float = GRAVITY
    air_density: float = AIR_DENSITY
    drag_area: float = .06
    thrust_max: float = 45.
    thrust_tau: float = .08
    wind: tuple[float, float, float] = (0., 0., 0.)
    body_radius: float = .35
    v_max: float = 12.
    a_max: float = 18.
    r_dock: float = 16.  # Legacy name: outer OBSERVATION radius, not contact.
    min_separation: float = 5.
    stand_off_distance: float = 10.
    service_duration: float = 2.
    n_chasers: int = 6
    n_targets: int = 6
    width: float = 100.
    depth: float = 60.
    height: float = 80.
    ground_z: float = 0.
    duration: float | None = 35.
    continuous: bool = False
    arrival_interval: float = 4.
    max_tracks: int = 24
    trajectory: str = 'mixed'
    chasers_per_target: int = 1
    recycle_on_service: bool = False
    relaunch_delay: float = 1.5
    offstage_margin: float = 40.
    target_speed_min: float = 3.
    target_speed_max: float = 7.
    profile_speeds: tuple[tuple[str, float, float], ...] = ()
    altitude_min: float = 20.
    altitude_max: float = 60.
    target_accel_max: float = 3.
    turn_accel: float = .5
    turn_frequency: float = .5
    waypoint_period: float = 4.
    drift_std: float = .12
    position_noise: float = .08
    velocity_noise: float = .025
    sensor_range: float = 150.
    broadcast_range: float = 150.
    broadcast_ttl: float = 1.
    sigma_base: float = .6
    zem_scale: float = .015
    n_nav: float = 3.
    cruise_speed: float = 10.
    allocation_period: float = .5

    def __post_init__(self):
        positive = ('dt', 'tau', 'drag_coeff', 'v_max', 'a_max', 'r_dock',
                    'min_separation', 'stand_off_distance', 'service_duration',
                    'width', 'depth', 'height', 'sensor_range',
                    'physics_dt', 'mass', 'gravity', 'thrust_max', 'body_radius',
                    'arrival_interval', 'offstage_margin', 'relaunch_delay',
                    'broadcast_range', 'broadcast_ttl', 'sigma_base', 'n_nav',
                    'cruise_speed', 'allocation_period', 'target_speed_min',
                    'target_speed_max', 'target_accel_max', 'waypoint_period')
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('drift_std', 'position_noise', 'velocity_noise', 'zem_scale',
                     'turn_accel', 'turn_frequency', 'air_density', 'drag_area', 'thrust_tau'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('n_chasers', 'n_targets', 'max_tracks', 'chasers_per_target'):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.duration is None:
            if not self.continuous:
                raise ValueError('duration=None requires continuous=True')
        elif not np.isfinite(self.duration) or self.duration <= 0.:
            raise ValueError('duration must be finite and positive or None in continuous mode')
        if self.continuous and self.max_tracks < self.n_targets:
            raise ValueError('max_tracks must accommodate the initial target count')
        if self.trajectory not in TRAJECTORIES:
            raise ValueError(f'trajectory must be one of {TRAJECTORIES}')
        if self.recycle_on_service and self.chasers_per_target != 1:
            raise ValueError('recycle_on_service requires chasers_per_target=1')
        seen_profiles = set()
        normalized_speeds = []
        for item in self.profile_speeds:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError('profile_speeds entries must be (profile, minimum, maximum)')
            profile, minimum, maximum = item
            if profile not in TRAJECTORIES[1:] or profile in seen_profiles:
                raise ValueError('profile_speeds requires unique actual trajectory names')
            if not np.isfinite([minimum, maximum]).all() or not 0. < minimum <= maximum:
                raise ValueError('profile speed bounds must be finite, positive, and ordered')
            seen_profiles.add(profile)
            normalized_speeds.append((profile, float(minimum), float(maximum)))
        object.__setattr__(self, 'profile_speeds', tuple(normalized_speeds))
        if np.asarray(self.wind).shape != (3,) or not np.isfinite(self.wind).all():
            raise ValueError('wind must be a finite three-vector in m/s')
        ratio = self.dt/self.physics_dt
        if ratio < 1 or abs(ratio-round(ratio)) > 1e-9:
            raise ValueError('dt must be an integer multiple of physics_dt')
        if not np.isfinite([self.ground_z, self.altitude_min, self.altitude_max]).all():
            raise ValueError('altitude bounds and ground_z must be finite')
        if not self.min_separation < self.stand_off_distance < self.r_dock:
            raise ValueError('require min_separation < stand_off_distance < observation radius')
        if self.min_separation <= 2.*self.body_radius:
            raise ValueError('min_separation must exceed the physical body diameter (2 * body_radius)')
        if self.target_speed_max < self.target_speed_min:
            raise ValueError('target_speed_max must be at least target_speed_min')
        if not self.ground_z < self.altitude_min < self.altitude_max <= self.ground_z+self.height:
            raise ValueError('altitude bounds must lie above ground within the displayed volume')

    def speed_range(self, profile):
        """Return an actual profile's override, or the global target speed range."""
        for name, minimum, maximum in self.profile_speeds:
            if name == profile:
                return minimum, maximum
        return self.target_speed_min, self.target_speed_max


def _inside_interval(r0, delta, radius):
    """Open-sphere interval on a linear relative path, clipped to one tick."""
    aa = float(np.dot(delta, delta))
    cc = float(np.dot(r0, r0)-radius*radius)
    if aa < 1e-24:
        return (0., 1.) if cc < 0. else None
    bb = 2.*float(np.dot(r0, delta))
    discriminant = bb*bb-4.*aa*cc
    if discriminant <= 0.:
        return None
    root = np.sqrt(discriminant)
    lo, hi = max(0., (-bb-root)/(2.*aa)), min(1., (-bb+root)/(2.*aa))
    return (lo, hi) if hi > lo else None


def _safe_intervals(r0, delta, inner, outer, end=1.):
    interval = _inside_interval(r0, delta, outer)
    if interval is None:
        return []
    lo, hi = interval[0], min(interval[1], end)
    if hi <= lo:
        return []
    unsafe = _inside_interval(r0, delta, inner)
    if unsafe is None or unsafe[1] <= lo or unsafe[0] >= hi:
        return [(lo, hi)]
    pieces = [(lo, min(hi, unsafe[0])), (max(lo, unsafe[1]), hi)]
    return [(a, b) for a, b in pieces if b > a]


class Simulation:
    """Reusable observers, prescribed tracks, and continuous observation dwell.

    Desired net acceleration commands feed the force-based MuJoCo world. Service
    requires the same observer continuously within min_separation <= range <
    r_dock. Optional assignment metadata restricts the observation recipient;
    without it, the nearest physically eligible track is used. Optional recycling
    follows safe service, never contact. Separation violations are measured.
    Track motion is independent of commands, sensors, service, and reservation.
    """
    def __init__(self, config: Config = Config(), seed: int = 7):
        if seed < 0:
            raise ValueError('seed must be nonnegative')
        self.config, self.seed, self.tick = config, seed, 0
        self.capacity = config.max_tracks if config.continuous else config.n_targets
        m, n = self.capacity, config.n_chasers
        self.target_pos = np.zeros((m, 3))
        self.target_vel = np.zeros((m, 3))
        self.target_present = np.zeros(m, dtype=bool)
        self.target_ids = np.full(m, -1, dtype=int)
        self.trajectory_names = np.full(m, '', dtype=object)
        self.status = np.full(m, -1, dtype=np.int8)
        self.target_event_pos = np.full((m, 3), np.nan)
        self._target_speeds = np.zeros(m)
        self._turn_phases = np.zeros(m)
        self._birth_times = np.zeros(m)
        self._spawn_pos = np.zeros((m, 3))
        self._waypoint_cache = {}  # at most two live time-index entries
        self._next_target_id = 0
        self._next_arrival = config.arrival_interval
        self._arrivals = self._serviced = self._escaped = self._dropped_arrivals = 0
        self._service_time_sum = 0.
        self.service_times = deque(maxlen=256)
        self.service_dwell = np.zeros((n, m))
        self.service_agent = np.full(m, -1, dtype=int)
        self.observation_assignments = None
        self.target_reservations = np.zeros(m, dtype=int)
        self.minimum_target_separation = np.inf
        self.minimum_peer_separation = np.inf
        self.near_contact_violations = 0
        self._target_close = np.zeros((n, m), dtype=bool)
        self._peer_close = np.zeros((n, n), dtype=bool)
        self.chaser_pos = np.column_stack((
            np.full(n, .12*config.width),
            np.linspace(.12, .88, n)*config.depth,
            np.full(n, .5*(config.altitude_min+config.altitude_max))))
        if n > 7:
            # Multiple altitude rows keep the larger fleet out of near-contact
            # at initialization in the default arena; arbitrary arenas may not fit.
            cols = min(7, n)
            rows = int(np.ceil(n/cols))
            margin = min(6., .2*(config.altitude_max-config.altitude_min))
            for i in range(n):
                self.chaser_pos[i, 1] = np.linspace(.12, .88, cols)[i % cols]*config.depth
                self.chaser_pos[i, 2] = np.linspace(config.altitude_min+margin, config.altitude_max-margin, rows)[i//cols]
        self.chaser_vel = np.tile([config.target_speed_min, 0., 0.], (n, 1))
        self.chaser_active = np.ones(n, dtype=bool)
        self.chaser_generation = np.zeros(n, dtype=int)
        self.base_positions = self.chaser_pos.copy()
        self.base_positions[:, 2] = config.ground_z+config.body_radius+.1
        if n > 7:
            # Ground pads use x/y rows, not the airborne fleet's y/z rows.
            spacing = config.min_separation+2.
            self.base_positions[:, 0] = 8.+spacing*(np.arange(n)//7)
        self._launch_at = np.full(n, np.inf)
        self.completed_observers = 0
        self.relaunches = 0
        self.lifecycle_events = deque(maxlen=128)
        self.events = self.lifecycle_events
        self.retirement_target = np.full(n, -1, dtype=int)
        self.retirement_times = np.full(n, np.nan)
        for j in range(config.n_targets):
            self._spawn(j, initial=True)
        self.previous_chaser_pos = self.chaser_pos.copy()
        self.previous_target_pos = self.target_pos.copy()
        self.world = PhysicsWorld(config, m)

    def target_index(self, target_id):
        """Resolve a live global ID to its current storage slot, or None."""
        slots = np.flatnonzero(self.target_present & (self.target_ids == target_id))
        return int(slots[0]) if slots.size else None

    def set_observation_assignments(self, mapping):
        """Set mission metadata, separately from the physical acceleration command.

        None restores nearest-track fixture semantics; an empty mapping idles all
        observers. Unknown/stale target IDs are treated as unavailable in step.
        The mission reservation layer owns the per-target capacity restriction.
        """
        if mapping is None:
            self.observation_assignments = None
        else:
            normalized = {}
            for craft, target in mapping.items():
                if isinstance(craft, bool) or not isinstance(craft, (int, np.integer)) or not 0 <= craft < self.config.n_chasers:
                    raise ValueError('unknown observer ID')
                if target is not None and (isinstance(target, bool) or not isinstance(target, (int, np.integer)) or target < 0):
                    raise ValueError('target ID must be a nonnegative integer or None')
                normalized[int(craft)] = None if target is None else int(target)
            self.observation_assignments = normalized
        self._refresh_reservations()

    def _refresh_reservations(self):
        self.target_reservations.fill(0)
        for craft, target in (self.observation_assignments or {}).items():
            if not self.chaser_active[craft]:
                continue
            slot = self.target_index(target) if target is not None else None
            if slot is not None and self.status[slot] == ACTIVE:
                self.target_reservations[slot] += 1

    def _spawn(self, slot, *, initial=False):
        cfg = self.config
        identity = self._next_target_id
        self._next_target_id += 1
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 11, identity]))
        span = cfg.altitude_max-cfg.altitude_min
        fraction = slot/max(1, cfg.n_targets-1)
        x = (.28+.12*fraction)*cfg.width if initial else -12.
        y = (.12+.76*fraction)*cfg.depth if initial else rng.uniform(.12, .88)*cfg.depth
        z = rng.uniform(cfg.altitude_min+.1*span, cfg.altitude_max-.1*span)
        if initial and cfg.n_targets > 7:
            cols, rows = 7, int(np.ceil(cfg.n_targets/7))
            margin = min(6., .2*span)
            y = np.linspace(.12, .88, cols)[slot % cols]*cfg.depth
            z = np.linspace(cfg.altitude_min+margin, cfg.altitude_max-margin, rows)[slot//cols]
        self.target_pos[slot] = [x, y, z]
        self._spawn_pos[slot] = self.target_pos[slot]
        profile = TRAJECTORIES[1+identity % (len(TRAJECTORIES)-1)] if cfg.trajectory == 'mixed' else cfg.trajectory
        self.trajectory_names[slot] = profile
        self._target_speeds[slot] = rng.uniform(*cfg.speed_range(profile))
        self._turn_phases[slot] = rng.uniform(0., 2.*np.pi)
        self._birth_times[slot] = self.time
        self.target_vel[slot] = [self._target_speeds[slot], 0., 0.]
        self.target_present[slot] = True
        self.target_ids[slot] = identity
        self.status[slot] = ACTIVE
        self.target_event_pos[slot] = np.nan
        self.service_dwell[:, slot] = 0.
        self.service_agent[slot] = -1
        self._target_close[:, slot] = False
        self.target_reservations[slot] = 0
        self._arrivals += 1
        self._waypoint_cache.clear()

    def _manage_arrivals(self):
        cfg = self.config
        if not cfg.continuous:
            return
        offstage = self.target_present & (self.target_pos[:, 0] > cfg.width+cfg.offstage_margin)
        for slot in np.flatnonzero(offstage):
            if self.status[slot] == ACTIVE:
                self._escaped += 1
            self.target_present[slot] = False
            self.status[slot] = -1
            self.service_dwell[:, slot] = 0.
            self._target_close[:, slot] = False
        while self.time >= self._next_arrival-1e-10:
            free = np.flatnonzero(~self.target_present)
            if free.size:
                self._spawn(int(free[0]))
            else:
                self._dropped_arrivals += 1
            self._next_arrival += cfg.arrival_interval
        self._refresh_reservations()

    def _complete_observer(self, craft, slot, fraction, event_time):
        """Withdraw the safe-observation winner; this event is never a collision."""
        target_id = int(self.target_ids[slot])
        craft_position = self.previous_chaser_pos[craft]+fraction*(self.chaser_pos[craft]-self.previous_chaser_pos[craft])
        self.lifecycle_events.append(dict(
            type='completion', time=float(event_time), chaser_id=int(craft),
            generation=int(self.chaser_generation[craft]), target_id=target_id,
            position=tuple(float(x) for x in craft_position),
            target_position=tuple(float(x) for x in self.target_event_pos[slot])))
        self.chaser_active[craft] = False
        self.chaser_pos[craft] = craft_position
        self.chaser_vel[craft] = 0.
        self.retirement_target[craft] = target_id
        self.retirement_times[craft] = event_time
        self._launch_at[craft] = event_time+self.config.relaunch_delay
        self.service_dwell[craft] = 0.
        self.target_present[slot] = False
        self._target_close[craft] = False
        self._target_close[:, slot] = False
        self._peer_close[craft] = False
        self._peer_close[:, craft] = False
        self.world.reset_craft(craft, craft_position, active=False)
        self.world.remove_target(slot)
        if self.observation_assignments is not None:
            self.observation_assignments[craft] = None
        self.completed_observers += 1

    def _manage_launches(self):
        if not self.config.recycle_on_service:
            return
        clearance = self.config.min_separation+1.
        for craft in np.flatnonzero(~self.chaser_active & (self._launch_at <= self.time+1e-10)):
            pad = self.base_positions[craft]
            peers = self.chaser_pos[self.chaser_active]
            targets = self.target_pos[self.target_present]
            if (np.any(np.linalg.norm(peers-pad, axis=1) < clearance) or
                    np.any(np.linalg.norm(targets-pad, axis=1) < clearance)):
                continue
            self.chaser_generation[craft] += 1
            self.chaser_active[craft] = True
            self.chaser_pos[craft] = pad
            self.chaser_vel[craft] = 0.
            self.previous_chaser_pos[craft] = pad
            self._launch_at[craft] = np.inf
            self.retirement_target[craft] = -1
            self.retirement_times[craft] = np.nan
            self.service_dwell[craft] = 0.
            self._target_close[craft] = False
            self._peer_close[craft] = False
            self._peer_close[:, craft] = False
            if self.observation_assignments is not None:
                self.observation_assignments[craft] = None
            self.world.reset_craft(int(craft), pad)
            self.relaunches += 1
            self.lifecycle_events.append(dict(
                type='launch', time=float(self.time), chaser_id=int(craft),
                generation=int(self.chaser_generation[craft]), target_id=None,
                position=tuple(float(x) for x in pad)))

    @property
    def time(self):
        return self.tick*self.config.dt

    @property
    def done(self):
        expired = self.config.duration is not None and self.time >= self.config.duration-1e-10
        return expired or (not self.config.continuous and not np.any(self.target_present & (self.status == ACTIVE)))

    def observe(self, agent_id: int) -> Observation:
        """Return all physical tracks in range, plus target service availability.

        Present completed tracks remain visible for collision avoidance when
        recycling is disabled. Only target_available permits allocation. Noise is
        independently keyed by seed, tick and observer, never by read order.
        """
        cfg = self.config
        if not 0 <= agent_id < cfg.n_chasers:
            raise ValueError('unknown agent ID')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 51, self.tick, agent_id]))
        cp = self.chaser_pos+rng.normal(0., cfg.position_noise, self.chaser_pos.shape)
        cv = self.chaser_vel+rng.normal(0., cfg.velocity_noise, self.chaser_vel.shape)
        tp = self.target_pos+rng.normal(0., cfg.position_noise, self.target_pos.shape)
        tv = self.target_vel+rng.normal(0., cfg.velocity_noise, self.target_vel.shape)
        visible_peers = self.chaser_active.copy()
        visible_peers[agent_id] = True  # Own telemetry remains available while absent.
        ci = np.flatnonzero(visible_peers & (np.linalg.norm(self.chaser_pos-self.chaser_pos[agent_id], axis=1) <= cfg.sensor_range))
        ti = np.flatnonzero(self.target_present & (np.linalg.norm(self.target_pos-self.chaser_pos[agent_id], axis=1) <= cfg.sensor_range))
        return Observation(self.time, agent_id, ci, cp[ci], cv[ci], self.target_ids[ti].copy(), tp[ti], tv[ti],
                           active=bool(self.chaser_active[agent_id]), target_available=(self.status[ti] == ACTIVE))

    def _waypoint(self, index):
        if index not in self._waypoint_cache:
            cfg = self.config
            values = np.zeros((self.capacity, 2))
            span = cfg.altitude_max-cfg.altitude_min
            for j in np.flatnonzero(self.target_present):
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, 81, index, int(self.target_ids[j])]))
                values[j] = [rng.uniform(.12*cfg.depth, .88*cfg.depth), rng.uniform(cfg.altitude_min+.2*span, cfg.altitude_max-.2*span)]
            self._waypoint_cache[index] = values
            for old in list(self._waypoint_cache):
                if old < index-1:
                    del self._waypoint_cache[old]
        return self._waypoint_cache[index]

    def _advance_targets(self):
        """Advance controlled reference tracks, independent of mission outcomes.

        Random tracks use smooth waypoint steering and bounded disturbance.
        Other profiles prescribe tangent velocities: a single altitude arc,
        straight constant-height flight, a lateral weave, or a drifting orbit
        that transitions to outbound flight after one revolution. All profiles
        respect the speed/altitude envelope through velocity constraints.
        """
        cfg = self.config
        phase = self.time/cfg.waypoint_period
        index = int(np.floor(phase))
        blend = .5-.5*np.cos(np.pi*(phase-index))
        waypoint = (1.-blend)*self._waypoint(index)+blend*self._waypoint(index+1)
        span = cfg.altitude_max-cfg.altitude_min
        for j in np.flatnonzero(self.target_present):
            profile = self.trajectory_names[j]
            speed_min, speed_max = cfg.speed_range(profile)
            # Identity-keyed noise is unchanged if the same track occupies a
            # different pool slot; sensor reads and service never consume it.
            track_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 43, self.tick, int(self.target_ids[j])]))
            jitter = track_rng.normal(0., cfg.drift_std*np.sqrt(cfg.dt), 3)
            speed, age = self._target_speeds[j], self.time-self._birth_times[j]
            position = self.target_pos[j]
            previous_velocity = self.target_vel[j].copy()
            velocity = previous_velocity.copy()
            if profile == 'random':
                direction = np.r_[25., waypoint[j]-position[1:]]
                desired = direction/np.linalg.norm(direction)*speed
                steering = limit_norm((desired-velocity)/2., cfg.turn_accel)
                steering[1] += cfg.turn_accel*np.sin(cfg.turn_frequency*self.time+self._turn_phases[j])
                velocity += steering*cfg.dt+jitter
            elif profile == 'constant-height':
                velocity[:] = [speed, 0., 0.]
            elif profile == 'parabolic':
                # Arc starts and ends at spawn altitude; it remains controlled flight.
                length = max(cfg.width-self._spawn_pos[j, 0], 1.)
                u = np.clip((position[0]-self._spawn_pos[j, 0])/length, 0., 1.)
                amplitude = .75*(cfg.altitude_max-self._spawn_pos[j, 2])
                slope = 4.*amplitude*(1.-2.*u)/length if position[0] < cfg.width else 0.
                direction = np.array([1., 0., slope])
                velocity[:] = direction/np.linalg.norm(direction)*speed
            elif profile == 'weave':
                slope = .65*np.sin(.55*age+self._turn_phases[j])
                direction = np.array([1., slope, 0.])
                velocity[:] = direction/np.linalg.norm(direction)*speed
            else:  # orbit: prescribed horizontal circle with an advancing center
                omega = .4
                if age < 2.*np.pi/omega:
                    tangent = .5*(speed_min+speed_max)
                    drift = .4*(speed_max-speed_min)
                    angle = omega*age+self._turn_phases[j]
                    velocity[:] = [drift+tangent*np.cos(angle), tangent*np.sin(angle), 0.]
                else:
                    # A finite orbit does not permanently occupy an arrival slot.
                    velocity[:] = [speed, 0., 0.]
            if profile != 'orbit':
                velocity[1:] = limit_norm(velocity[1:], .8*speed_max)
                if cfg.altitude_min <= position[2] <= cfg.altitude_max:
                    velocity[2] = np.clip(velocity[2], (cfg.altitude_min-position[2])/cfg.dt,
                                          (cfg.altitude_max-position[2])/cfg.dt)
                lateral_squared = np.sum(velocity[1:]**2)
                lower = max(.2*speed_min, np.sqrt(max(0., speed_min**2-lateral_squared)))
                upper = np.sqrt(max(0., speed_max**2-lateral_squared))
                velocity[0] = np.clip(velocity[0], lower, upper)
            else:
                magnitude = np.linalg.norm(velocity)
                velocity *= np.clip(magnitude, speed_min, speed_max)/max(magnitude, 1e-12)
            # Constant sampled air-track speed with a turn angle whose chord
            # bounds |delta-v| by a_max*dt. This avoids birth and orbit-exit jumps.
            old_norm = np.linalg.norm(previous_velocity)
            old_direction = previous_velocity/old_norm if old_norm > 1e-12 else np.array([1., 0., 0.])
            desired_direction = velocity/max(np.linalg.norm(velocity), 1e-12)
            # Begin altitude braking well before the hard envelope. When the
            # remaining height is inside a conservative stopping distance, use
            # the full turn budget to flatten vertical motion first.
            vz = previous_velocity[2]
            clearance = cfg.altitude_max-position[2] if vz > 0. else position[2]-cfg.altitude_min
            if vz != 0. and clearance <= vz*vz/(.6*cfg.target_accel_max)+2.*abs(vz)*cfg.dt:
                horizontal = previous_velocity.copy()
                horizontal[2] = 0.
                desired_direction = horizontal/max(np.linalg.norm(horizontal), 1e-12)
            cosine = np.clip(np.dot(old_direction, desired_direction), -1., 1.)
            angle = np.arccos(cosine)
            maximum_angle = 2.*np.arcsin(min(1., cfg.target_accel_max*cfg.dt/(2.*speed)))
            turn = min(angle, maximum_angle)
            orthogonal = desired_direction-cosine*old_direction
            length = np.linalg.norm(orthogonal)
            if length < 1e-10:
                orthogonal = np.cross(old_direction, [0., 0., 1.])
                length = np.linalg.norm(orthogonal)
                if length < 1e-10:
                    orthogonal = np.array([1., 0., 0.])
                    length = 1.
            velocity = speed*(np.cos(turn)*old_direction+np.sin(turn)*orthogonal/length)
            # Exact constant-height trajectories exclude all vertical noise.
            if profile in ('constant-height', 'weave', 'orbit'):
                velocity[2] = 0.
                velocity *= speed/max(np.linalg.norm(velocity), 1e-12)
            # The velocity envelope is a prescribed-track boundary constraint,
            # never a craft teleport or a service/separation correction.
            if cfg.altitude_min <= position[2] <= cfg.altitude_max:
                vertical = np.clip(velocity[2], (cfg.altitude_min-position[2])/cfg.dt,
                                   (cfg.altitude_max-position[2])/cfg.dt)
                if vertical != velocity[2]:
                    horizontal_length = np.linalg.norm(velocity[:2])
                    velocity[:2] *= np.sqrt(max(0., speed*speed-vertical*vertical))/max(horizontal_length, 1e-12)
                    velocity[2] = vertical
            self.target_vel[j] = velocity
            self.target_pos[j] += velocity*cfg.dt

    def _record_separation(self):
        cfg = self.config
        def record(start, end, previous_close, mask=None):
            delta = end-start
            fraction = np.clip(-np.sum(start*delta, axis=-1)/np.maximum(np.sum(delta*delta, axis=-1), 1e-24), 0., 1.)
            minimum = np.linalg.norm(start+fraction[..., None]*delta, axis=-1)
            crossed = minimum < cfg.min_separation
            if mask is not None:
                crossed &= mask
                values = minimum[mask]
            else:
                values = minimum.reshape(-1)
            self.near_contact_violations += int(np.count_nonzero(crossed & ~previous_close))
            close_now = np.linalg.norm(end, axis=-1) < cfg.min_separation
            if mask is not None:
                close_now &= mask
            return float(values.min()) if values.size else np.inf, close_now
        start = self.previous_chaser_pos[:, None]-self.previous_target_pos[None]
        end = self.chaser_pos[:, None]-self.target_pos[None]
        minimum, self._target_close = record(start, end, self._target_close,
                                             self.chaser_active[:, None] & self.target_present[None])
        self.minimum_target_separation = min(self.minimum_target_separation, minimum)
        start = self.previous_chaser_pos[:, None]-self.previous_chaser_pos[None]
        end = self.chaser_pos[:, None]-self.chaser_pos[None]
        minimum, self._peer_close = record(start, end, self._peer_close,
                                           np.triu(self.chaser_active[:, None] & self.chaser_active[None], 1))
        self.minimum_peer_separation = min(self.minimum_peer_separation, minimum)

    def step(self, commands: np.ndarray):
        """Advance one tick; count continuous observation before +x escape."""
        cfg = self.config
        commands = np.asarray(commands, dtype=float)
        if commands.shape != (cfg.n_chasers, 3) or not np.isfinite(commands).all():
            raise ValueError(f'commands must be a finite ({cfg.n_chasers}, 3) array')
        self._manage_arrivals()
        old_generation = self.chaser_generation.copy()
        self._manage_launches()
        if np.any(self.chaser_generation != old_generation):
            # The caller computed these commands for the old incarnation.
            commands = commands.copy()
            commands[self.chaser_generation != old_generation] = 0.
        # Birth/recycling happens before the sweep snapshot: no teleport segment.
        self.previous_chaser_pos = self.chaser_pos.copy()
        self.previous_target_pos = self.target_pos.copy()
        self._advance_targets()
        self.chaser_pos, self.chaser_vel = self.world.step(
            self.chaser_pos, self.chaser_vel, commands, self.target_pos,
            self.target_present, target_start=self.previous_target_pos,
            chaser_active=self.chaser_active)
        self._record_separation()
        deadline = np.ones(self.capacity)
        events = []
        for j in np.flatnonzero(self.target_present & (self.status == ACTIVE)):
            x0, x1 = self.previous_target_pos[j, 0], self.target_pos[j, 0]
            if x0 >= cfg.width:
                deadline[j] = 0.
                events.append((0., 0, int(j), -1))
            elif x1 >= cfg.width:
                deadline[j] = (cfg.width-x0)/(x1-x0)
                events.append((deadline[j], 0, int(j), -1))
        previous_dwell = self.service_dwell.copy()
        self.service_dwell.fill(0.)
        for i in np.flatnonzero(self.chaser_active):
            candidates = []
            assigned = None if self.observation_assignments is None else self.observation_assignments.get(i)
            for j in np.flatnonzero(self.target_present & (self.status == ACTIVE)):
                if self.observation_assignments is not None and assigned != self.target_ids[j]:
                    continue
                r0 = self.previous_chaser_pos[i]-self.previous_target_pos[j]
                delta = self.chaser_pos[i]-self.target_pos[j]-r0
                intervals = _safe_intervals(r0, delta, cfg.min_separation, cfg.r_dock, deadline[j])
                if intervals:
                    midpoint = .5*sum(intervals[0])
                    candidates.append((float(np.linalg.norm(r0+midpoint*delta)), int(j), intervals))
            if not candidates:
                continue
            _, j, intervals = min(candidates, key=lambda item: (item[0], item[1]))
            start_distance = np.linalg.norm(self.previous_chaser_pos[i]-self.previous_target_pos[j])
            end_distance = np.linalg.norm(self.chaser_pos[i]-self.target_pos[j])
            start_in_band = cfg.min_separation <= start_distance < cfg.r_dock
            end_in_band = cfg.min_separation <= end_distance < cfg.r_dock
            dwell = 0.
            for lo, hi in intervals:
                dwell = previous_dwell[i, j] if lo <= 1e-12 and start_in_band else 0.
                duration = (hi-lo)*cfg.dt
                if dwell+duration >= cfg.service_duration-1e-12:
                    fraction = lo+max(0., cfg.service_duration-dwell)/cfg.dt
                    events.append((min(fraction, hi), 1, j, i))
                    break
                dwell += duration
                self.service_dwell[i, j] = dwell if hi >= 1.-1e-12 and end_in_band else 0.
        for fraction, kind, j, agent in sorted(events):
            if self.status[j] != ACTIVE:
                continue
            self.status[j] = SERVICED if kind else ESCAPED
            self.target_event_pos[j] = self.previous_target_pos[j]+fraction*(self.target_pos[j]-self.previous_target_pos[j])
            self.service_dwell[:, j] = 0.
            if kind:
                self.service_agent[j] = agent
                event_time = self.time+fraction*cfg.dt
                self.service_times.append(event_time)
                # Service latency is measured from this object's admission,
                # not the age of a continuously running session.
                self._service_time_sum += event_time-self._birth_times[j]
                self._serviced += 1
                if cfg.recycle_on_service:
                    self._complete_observer(agent, j, fraction, event_time)
            else:
                self._escaped += 1
        self._refresh_reservations()
        self.tick += 1

    def stats(self):
        serviced = self._serviced
        def finite_or_none(value):
            return float(value) if np.isfinite(value) else None
        return dict(elapsed=round(self.time, 4), serviced=serviced,
                    unserviced=self._arrivals-serviced, escaped=self._escaped,
                    arrivals=self._arrivals, dropped_arrivals=self._dropped_arrivals,
                    present=int(np.count_nonzero(self.target_present)),
                    pending=int(np.count_nonzero(self.target_present & (self.status == ACTIVE))),
                    reserved=int(np.count_nonzero(self.target_reservations)),
                    physics_contacts=self.world.contact_count,
                    ground_contact_observations=self.world.ground_contact_observations,
                    body_contact_observations=self.world.body_contact_observations,
                    active_chasers=int(np.count_nonzero(self.chaser_active)),
                    retired_chasers=int(np.count_nonzero(~self.chaser_active)),
                    completed_observers=self.completed_observers, relaunches=self.relaunches,
                    departures=self.completed_observers, launches=self.relaunches,
                    redundant_consumptions=0,
                    minimum_separation=finite_or_none(min(self.minimum_target_separation, self.minimum_peer_separation)),
                    minimum_target_separation=finite_or_none(self.minimum_target_separation),
                    minimum_peer_separation=finite_or_none(self.minimum_peer_separation),
                    near_contact_violations=self.near_contact_violations,
                    mean_service_time=self._service_time_sum/serviced if serviced else None)
