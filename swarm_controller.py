"""Local 3-D guidance with explicit simulated peer-broadcast deconfliction."""
from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING
import numpy as np
import torch
from estimator import Observation, estimate_pairs

if TYPE_CHECKING:
    from sim_engine import Config


@dataclass(frozen=True)
class PeerBroadcast:
    """One craft's immutable announcement; capability is its own local score."""
    sender_id: int
    epoch: int
    round_index: int
    time: float
    target_id: int | None
    capability: float
    active: bool = True


def coverage_loss(x: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
    """Diagnostic sum_j product_i (1-X_ij*P_ij), including unassigned targets.

    Hard choices come from own commitment and received current peer packets.
    This local independent-event surrogate is not a global matching objective.
    """
    return torch.prod(1.-x*probability, dim=0).sum()


def _track_ids(values, name: str, upper: int | None = None) -> np.ndarray:
    ids = np.asarray(values)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or
            np.any(ids < 0) or len(np.unique(ids)) != len(ids) or
            (upper is not None and np.any(ids >= upper))):
        raise ValueError(f'{name} must be unique nonnegative integer track IDs in the permitted range')
    return ids


def available_targets(obs: Observation) -> np.ndarray:
    """Mission eligibility is separate from physical tracks used for safety."""
    values = getattr(obs, 'target_available', None)
    mask = np.ones(len(obs.target_ids), dtype=bool) if values is None else np.asarray(values, dtype=bool)
    if mask.shape != (len(obs.target_ids),):
        raise ValueError('target eligibility must match observed target tracks')
    return mask


def proportional_navigation(r, v_rel, n_nav: float, a_max: float) -> np.ndarray:
    """3-D PN: a = -N*Vc*cross(r_hat, cross(r,v_rel)/||r||^2).

    Inputs are matching finite (..., 3) relative vectors. Vc is signed closing
    speed; the controller handles acquisition/nonclosing states separately.
    Coincident points return zero and the acceleration vector norm is capped.
    """
    r, v_rel = np.asarray(r, dtype=float), np.asarray(v_rel, dtype=float)
    if (r.ndim < 1 or r.shape[-1] != 3 or r.shape != v_rel.shape or
            not np.isfinite(r).all() or not np.isfinite(v_rel).all()):
        raise ValueError('relative states must be matching finite (..., 3) arrays')
    if not np.isfinite([n_nav, a_max]).all() or n_nav <= 0 or a_max <= 0:
        raise ValueError('n_nav and a_max must be finite and positive')
    distance = np.linalg.norm(r, axis=-1, keepdims=True)
    los = r/np.maximum(distance, 1e-9)
    closing = -np.sum(los*v_rel, axis=-1, keepdims=True)
    omega = np.cross(r, v_rel)/np.maximum(distance**2, 1e-12)
    command = -n_nav*closing*np.cross(los, omega)
    return command*np.minimum(1., a_max/np.maximum(np.linalg.norm(command, axis=-1, keepdims=True), 1e-12))


class SwarmController:
    """One craft uses local tracks and its own received peer-message buffer.

    During an auction epoch every craft proposes one locally ranked target.
    A target's highest transmitted capability wins, with lower sender ID as the
    deterministic tie-break. Losers reject that target for this epoch and try
    their next option. The harness must deliver all round packets within each
    connected component before resolving, then finalize only after a round in
    which no proposal changes. Actual target_id stays None during negotiation.

    GNN mode uses one learned forward pass and its own row argmax, without an
    auction or an exclusivity guarantee. Auction is decentralized deconfliction over reliable synchronous broadcasts,
    not optimal matching. Disconnected components cannot exclude each other's
    unseen claims. Greedy broadcasts its nearest-target choice for observability
    but ignores peer conflicts, preserving the comparison baseline.
    """
    def __init__(self, agent_id: int, config: 'Config', mode: str = 'auction', model_path=None):
        mode = 'auction' if mode == 'coordinated' else mode
        if mode not in ('greedy', 'auction', 'gnn'):
            raise ValueError('mode must be greedy, auction or gnn')
        if not isinstance(agent_id, (int, np.integer)) or not 0 <= agent_id < config.n_chasers:
            raise ValueError('agent_id must belong to the initial chaser roster')
        self.agent_id, self.config, self.mode = agent_id, config, mode
        self.obs: Observation | None = None
        self.target_id: int | None = None
        self.allocation_loss = 0.
        self.estimates = None
        self.peer_buffer: dict[int, PeerBroadcast] = {}
        self.retired_peer_ids: set[int] = set()
        self._last_peer_rounds: dict[int, int] = {}
        self._last_allocation = -np.inf
        self._signature = None
        self._needs_allocation = True
        self._retired = False
        self._epoch = 0
        self._round_index = -1
        self._allocation_started = False
        self._negotiating = False
        self._proposal_id: int | None = None
        self._proposal_capability = 0.
        self._options: list[int] = []
        self._capabilities: dict[int, float] = {}
        self._rejected: set[int] = set()
        self.gnn_forward_calls = 0
        self.gnn_forward_seconds = 0.
        self.gnn_compute_seconds = 0.
        self.gnn_last_logits = None
        self._gnn_scores: dict[int, float] = {}
        self.reservation_slot = 0
        self.reservation_capacity = 1
        self.model = None
        if self.mode == 'gnn':
            from gnn_model import load_policy
            path = Path(model_path) if model_path is not None else Path(__file__).parent/'output'/'swarm_gnn_model.pt'
            if not path.is_file():
                raise FileNotFoundError(f'GNN checkpoint missing: {path}. Train it with: .venv/bin/python train_gnn.py')
            self.model = load_policy(path)

    def reset_for_launch(self, *, epoch: int | None = None) -> None:
        """Start a fresh observer in this fixed bay after a trusted launch event.

        The harness calls this when the plant's bay generation changes, before
        supplying fresh telemetry. Learned weights and cumulative inference
        metrics survive; sensor tracks, claims, and old incarnation caches do
        not. The optional epoch joins the harness's current transport clock:
        advancing to a newer epoch starts its round counter again, while an
        unchanged epoch preserves round ordering. Epochs cannot move backward.
        """
        if epoch is not None:
            if (not isinstance(epoch, (int, np.integer)) or isinstance(epoch, (bool, np.bool_))
                    or epoch < self._epoch):
                raise ValueError('launch epoch must be an integer at or after the current epoch')
            if epoch > self._epoch:
                self._epoch, self._round_index = int(epoch), -1
                self._allocation_started = True
        self.obs, self.estimates, self.target_id = None, None, None
        self.allocation_loss = 0.
        self.peer_buffer.clear()
        self.retired_peer_ids.clear()
        self._last_peer_rounds.clear()
        self._last_allocation = -np.inf
        self._signature = None
        self._needs_allocation, self._retired, self._negotiating = True, False, False
        self._proposal_id, self._proposal_capability = None, 0.
        self._options.clear()
        self._capabilities.clear()
        self._rejected.clear()
        self.gnn_last_logits = None
        self._gnn_scores.clear()
        self.reservation_slot, self.reservation_capacity = 0, 1

    def forget_peer_incarnation(self, peer_id: int) -> None:
        """Forget an old bay occupant after the harness confirms its relaunch.

        Ordinary active radio packets cannot undo retirement. Only this trusted
        plant lifecycle notification clears the peer tombstone and packet order
        cache; reliable synchronous transport supplies the new incarnation.
        """
        if (not isinstance(peer_id, (int, np.integer)) or isinstance(peer_id, (bool, np.bool_))
                or not 0 <= peer_id < self.config.n_chasers):
            raise ValueError('peer_id must belong to the observer bay roster')
        self.peer_buffer.pop(peer_id, None)
        self._last_peer_rounds.pop(peer_id, None)
        self.retired_peer_ids.discard(peer_id)
        self._needs_allocation = True
        self._refresh_loss()

    @property
    def allocation_due(self) -> bool:
        """Whether current telemetry calls for a new coordinated epoch."""
        return bool(self.obs is not None and self.obs.active and not self._retired
                    and not self._negotiating and (self._needs_allocation or
                    self.obs.time-self._last_allocation >= self.config.allocation_period-1e-9))

    def _expire_peers(self) -> None:
        if self.obs is None:
            return
        for sender_id, message in list(self.peer_buffer.items()):
            if message.epoch != self._epoch or self.obs.time-message.time > self.config.broadcast_ttl:
                del self.peer_buffer[sender_id]
                self._needs_allocation = True

    def _capability(self, target_id: int | None) -> float:
        if target_id is None or self.estimates is None or self.obs is None:
            return 0.
        rows = np.flatnonzero(self.obs.chaser_ids == self.agent_id)
        cols = np.flatnonzero((self.obs.target_ids == target_id) & available_targets(self.obs))
        if not len(rows) or not len(cols):
            return 0.
        row, col = int(rows[0]), int(cols[0])
        score = self.estimates.probability[row, col]/(
            1.+self.estimates.distance[row, col]/self.config.cruise_speed)
        return float(np.clip(score, 0., 1.))

    def _refresh_loss(self) -> None:
        if self.estimates is None or self.obs is None or self._retired:
            self.allocation_loss = 0.
            return
        obs = self.obs
        eligible = available_targets(obs)
        probability = self.estimates.probability[:, eligible]
        x = np.zeros_like(probability)
        rows = {int(value): index for index, value in enumerate(obs.chaser_ids)}
        cols = {int(value): index for index, value in enumerate(obs.target_ids[eligible])}
        choices = {sender: message.target_id for sender, message in self.peer_buffer.items()
                   if message.active and message.epoch == self._epoch and
                   obs.time-message.time <= self.config.broadcast_ttl}
        choices[self.agent_id] = self.target_id
        for sender_id, target_id in choices.items():
            if sender_id in rows and target_id in cols:
                x[rows[sender_id], cols[target_id]] = 1.
        self.allocation_loss = float(coverage_loss(
            torch.as_tensor(x), torch.as_tensor(probability)))

    def update_telemetry(self, obs: Observation) -> None:
        """Refresh local estimates; coordinated choices require explicit rounds."""
        if obs.agent_id != self.agent_id:
            raise ValueError('packet must belong to this craft')
        if not np.isfinite(obs.time) or obs.time < 0:
            raise ValueError('observation time must be finite and nonnegative')
        self.obs = obs
        if not obs.active or self._retired:
            self._retired = True
            self.target_id, self.allocation_loss = None, 0.
            self.estimates, self._signature = None, None
            self._proposal_id, self._proposal_capability = None, 0.
            self.peer_buffer.clear()
            self._last_peer_rounds.clear()
            self._negotiating, self._needs_allocation = False, False
            return
        cfg = self.config
        ci = _track_ids(obs.chaser_ids, 'chaser_ids', cfg.n_chasers)
        # Object IDs are monotonic across continuous spawning; n_targets is
        # only the initial population, not an upper bound on an object's ID.
        ti = _track_ids(obs.target_ids, 'target_ids')
        if self.agent_id not in ci:
            raise ValueError('active packet must contain this craft')
        self.estimates = estimate_pairs(obs.chaser_pos, obs.chaser_vel,
            obs.target_pos, obs.target_vel, r_dock=cfg.r_dock,
            sigma_base=cfg.sigma_base, zem_scale=cfg.zem_scale, ground_z=cfg.ground_z,
            min_separation=cfg.min_separation, perimeter_x=cfg.width)
        if self.estimates.distance.shape != (len(ci), len(ti)):
            raise ValueError('state arrays and track IDs must agree')
        eligible = available_targets(obs)
        target_ids = ti[eligible]
        signature = (tuple(ci), tuple(target_ids))
        self._needs_allocation |= signature != self._signature
        self._signature = signature
        self._expire_peers()
        if self.target_id is not None and self.target_id not in target_ids:
            self.target_id = None
            self._needs_allocation = True
        if self.mode in ('greedy', 'gnn') and self.allocation_due:
            row = int(np.flatnonzero(ci == self.agent_id)[0])
            if not len(target_ids):
                self.target_id = None
            elif self.mode == 'greedy':
                self.target_id = int(target_ids[np.argmin(self.estimates.distance[row, eligible])])
            else:
                from gnn_model import build_sensor_graph
                started = time.perf_counter()
                features, adjacency, _ = build_sensor_graph(obs, cfg)
                features = torch.as_tensor(features, dtype=torch.float32)
                adjacency = torch.as_tensor(adjacency, dtype=torch.bool)
                mask = torch.as_tensor(np.broadcast_to(eligible, (len(ci), len(ti))).copy(), dtype=torch.bool)
                forward_started = time.perf_counter()
                with torch.inference_mode():
                    logits = self.model(features, adjacency, mask)
                self.gnn_forward_seconds += time.perf_counter()-forward_started
                self.gnn_forward_calls += 1
                if logits.shape != (len(ci), len(ti)) or not torch.isfinite(logits).all():
                    raise ValueError('GNN policy must return finite logits for each sensed craft and target')
                self.gnn_last_logits = logits.detach().cpu().numpy().copy()
                self._gnn_scores = {int(target_id): float(self.gnn_last_logits[row, col])
                                    for col, target_id in enumerate(ti) if eligible[col]}
                self.target_id = int(ti[int(torch.argmax(logits[row].masked_fill(~mask[row], -torch.inf)).item())])
                self.gnn_compute_seconds += time.perf_counter()-started
            self._last_allocation, self._needs_allocation = obs.time, False
        self._refresh_loss()

    def reservation_candidates(self) -> list[tuple[int, float, float]]:
        """Return local eligible (object ID, policy score, distance) candidates.

        The simulator's explicit mission reservation authority consumes these
        values without querying plant truth or running another model forward.
        GNN scores are cached by stable object ID, so changing physical-track
        columns cannot accidentally transfer a logit to a newly spawned object.
        Zero reachability remains a pursuit candidate, since it is a heuristic
        rather than a proof that a visible observation assignment is infeasible.
        """
        if self.obs is None or self.estimates is None or self._retired or not self.obs.active:
            return []
        row = int(np.flatnonzero(self.obs.chaser_ids == self.agent_id)[0])
        candidates = []
        for col in np.flatnonzero(available_targets(self.obs)):
            target_id = int(self.obs.target_ids[col])
            distance = float(self.estimates.distance[row, col])
            if self.mode == 'gnn':
                if target_id not in self._gnn_scores:
                    continue
                score = self._gnn_scores[target_id]
            elif self.mode == 'greedy':
                score = -distance
            else:
                score = self._capability(target_id)
            if np.isfinite(score) and np.isfinite(distance):
                candidates.append((target_id, float(score), distance))
        return sorted(candidates, key=lambda item: (-item[1], item[2], item[0]))

    def accept_reservation(self, target_id: int | None, *, slot: int = 0, capacity: int = 1) -> None:
        """Accept the simulator mission authority's explicit capacity decision.

        Standalone controllers still use their raw policies. Scenario invokes
        this after those proposals to enforce the configured global capacity;
        it is not a decentralized uniqueness guarantee or a GNN auction.
        """
        if (not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1
                or not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < capacity):
            raise ValueError('reservation slot must belong to a positive integer capacity')
        if target_id is not None:
            if (not isinstance(target_id, (int, np.integer)) or isinstance(target_id, (bool, np.bool_))
                    or target_id < 0 or target_id not in {item[0] for item in self.reservation_candidates()}):
                raise ValueError('reservation must refer to a currently eligible local object')
            target_id = int(target_id)
        self.target_id = target_id
        self.reservation_slot, self.reservation_capacity = slot, capacity
        self._refresh_loss()

    def begin_allocation(self, epoch: int) -> None:
        """Stage local preferences, clearing commitment until agreement completes."""
        if self.obs is None:
            raise RuntimeError('update_telemetry must precede allocation')
        if (not isinstance(epoch, (int, np.integer)) or isinstance(epoch, (bool, np.bool_))
                or epoch < 0 or (self._allocation_started and epoch <= self._epoch)):
            raise ValueError('allocation epochs must be nonnegative and strictly increasing')
        if self.mode != 'auction':
            raise RuntimeError('only auction mode negotiates allocations')
        previous = self.target_id
        self.target_id = None
        self._epoch, self._round_index = int(epoch), -1
        self._allocation_started = True
        self.peer_buffer.clear()
        self._last_peer_rounds.clear()
        self._rejected.clear()
        self._options, self._capabilities = [], {}
        self._proposal_id, self._proposal_capability = None, 0.
        self._negotiating = bool(self.obs.active and not self._retired)
        if self._negotiating:
            row = int(np.flatnonzero(self.obs.chaser_ids == self.agent_id)[0])
            eligible = available_targets(self.obs)
            target_ids = self.obs.target_ids[eligible]
            distance = self.estimates.distance[row, eligible]
            scores = self.estimates.probability[row, eligible]/(1.+distance/self.config.cruise_speed)
            ranked = np.lexsort((target_ids, distance, -scores))
            self._options = [int(target_ids[index]) for index in ranked]
            self._capabilities = {int(target_id): float(np.clip(score, 0., 1.))
                                  for target_id, score in zip(target_ids, scores)}
            if previous in self._options:
                self._options.remove(previous)
                self._options.insert(0, previous)
            self._stage_next()
        self._refresh_loss()

    def _stage_next(self) -> None:
        self._proposal_id = next((target_id for target_id in self._options
                                  if target_id not in self._rejected), None)
        self._proposal_capability = self._capabilities.get(self._proposal_id, 0.)

    def make_broadcast(self, round_index: int) -> PeerBroadcast:
        """Snapshot own choice; retirement always emits a zero-capability tombstone."""
        if self.obs is None:
            raise RuntimeError('update_telemetry must precede broadcast')
        if (not isinstance(round_index, (int, np.integer)) or
                isinstance(round_index, (bool, np.bool_)) or round_index < self._round_index
                or round_index < 0):
            raise ValueError('broadcast rounds must be nonnegative and nondecreasing')
        self._round_index = int(round_index)
        active = bool(self.obs.active and not self._retired)
        target_id = self._proposal_id if self._negotiating else self.target_id
        capability = self._proposal_capability if self._negotiating else self._capability(target_id)
        return PeerBroadcast(self.agent_id, self._epoch, self._round_index,
            float(self.obs.time), target_id if active else None,
            capability if active and target_id is not None else 0., active)

    def receive_broadcast(self, message: PeerBroadcast) -> None:
        """Accept fresh per-sender packets; malformed or stale packets are ignored.

        A target=None/zero-capability active packet withdraws a claim but keeps
        the peer observable. A valid inactive packet removes the buffered peer
        and permanently tombstones its ID for this inactive craft session.
        Epoch and TTL checks also apply to retirement packets.
        """
        if self.obs is None or self._retired or not isinstance(message, PeerBroadcast):
            return
        integers = (message.sender_id, message.epoch, message.round_index)
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
               for value in integers):
            return
        if (message.sender_id == self.agent_id or not 0 <= message.sender_id < self.config.n_chasers
                or message.epoch != self._epoch or message.round_index < 0
                or message.round_index <= self._last_peer_rounds.get(message.sender_id, -1)
                or not isinstance(message.active, (bool, np.bool_))):
            return
        try:
            if (not np.isfinite(message.time) or not np.isfinite(message.capability)
                    or message.time < 0 or message.time > self.obs.time+1e-9
                    or self.obs.time-message.time > self.config.broadcast_ttl
                    or not 0. <= message.capability <= 1.):
                return
        except (TypeError, ValueError):
            return
        target_id = message.target_id
        if target_id is not None and (not isinstance(target_id, (int, np.integer))
                or isinstance(target_id, (bool, np.bool_)) or target_id < 0):
            return
        if ((target_id is None and message.capability != 0.) or
                (not message.active and (target_id is not None or message.capability != 0.))):
            return
        if message.sender_id in self.retired_peer_ids:
            return
        self._last_peer_rounds[message.sender_id] = message.round_index
        if not message.active:
            self.retired_peer_ids.add(message.sender_id)
            self.peer_buffer.pop(message.sender_id, None)
            self._needs_allocation = True
        else:
            self.peer_buffer[message.sender_id] = message
        self._expire_peers()
        self._refresh_loss()

    def _proposal_loses(self) -> bool:
        if self._proposal_id is None:
            return False
        own_priority = (self._proposal_capability, -self.agent_id)
        return any(message.target_id == self._proposal_id and
                   (message.capability, -message.sender_id) > own_priority
                   for message in self.peer_buffer.values())

    def resolve_round(self) -> bool:
        """Reject a losing local proposal and try the next target; winners hold."""
        if not self._negotiating or self._retired:
            return False
        self._expire_peers()
        if not self._proposal_loses():
            return False
        self._rejected.add(self._proposal_id)
        self._stage_next()
        return True

    def finalize_allocation(self) -> None:
        """Commit only this craft's winning proposal after a stable transport round."""
        if not self._negotiating or self._retired:
            return
        self._expire_peers()
        self.target_id = None if self._proposal_loses() else self._proposal_id
        self._negotiating = False
        self._last_allocation = self.obs.time
        self._needs_allocation = False
        self._refresh_loss()

    def get_control_commands(self) -> np.ndarray:
        """Track at standoff using acceleration commands, with local separation.

        The velocity reference follows observed target velocity plus a radial
        range servo toward stand_off_distance. PN contributes only a small,
        bounded lateral term outside the observation band. Every physical target
        and peer track imposes a predictive closing-speed/braking constraint,
        including serviced or unassigned targets. This reactive noisy-sensor
        safeguard cannot certify safety for infeasible initial conditions.
        """
        if self.obs is None:
            raise RuntimeError('update_telemetry must precede control')
        obs, cfg = self.obs, self.config
        if self._retired or not obs.active:
            return np.zeros(3)
        row = int(np.flatnonzero(obs.chaser_ids == self.agent_id)[0])
        position, velocity = obs.chaser_pos[row], obs.chaser_vel[row]
        desired_velocity = np.zeros(3)
        pn_augmentation = np.zeros(3)
        approach_speed = min(cfg.cruise_speed, cfg.v_max)
        if self.target_id is not None:
            matches = np.flatnonzero((obs.target_ids == self.target_id) & available_targets(obs))
            if len(matches):
                col = int(matches[0])
                relative = obs.target_pos[col]-position
                distance = float(np.linalg.norm(relative))
                los = relative/max(distance, 1e-9)
                if self.reservation_capacity > 1:
                    # Stable mission slots prevent multiple observers from
                    # sharing the same standoff point. Safety projection below
                    # still handles every peer and physical object, including
                    # cases where requested capacity exceeds geometric room.
                    clearance = cfg.min_separation+1.5+2.*cfg.position_noise
                    radius = max(cfg.stand_off_distance,
                                 clearance/(2.*np.sin(np.pi/self.reservation_capacity)))
                    radius = min(radius, (cfg.stand_off_distance+cfg.r_dock)/2.)
                    # Begin on opposite lateral sides for two observers of
                    # the usual +x route, avoiding a shared head-on approach.
                    angle = np.pi/2.+2.*np.pi*self.reservation_slot/self.reservation_capacity
                    offset = radius*np.array([np.cos(angle), np.sin(angle), 0.])
                    correction = 1.15*(relative+offset)
                    correction *= min(1., approach_speed/max(np.linalg.norm(correction), 1e-9))
                    desired_velocity = obs.target_vel[col]+correction
                else:
                    radial = np.clip(1.15*(distance-cfg.stand_off_distance),
                                     -approach_speed, approach_speed)
                    desired_velocity = obs.target_vel[col]+radial*los
                if distance > cfg.r_dock and self.reservation_capacity == 1:
                    blend = min(1., (distance-cfg.r_dock)/cfg.r_dock)
                    pn = proportional_navigation(relative, obs.target_vel[col]-velocity,
                                                 cfg.n_nav, cfg.a_max*.25)
                    pn_augmentation = .2*blend*(pn-np.dot(pn, los)*los)
        # A safety buffer covers sensor error and one update's response delay.
        safety_radius = cfg.min_separation+1.5+2.*cfg.position_noise
        peer_mask = obs.chaser_ids != self.agent_id
        obstacle_positions = np.concatenate((obs.target_pos, obs.chaser_pos[peer_mask]))
        obstacle_velocities = np.concatenate((obs.target_vel, obs.chaser_vel[peer_mask]))
        obstacles = []
        for other_position, other_velocity in zip(obstacle_positions, obstacle_velocities):
            relative = other_position-position
            distance = float(np.linalg.norm(relative))
            if distance > 1e-9:
                normal = relative/distance
            else:
                relative_velocity = velocity-other_velocity
                normal = relative_velocity/max(np.linalg.norm(relative_velocity), 1e-9)
                if np.linalg.norm(normal) < .5:
                    normal = np.array([1., 0., 0.])
            closing = float(np.dot(velocity-other_velocity, normal))
            obstacles.append((distance, normal, other_velocity, closing))
        # Project the velocity reference away from excessive relative closing.
        for _ in range(2):
            for distance, normal, other_velocity, closing in sorted(obstacles, key=lambda item: item[0]):
                margin = distance-safety_radius
                allowed_closing = (np.sqrt(2.*cfg.a_max*.65*margin) if margin > 0.
                                   else 2.*margin)
                excess = np.dot(desired_velocity-other_velocity, normal)-allowed_closing
                if excess > 0.:
                    desired_velocity -= excess*normal
        desired_velocity *= min(1., cfg.v_max/max(np.linalg.norm(desired_velocity), 1e-9))
        acceleration = 3.5*(desired_velocity-velocity)+pn_augmentation
        # When stopping distance reaches the protected radius, brake now rather
        # than waiting for a velocity reference to settle through the plant lag.
        for distance, normal, _, closing in sorted(obstacles, key=lambda item: item[0], reverse=True):
            margin = distance-safety_radius
            reaction_distance = max(closing, 0.)*(cfg.dt+getattr(cfg, 'thrust_tau', cfg.tau))
            stop_distance = max(closing, 0.)**2/(2.*cfg.a_max*.65)
            if margin <= reaction_distance+stop_distance:
                required = (-cfg.a_max if margin <= 0. else
                            -min(cfg.a_max, max(closing, 0.)**2/(2.*max(margin-reaction_distance, .1))))
                excess = float(np.dot(acceleration, normal))-required
                if excess > 0.:
                    acceleration -= excess*normal
        acceleration *= min(1., cfg.a_max/max(np.linalg.norm(acceleration), 1e-9))
        # MuJoCo applies the net acceleration through force/gravity compensation
        # and actuator lag. Do not invert the retired velocity-response model.
        return acceleration
