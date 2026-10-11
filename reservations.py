"""Explicit simulator mission reservations, separate from local policy inference.

This authority enforces a global per-object assignment capacity even when radio
components are disconnected. It uses only candidates exposed by each local
controller, never hidden object positions, plant statuses, or global matching
labels. It is centralized mission arbitration; raw GNN/greedy/auction policies
retain their own proposal behavior when used without this simulator service.
"""
from collections import Counter
import math
from numbers import Integral


class ReservationManager:
    """Keep valid incumbents, then arbitrate new proposals and local alternatives.

    Existing reservations remain while the owner locally observes the object as
    eligible. New conflicts favor larger policy score, then lower craft ID;
    each craft's tied alternatives favor shorter distance and lower object ID.
    Alternatives use the same deterministic
    order. This one-pass capacity rule is not an optimal matching solver and does
    not execute hidden model forwards or distributed auction rounds.
    """
    def __init__(self, capacity: int = 1):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self.capacity = capacity
        self._assignments: dict[int, int | None] = {}
        self._slots: dict[int, int] = {}
        self.peak_capacity_violations = 0

    @property
    def assignments(self) -> dict[int, int | None]:
        return self._assignments.copy()

    @property
    def slots(self) -> dict[int, int]:
        return self._slots.copy()

    def forget_agent(self, agent_id: int) -> None:
        """Release an old bay occupant's incumbent priority before relaunch.

        The new incarnation competes from its own fresh local candidates; it
        cannot inherit the former observer's reservation or observation slot.
        Repeated notifications for an already forgotten bay are harmless.
        """
        if not isinstance(agent_id, Integral) or isinstance(agent_id, bool) or agent_id < 0:
            raise ValueError('agent_id must be a nonnegative integer bay ID')
        self._assignments.pop(agent_id, None)
        self._slots.pop(agent_id, None)

    def assign(self, controllers) -> dict[int, int | None]:
        """Apply one capacity decision to all provided craft and return its map.

        Call after local policy updates/auction finalization and before issuing
        commands. Incoming target_id values are policy proposals; this manager's
        own previous map preserves service continuity across periodic reranking.
        A disappeared craft or unavailable/unobserved object releases its claim.
        """
        if not isinstance(self.capacity, int) or isinstance(self.capacity, bool) or self.capacity < 1:
            raise ValueError('capacity must remain a positive integer')
        by_id = {}
        candidates = {}
        for controller in controllers:
            agent_id = controller.agent_id
            if agent_id in by_id:
                raise ValueError('controller IDs must be unique')
            by_id[agent_id] = controller
            local = {}
            for target_id, score, distance in controller.reservation_candidates():
                if target_id in local or not math.isfinite(score) or not math.isfinite(distance) or distance < 0:
                    raise ValueError('local candidates must be unique with finite scores and nonnegative distances')
                local[target_id] = (score, distance)
            candidates[agent_id] = local
        chosen = {agent_id: None for agent_id in sorted(by_id)}
        occupants: dict[int, set[int]] = {}

        def claim(agent_id, target_id):
            occupants.setdefault(target_id, set()).add(agent_id)
            chosen[agent_id] = target_id

        # Capacity is fixed by the scenario; sorting additionally makes an
        # explicit runtime capacity reduction release excess holders predictably.
        for agent_id in sorted(by_id):
            incumbent = self._assignments.get(agent_id)
            if incumbent in candidates[agent_id] and len(occupants.get(incumbent, ())) < self.capacity:
                claim(agent_id, incumbent)

        proposals, alternatives = [], []
        for agent_id, controller in by_id.items():
            if chosen[agent_id] is not None:
                continue
            for target_id, (score, distance) in candidates[agent_id].items():
                priority = (-score, agent_id, distance, target_id)
                alternatives.append(priority)
                if target_id == controller.target_id:
                    proposals.append(priority)
        for priority in sorted(proposals)+sorted(alternatives):
            _, agent_id, _, target_id = priority
            if chosen[agent_id] is None and len(occupants.get(target_id, ())) < self.capacity:
                claim(agent_id, target_id)

        counts = Counter(target_id for target_id in chosen.values() if target_id is not None)
        violations = sum(max(0, count-self.capacity) for count in counts.values())
        self.peak_capacity_violations = max(self.peak_capacity_violations, violations)
        if violations:
            raise RuntimeError('reservation capacity invariant violated')
        slots = {}
        for target_id, members in occupants.items():
            used = set()
            for agent_id in sorted(members):
                old_slot = self._slots.get(agent_id)
                if (self._assignments.get(agent_id) == target_id and old_slot is not None
                        and 0 <= old_slot < self.capacity and old_slot not in used):
                    slots[agent_id] = old_slot
                    used.add(old_slot)
            for agent_id in sorted(members):
                if agent_id not in slots:
                    slot = next(value for value in range(self.capacity) if value not in used)
                    slots[agent_id] = slot
                    used.add(slot)
        for agent_id, target_id in chosen.items():
            by_id[agent_id].accept_reservation(target_id, slot=slots.get(agent_id, 0), capacity=self.capacity)
        self._assignments, self._slots = chosen, slots
        return chosen.copy()
