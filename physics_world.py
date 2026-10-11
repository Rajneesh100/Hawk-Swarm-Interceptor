"""MuJoCo force integration for idealized, attitude-stabilized observers.

Commands are desired net acceleration in world axes. The ideal flight controller
adds gravity compensation, limits total thrust, then applies first-order motor
lag. MuJoCo integrates gravity, quadratic air-relative drag, and contacts at the
physics rate. Vector thrust abstracts attitude/rotor mixing; this is not a
validated multirotor aerodynamic model. Prescribed target tracks are mocaps.
"""
import numpy as np
import mujoco


def _limit(v, maximum):
    return v * np.minimum(1., maximum / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


class PhysicsWorld:
    def __init__(self, cfg, capacity):
        self.config = cfg
        self.substeps = int(round(cfg.dt / cfg.physics_dt))
        bodies = []
        for i in range(cfg.n_chasers):
            # Compile the gravity-compensation path so it can be toggled for
            # absent bay bodies at runtime (MuJoCo's ngravcomp is immutable).
            bodies.append(f'''<body name="craft_{i}" pos="0 0 40" gravcomp="1"><freejoint name="joint_{i}"/>
                <geom name="craft_geom_{i}" type="sphere" size="{cfg.body_radius}" mass="{cfg.mass}"
                friction="0.5 0.01 0.001" condim="3"/></body>''')
        for j in range(capacity):
            bodies.append(f'''<body name="track_{j}" mocap="true" pos="0 0 -10000">
                <geom name="track_geom_{j}" type="sphere" size="{cfg.body_radius}"/></body>''')
        xml = f'''<mujoco model="noncontact_observation">
            <option timestep="{cfg.physics_dt}" gravity="0 0 {-cfg.gravity}" integrator="Euler"/>
            <default><geom solref="0.01 1" solimp="0.95 0.99 0.001"/></default>
            <worldbody><geom name="ground" type="plane" size="0 0 0.1" pos="0 0 {cfg.ground_z}"/>
            {''.join(bodies)}</worldbody></mujoco>'''
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._body_ids = np.array([self.model.body(f'craft_{i}').id for i in range(cfg.n_chasers)])
        self.model.body_gravcomp[self._body_ids] = 0.
        self._qpos = np.array([self.model.jnt_qposadr[self.model.joint(f'joint_{i}').id] for i in range(cfg.n_chasers)])
        self._qvel = np.array([self.model.jnt_dofadr[self.model.joint(f'joint_{i}').id] for i in range(cfg.n_chasers)])
        self._mocap = np.array([self.model.body_mocapid[self.model.body(f'track_{j}').id] for j in range(capacity)])
        self._target_geoms = np.array([self.model.geom(f'track_geom_{j}').id for j in range(capacity)])
        self._craft_geoms = np.array([self.model.geom(f'craft_geom_{i}').id for i in range(cfg.n_chasers)])
        self._ground_geom = self.model.geom('ground').id
        self._active = np.ones(cfg.n_chasers, dtype=bool)
        self.thrust_forces = _limit(np.tile([0., 0., cfg.mass*cfg.gravity], (cfg.n_chasers, 1)), cfg.thrust_max)
        self.contact_count = 0  # cumulative substep contact observations, not unique impacts
        self.ground_contact_observations = 0
        self.body_contact_observations = 0
        self.last_contacts = 0
        self.last_acceleration = np.zeros((cfg.n_chasers, 3))

    def reset_craft(self, craft_id, position, *, active=True):
        """Reset one bay's physical incarnation, including motor and angular state."""
        qp, qv, body = self._qpos[craft_id], self._qvel[craft_id], self._body_ids[craft_id]
        self._active[craft_id] = bool(active)
        self.data.qpos[qp:qp+3] = position
        self.data.qpos[qp+3:qp+7] = [1., 0., 0., 0.]
        self.data.qvel[qv:qv+6] = 0.
        self.data.qacc_warmstart[qv:qv+6] = 0.
        self.data.xfrc_applied[body] = 0.
        self.thrust_forces[craft_id] = 0.
        self.last_acceleration[craft_id] = 0.
        self.model.geom_contype[self._craft_geoms[craft_id]] = int(active)
        self.model.geom_conaffinity[self._craft_geoms[craft_id]] = int(active)
        self.model.geom_rgba[self._craft_geoms[craft_id], 3] = float(active)
        # Absent bodies retain an inert storage pose with no gravity or contacts.
        self.model.body_gravcomp[body] = 0. if active else 1.
        mujoco.mj_forward(self.model, self.data)

    def remove_target(self, slot):
        """Remove a completed prescribed track from contact geometry immediately."""
        self.model.geom_contype[self._target_geoms[slot]] = 0
        self.model.geom_conaffinity[self._target_geoms[slot]] = 0
        self.model.geom_rgba[self._target_geoms[slot], 3] = 0.
        self.data.mocap_pos[self._mocap[slot]] = [0., 0., -10000.]
        mujoco.mj_forward(self.model, self.data)

    def step(self, positions, velocities, commands, target_positions, target_present, *, target_start=None, chaser_active=None):
        """Integrate one control period; editable caller arrays are authoritative.

        target_start optionally interpolates each prescribed track over substeps.
        Contacts affect free craft, never the prescribed reference trajectories.
        """
        cfg = self.config
        active = np.ones(cfg.n_chasers, dtype=bool) if chaser_active is None else np.asarray(chaser_active, dtype=bool)
        if active.shape != (cfg.n_chasers,):
            raise ValueError('chaser_active must have one entry per craft')
        self._active[:] = active
        self.model.geom_contype[self._craft_geoms] = active.astype(int)
        self.model.geom_conaffinity[self._craft_geoms] = active.astype(int)
        self.model.geom_rgba[self._craft_geoms, 3] = active.astype(float)
        self.model.body_gravcomp[self._body_ids] = (~active).astype(float)
        for i, (qp, qv) in enumerate(zip(self._qpos, self._qvel)):
            self.data.qpos[qp:qp+3] = positions[i]
            self.data.qvel[qv:qv+3] = velocities[i] if active[i] else 0.
            if not active[i]:
                self.data.qvel[qv:qv+6] = 0.
        present = np.asarray(target_present, dtype=bool)
        self.model.geom_contype[self._target_geoms] = present.astype(int)
        self.model.geom_conaffinity[self._target_geoms] = present.astype(int)
        self.model.geom_rgba[self._target_geoms, 3] = present.astype(float)
        starts = target_positions if target_start is None else target_start
        desired = cfg.mass * (_limit(np.asarray(commands, dtype=float), cfg.a_max) + [0., 0., cfg.gravity])
        desired = _limit(desired, cfg.thrust_max)
        desired[~active] = 0.
        self.thrust_forces[~active] = 0.
        alpha = 1. if cfg.thrust_tau == 0. else -np.expm1(-cfg.physics_dt/cfg.thrust_tau)
        wind = np.asarray(cfg.wind, dtype=float)
        self.last_contacts = 0
        for k in range(self.substeps):
            fraction = (k+1)/self.substeps
            target_step = starts + fraction*(target_positions-starts)
            self.data.mocap_pos[self._mocap] = np.where(present[:, None], target_step, [0., 0., -10000.])
            self.thrust_forces += alpha*(desired-self.thrust_forces)
            self.data.xfrc_applied.fill(0.)
            for i, (body, qv) in enumerate(zip(self._body_ids, self._qvel)):
                if not active[i]:
                    continue
                relative = self.data.qvel[qv:qv+3] - wind
                drag = -.5*cfg.air_density*cfg.drag_area*np.linalg.norm(relative)*relative
                self.data.xfrc_applied[body, :3] = self.thrust_forces[i] + drag
                # Damping angular motion is the idealized stabilization layer.
                self.data.xfrc_applied[body, 3:] = -.5*self.data.qvel[qv+3:qv+6]
            mujoco.mj_step(self.model, self.data)
            self.last_contacts += int(self.data.ncon)
            # Count contact solver observations, not distinct collisions. A
            # landed craft can contribute many ground support observations.
            for contact in self.data.contact[:self.data.ncon]:
                if self._ground_geom in contact.geom:
                    self.ground_contact_observations += 1
                else:
                    self.body_contact_observations += 1
        self.contact_count += self.last_contacts
        new_pos = np.array([self.data.qpos[q: q+3] for q in self._qpos])
        new_vel = np.array([self.data.qvel[q: q+3] for q in self._qvel])
        self.last_acceleration = (new_vel-velocities)/cfg.dt
        mujoco.mj_forward(self.model, self.data)
        return new_pos, new_vel
