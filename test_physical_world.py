"""Physical force integration and bounded continuous-arrival regressions."""
import unittest
import numpy as np
from sim_engine import Config, Simulation


class TestPhysicalWorld(unittest.TestCase):
    def test_mujoco_substeps_and_gravity_compensated_hover(self):
        sim = Simulation(Config(n_chasers=1, n_targets=1, drag_area=0., thrust_tau=0.), 1)
        sim.chaser_pos[:] = [[10., 10., 40.]]
        sim.chaser_vel[:] = 0.
        for _ in range(50):
            sim.step(np.zeros((1, 3)))
        import mujoco
        self.assertIsInstance(sim.world.model, mujoco.MjModel)
        self.assertIsInstance(sim.world.data, mujoco.MjData)
        self.assertAlmostEqual(sim.world.model.opt.timestep, .002)
        self.assertAlmostEqual(sim.world.data.time, 1., places=10)
        np.testing.assert_allclose(sim.chaser_pos, [[10., 10., 40.]], atol=1e-8)
        np.testing.assert_allclose(sim.chaser_vel, 0., atol=1e-8)

    def test_force_over_mass_acceleration_and_drag(self):
        cfg = Config(n_chasers=1, n_targets=1, drag_area=0., thrust_tau=0.)
        sim = Simulation(cfg, 1)
        sim.chaser_pos[:] = [[10., 10., 40.]]
        sim.chaser_vel[:] = 0.
        sim.step(np.array([[2., 0., 0.]]))
        self.assertAlmostEqual(sim.chaser_vel[0, 0], 2.*cfg.dt, places=8)
        self.assertAlmostEqual(sim.chaser_vel[0, 2], 0., places=8)
        drag = Simulation(Config(n_chasers=1, n_targets=1, thrust_tau=0.), 1)
        drag.chaser_pos[:] = [[10., 10., 40.]]
        drag.chaser_vel[:] = [[8., 0., 0.]]
        drag.step(np.zeros((1, 3)))
        self.assertLess(drag.chaser_vel[0, 0], 8.)

    def test_thrust_saturation_lag_and_wind_are_physical(self):
        cfg = Config(n_chasers=1, n_targets=1, thrust_max=20., thrust_tau=.2, wind=(8., 0., 0.))
        sim = Simulation(cfg, 1)
        sim.chaser_pos[:] = [[10., 10., 40.]]
        sim.chaser_vel[:] = 0.
        sim.step(np.array([[18., 0., 0.]]))
        self.assertLessEqual(np.linalg.norm(sim.world.thrust_forces[0]), 20.+1e-10)
        self.assertLess(sim.chaser_vel[0, 0], 18.*cfg.dt)
        passive = Simulation(Config(n_chasers=1, n_targets=1, wind=(8., 0., 0.)), 1)
        passive.chaser_pos[:] = [[10., 10., 40.]]
        passive.chaser_vel[:] = 0.
        passive.step(np.zeros((1, 3)))
        self.assertGreater(passive.chaser_vel[0, 0], 0.)

    def test_ground_contact_is_resolved_by_mujoco(self):
        sim = Simulation(Config(n_chasers=1, n_targets=1, thrust_max=.1, thrust_tau=0., drag_area=0.), 1)
        sim.chaser_pos[:] = [[10., 10., 1.]]
        sim.chaser_vel[:] = 0.
        for _ in range(100):
            sim.step(np.zeros((1, 3)))
        self.assertGreater(sim.world.contact_count, 0)
        self.assertGreater(sim.world.ground_contact_observations, 0)
        self.assertEqual(sim.world.body_contact_observations, 0)
        self.assertEqual(sim.world.contact_count,
                         sim.world.ground_contact_observations+sim.world.body_contact_observations)
        self.assertGreater(sim.chaser_pos[0, 2], sim.config.ground_z+.25)

    def test_body_contacts_are_classified_separately_from_ground_support(self):
        from unittest.mock import patch
        for obstacle in ('target', 'craft'):
            with self.subTest(obstacle=obstacle):
                count = 2 if obstacle == 'craft' else 1
                sim = Simulation(Config(n_chasers=count, n_targets=1,
                                        drag_area=0., thrust_tau=0.), 1)
                sim.chaser_pos[:] = [[10., 10., 40.]]
                sim.chaser_vel[:] = 0.
                sim.target_pos[:] = [[80., 40., 40.]]
                sim.target_vel[:] = 0.
                if obstacle == 'target':
                    sim.target_pos[0] = [10.5, 10., 40.]
                else:
                    sim.chaser_pos[1] = [10.5, 10., 40.]
                with patch.object(sim, '_advance_targets', lambda: None):
                    sim.step(np.zeros((count, 3)))
                self.assertGreater(sim.world.body_contact_observations, 0)
                self.assertEqual(sim.world.ground_contact_observations, 0)
                stats = sim.stats()
                self.assertEqual(stats['body_contact_observations'], sim.world.contact_count)
                self.assertEqual(stats['ground_contact_observations'], 0)

    def test_continuous_arrivals_use_global_ids_and_bounded_pool(self):
        cfg = Config(n_chasers=1, n_targets=1, continuous=True, arrival_interval=.04, max_tracks=2, duration=None)
        sim = Simulation(cfg, 3)
        first_id = int(sim.target_ids[0])
        for _ in range(4):
            sim.step(np.zeros((1, 3)))
        self.assertEqual(len(sim.target_ids), 2)
        self.assertEqual(np.count_nonzero(sim.target_present), 2)
        self.assertFalse(sim.done)
        self.assertEqual(len(np.unique(sim.target_ids[sim.target_present])), 2)
        sim.target_pos[0, 0] = cfg.width+cfg.offstage_margin+1.
        sim.step(np.zeros((1, 3)))
        for _ in range(3):
            sim.step(np.zeros((1, 3)))
        self.assertIsNone(sim.target_index(first_id))
        self.assertGreater(sim.target_ids[0], first_id)
        self.assertEqual(len(sim.service_dwell[0]), 2)
        self.assertGreater(sim.stats()['arrivals'], 2)

    def test_spawn_does_not_create_a_swept_collision_from_recycled_slot(self):
        cfg = Config(n_chasers=1, n_targets=1, continuous=True, max_tracks=1,
                     arrival_interval=.02, duration=None, drag_area=0., thrust_tau=0.)
        sim = Simulation(cfg, 1)
        sim.chaser_pos[:] = [[80., 30., 40.]]
        sim.chaser_vel[:] = 0.
        sim.target_pos[:] = [[cfg.width+cfg.offstage_margin+1., 30., 40.]]
        sim.step(np.zeros((1, 3)))
        baseline = sim.near_contact_violations
        sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.near_contact_violations, baseline)
        self.assertFalse(sim.service_dwell.any())

    def test_constant_height_profile_and_replay_are_deterministic(self):
        cfg = Config(n_chasers=1, n_targets=2, trajectory='constant-height', continuous=True,
                     max_tracks=4, arrival_interval=.5, duration=2.)
        a, b = Simulation(cfg, 42), Simulation(cfg, 42)
        initial_z = a.target_pos[:2, 2].copy()
        for _ in range(50):
            a.observe(0)
            a.step(np.zeros((1, 3)))
            b.step(np.ones((1, 3)))
        np.testing.assert_array_equal(a.target_ids, b.target_ids)
        np.testing.assert_array_equal(a.target_pos, b.target_pos)
        np.testing.assert_array_equal(a.target_vel, b.target_vel)
        np.testing.assert_array_equal(a.target_pos[:2, 2], initial_z)
        np.testing.assert_array_equal(a.target_vel[:2, 2], np.zeros(2))

    def test_profiles_keep_speed_altitude_and_acceleration_envelopes(self):
        from physics_constants import TRAJECTORIES
        for profile in TRAJECTORIES:
            cfg = Config(n_chasers=1, n_targets=3, trajectory=profile, duration=30.)
            sim = Simulation(cfg, 41)
            for tick in range(1300):  # includes the orbit-to-outbound transition
                sim.tick = tick
                previous = sim.target_vel.copy()
                sim._advance_targets()
                speed = np.linalg.norm(sim.target_vel, axis=1)
                acceleration = np.linalg.norm(sim.target_vel-previous, axis=1)/cfg.dt
                self.assertTrue(np.all(speed >= cfg.target_speed_min-1e-9), profile)
                self.assertTrue(np.all(speed <= cfg.target_speed_max+1e-9), profile)
                self.assertTrue(np.all(acceleration <= cfg.target_accel_max+1e-8), profile)
                self.assertTrue(np.all(sim.target_pos[:, 2] >= cfg.altitude_min-1e-9), profile)
                self.assertTrue(np.all(sim.target_pos[:, 2] <= cfg.altitude_max+1e-9), profile)
            self.assertLessEqual(len(sim._waypoint_cache), 2)

    def test_continuous_duration_and_pool_configuration_validation(self):
        with self.assertRaises(ValueError):
            Config(duration=None)
        with self.assertRaises(ValueError):
            Config(continuous=True, n_targets=3, max_tracks=2)
        with self.assertRaises(ValueError):
            Config(physics_dt=.003)
        with self.assertRaises(ValueError):
            Config(trajectory='uncontrolled')
        sim = Simulation(Config(n_chasers=1, n_targets=1, continuous=True, duration=.04), 1)
        sim.status[:] = 1
        self.assertFalse(sim.done)
        sim.step(np.zeros((1, 3)))
        sim.step(np.zeros((1, 3)))
        self.assertTrue(sim.done)

    def test_larger_fleet_starts_with_separated_observers(self):
        for count in (12, 24):
            sim = Simulation(Config(n_chasers=count, n_targets=count), 7)
            peer = np.linalg.norm(sim.chaser_pos[:, None]-sim.chaser_pos[None], axis=-1)
            peer[np.eye(count, dtype=bool)] = np.inf
            target = np.linalg.norm(sim.chaser_pos[:, None]-sim.target_pos[None], axis=-1)
            self.assertGreater(peer.min(), sim.config.min_separation+1.)
            self.assertGreater(target.min(), sim.config.min_separation+1.)

    def test_assignment_metadata_restricts_service_without_controlling_motion(self):
        from unittest.mock import patch
        cfg = Config(n_chasers=1, n_targets=2, service_duration=.04, drag_area=0., thrust_tau=0.)
        sim = Simulation(cfg, 1)
        sim.chaser_pos[:] = [[10., 10., 40.]]
        sim.chaser_vel[:] = 0.
        sim.target_pos[:] = [[20., 10., 40.], [10., 21., 40.]]
        sim.target_vel[:] = 0.
        sim.set_observation_assignments({0: int(sim.target_ids[1])})
        with patch.object(sim, '_advance_targets', lambda: None):
            for _ in range(2):
                sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.status.tolist(), [0, 1])
        np.testing.assert_allclose(sim.chaser_pos, [[10., 10., 40.]])


if __name__ == '__main__':
    unittest.main()
