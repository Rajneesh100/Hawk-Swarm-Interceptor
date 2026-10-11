"""Safe observation completion, bay relaunch, and per-profile speed regressions."""
import unittest
from unittest.mock import patch
import numpy as np
from sim_engine import Config, Simulation, SERVICED


class TestObserverCycle(unittest.TestCase):
    def fixture(self, **overrides):
        values = dict(n_chasers=2, n_targets=1, recycle_on_service=True,
                      service_duration=.04, relaunch_delay=.06, drag_area=0.,
                      thrust_tau=0., position_noise=0., velocity_noise=0.)
        values.update(overrides)
        sim = Simulation(Config(**values), 7)
        sim.chaser_pos[:] = [[10., 10., 40.], [40., 40., 40.]]
        sim.chaser_vel[:] = 0.
        sim.target_pos[:] = [[20., 10., 40.]]
        sim.target_vel[:] = 0.
        sim.set_observation_assignments({0: int(sim.target_ids[0]), 1: None})
        return sim

    def complete(self, sim):
        with patch.object(sim, '_advance_targets', lambda: None):
            sim.step(np.zeros((sim.config.n_chasers, 3)))
            sim.step(np.zeros((sim.config.n_chasers, 3)))

    def test_safe_completion_removes_only_winner_and_freezes_event_identity(self):
        sim = self.fixture()
        target_id = int(sim.target_ids[0])
        self.complete(sim)
        self.assertEqual(sim.status[0], SERVICED)
        self.assertFalse(sim.target_present[0])
        np.testing.assert_array_equal(sim.chaser_active, [False, True])
        np.testing.assert_array_equal(sim.chaser_vel[0], [0., 0., 0.])
        np.testing.assert_array_equal(sim.world.thrust_forces[0], [0., 0., 0.])
        self.assertEqual(sim.world.model.geom('craft_geom_0').contype[0], 0)
        self.assertEqual(sim.world.model.geom('track_geom_0').contype[0], 0)
        event = sim.lifecycle_events[-1]
        self.assertEqual((event['type'], event['chaser_id'], event['generation'], event['target_id']),
                         ('completion', 0, 0, target_id))
        self.assertAlmostEqual(event['time'], .04)
        self.assertEqual(event['position'], (10., 10., 40.))
        self.assertEqual(event['target_position'], (20., 10., 40.))
        sim.target_ids[0] = 999
        self.assertEqual(event['target_id'], target_id)
        self.assertEqual(sim.stats()['completed_observers'], 1)
        self.assertEqual(sim.stats()['active_chasers'], 1)
        self.assertEqual(sim.stats()['retired_chasers'], 1)
        self.assertEqual(sim.stats()['serviced'], 1)
        self.assertEqual(sim.world.contact_count, 0)

    def test_simultaneous_service_withdraws_only_one_winner(self):
        sim = self.fixture()
        sim.chaser_pos[1] = [30., 10., 40.]
        sim.set_observation_assignments(None)
        self.complete(sim)
        np.testing.assert_array_equal(sim.chaser_active, [False, True])
        self.assertEqual(sim.stats()['serviced'], 1)
        self.assertEqual(sim.stats()['completed_observers'], 1)
        self.assertEqual(len(sim.lifecycle_events), 1)

    def test_inactive_body_ignores_commands_and_is_not_a_sensed_peer(self):
        sim = self.fixture(relaunch_delay=2.)
        self.complete(sim)
        position = sim.chaser_pos[0].copy()
        for _ in range(4):
            sim.step(np.full((2, 3), 10.))
        np.testing.assert_array_equal(sim.chaser_pos[0], position)
        np.testing.assert_array_equal(sim.chaser_vel[0], 0.)
        np.testing.assert_array_equal(sim.world.thrust_forces[0], 0.)
        self.assertNotIn(0, sim.observe(1).chaser_ids)
        inactive_observation = sim.observe(0)
        self.assertFalse(inactive_observation.active)
        self.assertIn(0, inactive_observation.chaser_ids)
        self.assertIn(1, inactive_observation.chaser_ids)
        self.assertEqual(len(inactive_observation.target_ids), 0)
        violations = sim.near_contact_violations
        sim.chaser_pos[1] = position
        sim.chaser_vel[1] = 0.
        sim.step(np.zeros((2, 3)))
        self.assertEqual(sim.near_contact_violations, violations)
        self.assertEqual(sim.world.last_contacts, 0)

    def test_relaunch_has_new_generation_at_base_without_a_teleport_sweep(self):
        sim = self.fixture()
        self.complete(sim)
        while sim.time < .10-1e-10:
            sim.step(np.zeros((2, 3)))
            self.assertFalse(sim.chaser_active[0])
        before = sim.near_contact_violations
        sim.step(np.zeros((2, 3)))
        self.assertTrue(sim.chaser_active[0])
        self.assertEqual(sim.chaser_generation[0], 1)
        np.testing.assert_allclose(sim.chaser_pos[0], sim.base_positions[0], atol=1e-10)
        np.testing.assert_allclose(sim.previous_chaser_pos[0], sim.base_positions[0])
        self.assertEqual(sim.near_contact_violations, before)
        event = sim.lifecycle_events[-1]
        self.assertEqual((event['type'], event['chaser_id'], event['generation'], event['target_id']),
                         ('launch', 0, 1, None))
        self.assertAlmostEqual(event['time'], .10)
        self.assertEqual(sim.stats()['relaunches'], 1)

    def test_occupied_launch_bay_delays_relaunch(self):
        sim = self.fixture()
        self.complete(sim)
        sim.chaser_pos[1] = sim.base_positions[0]
        for _ in range(8):
            sim.step(np.zeros((2, 3)))
        self.assertFalse(sim.chaser_active[0])
        self.assertEqual(sim.chaser_generation[0], 0)
        sim.chaser_pos[1] = [80., 50., 40.]
        sim.step(np.zeros((2, 3)))
        self.assertTrue(sim.chaser_active[0])
        self.assertEqual(sim.chaser_generation[0], 1)

    def test_cycle_is_opt_in_and_requires_one_observer_per_target(self):
        sim = self.fixture(recycle_on_service=False)
        self.complete(sim)
        self.assertTrue(sim.target_present[0])
        self.assertTrue(sim.chaser_active.all())
        self.assertEqual(len(sim.lifecycle_events), 0)
        with self.assertRaises(ValueError):
            Config(recycle_on_service=True, chasers_per_target=2)
        with self.assertRaises(ValueError):
            Config(relaunch_delay=0.)

    def test_profile_speed_ranges_apply_to_every_track(self):
        for profile in ('random', 'constant-height', 'weave', 'parabolic', 'orbit'):
            cfg = Config(n_chasers=1, n_targets=2, trajectory=profile,
                         profile_speeds=((profile, 8., 9.),))
            self.assertEqual(cfg.speed_range(profile), (8., 9.))
            sim = Simulation(cfg, 3)
            for tick in range(1000):
                sim.tick = tick
                sim._advance_targets()
                speed = np.linalg.norm(sim.target_vel, axis=1)
                self.assertTrue(np.all(speed >= 8.-1e-9), profile)
                self.assertTrue(np.all(speed <= 9.+1e-9), profile)
            self.assertEqual(cfg.speed_range('unconfigured'), (3., 7.))

    def test_profile_speed_validation_and_bounded_events(self):
        for speeds in ((('mixed', 3., 7.),), (('orbit', 0., 7.),),
                       (('weave', 7., 3.),), (('random', 3., float('inf')),),
                       (('orbit', 3., 7.), ('orbit', 4., 8.))):
            with self.assertRaises(ValueError):
                Config(profile_speeds=speeds)
        sim = self.fixture()
        self.assertEqual(sim.lifecycle_events.maxlen, 128)

    def test_recycled_track_path_is_keyed_by_global_id_not_pool_slot(self):
        cfg = Config(n_chasers=1, n_targets=1, max_tracks=2,
                     continuous=True, trajectory='random')
        a, b = Simulation(cfg, 17), Simulation(cfg, 17)
        a.target_present[:] = False
        b.target_present[:] = False
        a._spawn(0)
        b._spawn(1)
        self.assertEqual(a.target_ids[0], b.target_ids[1])
        for tick in range(60):
            a.tick = b.tick = tick
            a._advance_targets()
            b._advance_targets()
            np.testing.assert_array_equal(a.target_pos[0], b.target_pos[1])
            np.testing.assert_array_equal(a.target_vel[0], b.target_vel[1])


if __name__ == '__main__':
    unittest.main()
