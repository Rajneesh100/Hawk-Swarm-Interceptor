"""Non-contact observation, neural allocation, peer protocol and camera regressions."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

import numpy as np
import torch

from sim_engine import Config, Simulation
from estimator import Observation, estimate_pairs
import swarm_controller as guidance
from main import project_isometric, run_headless
import main as harness


torch.set_num_threads(1)


def packet(agent_id, chaser_ids, target_ids, *, time=0., active=True):
    """A deterministic sensed scene, with all tracks above the ground plane."""
    return Observation(
        time, agent_id, np.array(chaser_ids, dtype=int),
        np.array([[float(i), 0., 10.] for i in chaser_ids]).reshape(-1, 3),
        np.tile([0., 0., 2.], (len(chaser_ids), 1)),
        np.array(target_ids, dtype=int),
        np.array([[float(j), 0., 30.] for j in target_ids]).reshape(-1, 3),
        np.tile([0., 0., -1.], (len(target_ids), 1)), active=active)


def exchange_until_stable(controllers):
    """Deliver actual broadcast packets over a fully connected test transport."""
    rounds = 1+sum(c.config.n_targets for c in controllers)
    for round_index in range(1, rounds+1):
        messages = [c.make_broadcast(round_index) for c in controllers]
        for controller in controllers:
            for message in messages:
                if message.sender_id != controller.agent_id:
                    controller.receive_broadcast(message)
        changed = [c.resolve_round() for c in controllers]
        if not any(changed):
            for controller in controllers:
                controller.finalize_allocation()
            return
    raise AssertionError('negotiation failed to converge within finite target rejections')


def negotiate(controllers, epoch=1):
    for controller in controllers:
        controller.begin_allocation(epoch)
    exchange_until_stable(controllers)


class TestTestbed(unittest.TestCase):
    def assert_braking(self, controller):
        """Unassigned motion should decelerate, not pursue an observation track."""
        row = int(np.flatnonzero(controller.obs.chaser_ids == controller.agent_id)[0])
        velocity = controller.obs.chaser_vel[row]
        command = controller.get_control_commands()
        acceleration = command  # Commands are desired net acceleration in m/s².
        self.assertLess(float(np.dot(acceleration, velocity)), 0.)


    def test_lateral_y_is_not_altitude(self):
        result = estimate_pairs(np.array([[0., -20., 10.]]), np.array([[0., 1., 0.]]),
                                np.array([[0., -10., 10.]]), np.array([[0., -1., 0.]]),
                                r_dock=1., sigma_base=1., zem_scale=.1, ground_z=0.)
        self.assertGreater(result.probability[0, 0], 0.)
        sim = Simulation(Config(n_chasers=1, n_targets=1, drift_std=0.), 1)
        sim.chaser_pos[:] = [[50., 20., 10.]]
        sim.target_pos[:] = [[0., -.1, 10.]]
        sim.target_vel[:] = [[0., -20., 0.]]
        sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.status.tolist(), [0])

    def test_vector_los_and_proportional_navigation(self):
        result = estimate_pairs(np.array([[0., 0., 10.]]), np.zeros((1, 3)),
                                np.array([[10., 0., 10.]]), np.array([[-2., 1., 0.]]),
                                r_dock=1., sigma_base=1., zem_scale=.1, ground_z=0.)
        np.testing.assert_allclose(result.omega_los[0, 0], [0., 0., .1])
        np.testing.assert_allclose(guidance.proportional_navigation(
            np.array([10., 0., 0.]), np.array([-2., 1., 0.]), 3., 18.),
            [0., .6, 0.], atol=1e-12)
        np.testing.assert_allclose(guidance.proportional_navigation(
            np.array([10., 0., 0.]), np.array([-2., 0., 1.]), 3., 18.),
            [0., 0., .6], atol=1e-12)

    def test_pn_rotates_with_coordinates_and_saturates_by_norm(self):
        rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
        r, v = np.array([10., 2., 3.]), np.array([-4., 1., 2.])
        a = guidance.proportional_navigation(r, v, 3., 18.)
        rotated = guidance.proportional_navigation(rotation @ r, rotation @ v, 3., 18.)
        np.testing.assert_allclose(rotated, rotation @ a, atol=1e-12)
        saturated = guidance.proportional_navigation(
            np.array([1., 0., 0.]), np.array([-100., 20., 20.]), 3., 18.)
        self.assertAlmostEqual(float(np.linalg.norm(saturated)), 18.)
        self.assertAlmostEqual(float(saturated[1]), float(saturated[2]))
        for separation, relative_velocity in ((np.zeros(3), v), (r, r)):
            np.testing.assert_array_equal(guidance.proportional_navigation(
                separation, relative_velocity, 3., 18.), np.zeros(3))

    def test_peer_capability_messages_change_the_target_winner(self):
        cfg = Config(n_chasers=2, n_targets=2)
        controllers = [guidance.SwarmController(i, cfg) for i in range(2)]
        for i, controller in enumerate(controllers):
            obs = packet(i, [i], [0, 1])
            obs.chaser_pos[:] = [[0., 0., 10.+10.*i]]
            obs.target_pos[:] = [[0., 0., 30.], [40., 0., 30.]]
            controller.update_telemetry(obs)
            self.assertIsNone(controller.target_id)
            self.assert_braking(controller)
            controller.begin_allocation(1)
            self.assertIsNone(controller.target_id)
            self.assert_braking(controller)
        first, second = [c.make_broadcast(0) for c in controllers]
        self.assertEqual((first.target_id, second.target_id), (0, 0))
        self.assertGreater(second.capability, first.capability)
        expected_probability = -np.expm1(-cfg.r_dock**2/(2*cfg.sigma_base**2))
        self.assertAlmostEqual(first.capability, expected_probability/(1+20./cfg.cruise_speed))
        self.assertAlmostEqual(second.capability, expected_probability/(1+10./cfg.cruise_speed))
        controllers[0].receive_broadcast(second)
        controllers[1].receive_broadcast(first)
        self.assertTrue(controllers[0].resolve_round())
        self.assertFalse(controllers[1].resolve_round())
        for controller in controllers:
            self.assertIsNone(controller.target_id)
            self.assert_braking(controller)
        exchange_until_stable(controllers)
        self.assertEqual([c.target_id for c in controllers], [1, 0])
        self.assertEqual(controllers[0].peer_buffer[1].target_id, 0)

    def test_equal_capability_tie_is_stable_by_sender_id(self):
        cfg = Config(n_chasers=2, n_targets=2)
        controllers = [guidance.SwarmController(i, cfg) for i in range(2)]
        for i, controller in enumerate(controllers):
            obs = packet(i, [i], [0, 1])
            obs.chaser_pos[:] = [[0., 0., 10.]]
            controller.update_telemetry(obs)
        negotiate(controllers)
        self.assertEqual([c.target_id for c in controllers], [0, 1])
        # A periodic epoch with unchanged observations keeps the same commitment.
        negotiate(controllers, epoch=2)
        self.assertEqual([c.target_id for c in controllers], [0, 1])

    def test_prior_target_stays_preferred_when_another_becomes_closer(self):
        cfg = Config(n_chasers=1, n_targets=2)
        controller = guidance.SwarmController(0, cfg)
        obs = packet(0, [0], [0, 1])
        obs.target_pos[:] = [[30., 0., 30.], [0., 0., 30.]]
        controller.update_telemetry(obs)
        negotiate([controller])
        self.assertEqual(controller.target_id, 1)
        changed = replace(obs, time=1., target_pos=np.array([[0., 0., 15.], [0., 0., 30.]]))
        controller.update_telemetry(changed)
        negotiate([controller], epoch=2)
        self.assertEqual(controller.target_id, 1)

    def test_connected_negotiation_handles_equal_and_unequal_populations(self):
        for n_chasers, n_targets in ((4, 4), (3, 5), (5, 3)):
            with self.subTest(chasers=n_chasers, targets=n_targets):
                cfg = Config(n_chasers=n_chasers, n_targets=n_targets)
                controllers = [guidance.SwarmController(i, cfg) for i in range(n_chasers)]
                for i, controller in enumerate(controllers):
                    controller.update_telemetry(packet(i, [i], list(range(n_targets))))
                negotiate(controllers)
                chosen = [c.target_id for c in controllers if c.target_id is not None]
                self.assertEqual(len(chosen), min(n_chasers, n_targets))
                self.assertEqual(len(set(chosen)), len(chosen))
                for controller in controllers:
                    if controller.target_id is None:
                        self.assert_braking(controller)

    def test_peer_negotiation_resolves_conflicting_noisy_local_views(self):
        cfg = Config(n_chasers=4, n_targets=6)
        controllers = [guidance.SwarmController(i, cfg) for i in range(4)]
        # Sensor views omit peers, reorder tracks, and disagree on relative motion.
        views = [([0], [0, 1, 4]), ([3, 1], [0, 2, 5]),
                 ([2], [0, 2, 5]), ([3, 0], [0, 1, 3])]
        for i, (controller, (chasers, targets)) in enumerate(zip(controllers, views)):
            obs = packet(i, chasers, targets)
            obs.target_pos[:, 0] += (i-2)*3.7
            obs.target_vel[:, 1] += i*.2
            controller.update_telemetry(obs)
        negotiate(controllers)
        commitments = [c.target_id for c in controllers if c.target_id is not None]
        self.assertEqual(len(set(commitments)), len(commitments))
        self.assertEqual(len(commitments), 4)
        for controller, (_, targets) in zip(controllers, views):
            self.assertIn(controller.target_id, targets)
            self.assertEqual(len(controller.peer_buffer), 3)

    def test_peer_buffer_rejects_stale_expired_and_out_of_order_packets(self):
        cfg = Config(n_chasers=2, n_targets=2, broadcast_ttl=1.)
        controller = guidance.SwarmController(0, cfg)
        controller.update_telemetry(packet(0, [0], [0, 1], time=10.))
        controller.begin_allocation(3)
        valid = guidance.PeerBroadcast(1, 3, 2, 10., 0, .8)
        controller.receive_broadcast(valid)
        self.assertEqual(controller.peer_buffer[1], valid)
        for ignored in (replace(valid, epoch=2, round_index=9),
                        replace(valid, epoch=4),
                        replace(valid, round_index=1, target_id=1),
                        replace(valid, round_index=3, time=8.),
                        replace(valid, round_index=3, time=11.),
                        replace(valid, round_index=3, capability=float('nan'))):
            controller.receive_broadcast(ignored)
            self.assertEqual(controller.peer_buffer[1], valid)
        controller.update_telemetry(packet(0, [0], [0, 1], time=11.1))
        controller.resolve_round()
        self.assertNotIn(1, controller.peer_buffer)

    def test_retirement_withdrawal_purges_claim_and_blocks_reactivation(self):
        cfg = Config(n_chasers=2, n_targets=2)
        controller = guidance.SwarmController(0, cfg)
        controller.update_telemetry(packet(0, [0], [0, 1]))
        controller.begin_allocation(1)
        claim = guidance.PeerBroadcast(1, 1, 0, 0., 0, .9)
        controller.receive_broadcast(claim)
        self.assertIn(1, controller.peer_buffer)
        controller.receive_broadcast(replace(claim, round_index=1,
            target_id=None, capability=0., active=False))
        self.assertNotIn(1, controller.peer_buffer)
        controller.receive_broadcast(replace(claim, round_index=2))
        self.assertNotIn(1, controller.peer_buffer)
        controller.begin_allocation(2)
        controller.receive_broadcast(replace(claim, epoch=2, round_index=3))
        self.assertNotIn(1, controller.peer_buffer)
        controller.update_telemetry(packet(0, [], [0, 1], active=False))
        controller.begin_allocation(3)
        withdrawn = controller.make_broadcast(0)
        self.assertFalse(withdrawn.active)
        self.assertIsNone(withdrawn.target_id)
        np.testing.assert_array_equal(controller.get_control_commands(), np.zeros(3))

    def test_mesh_relays_packets_across_multiple_hops(self):
        cfg = Config(n_chasers=3, n_targets=3, broadcast_range=1.1)
        positions = np.array([[0., 0., 10.], [1., 0., 10.], [2., 0., 10.]])
        controllers = [guidance.SwarmController(i, cfg) for i in range(3)]
        for i, controller in enumerate(controllers):
            obs = packet(i, [i], [0, 1, 2])
            obs.chaser_pos[:] = positions[i]
            controller.update_telemetry(obs)
            controller.begin_allocation(1)
        mesh = harness.SimulatedMesh(cfg)
        message = controllers[0].make_broadcast(0)
        delivered = mesh.exchange([message], controllers, positions, np.ones(3, dtype=bool), 0.)
        self.assertEqual(delivered, 2)
        self.assertEqual(controllers[2].peer_buffer[0], message)
        self.assertNotIn(0, controllers[0].peer_buffer)
        # Retirement removes the middle radio node; the endpoints cannot talk.
        later = controllers[0].make_broadcast(1)
        delivered = mesh.exchange([later], controllers, positions, np.array([True, False, True]), 0.)
        self.assertEqual(delivered, 0)
        self.assertEqual(controllers[2].peer_buffer[0].round_index, 0)
        # Transport rejects an expired packet before delivering it to any peer.
        delivered = mesh.exchange([later], controllers, positions, np.ones(3, dtype=bool), 2.)
        self.assertEqual(delivered, 0)

    def test_mesh_delivers_final_retirement_withdrawal_once(self):
        cfg = Config(n_chasers=2, n_targets=2, broadcast_range=2.)
        positions = np.array([[0., 0., 10.], [1., 0., 10.]])
        controllers = [guidance.SwarmController(i, cfg) for i in range(2)]
        for i, controller in enumerate(controllers):
            controller.update_telemetry(packet(i, [i], [0, 1]))
            controller.begin_allocation(1)
        mesh = harness.SimulatedMesh(cfg)
        claim = controllers[1].make_broadcast(0)
        self.assertEqual(mesh.exchange([claim], controllers, positions, np.ones(2, dtype=bool), 0.), 1)
        self.assertIn(1, controllers[0].peer_buffer)
        controllers[1].update_telemetry(packet(1, [], [0, 1], active=False))
        withdrawn = controllers[1].make_broadcast(1)
        active = np.array([True, False])
        self.assertEqual(mesh.exchange([withdrawn], controllers, positions, active, 0., retiring=(1,)), 1)
        self.assertNotIn(1, controllers[0].peer_buffer)
        replay = replace(claim, round_index=2)
        self.assertEqual(mesh.exchange([replay], controllers, positions, np.ones(2, dtype=bool), 0.), 0)
        self.assertNotIn(1, controllers[0].peer_buffer)


    def test_disconnected_mesh_cannot_guarantee_unique_global_claims(self):
        cfg = Config(n_chasers=2, n_targets=1, broadcast_range=10.)
        controllers = [guidance.SwarmController(i, cfg) for i in range(2)]
        positions = np.array([[0., 0., 10.], [100., 0., 10.]])
        for i, controller in enumerate(controllers):
            obs = packet(i, [i], [0])
            obs.chaser_pos[:] = positions[i]
            controller.update_telemetry(obs)
            controller.begin_allocation(1)
        mesh = harness.SimulatedMesh(cfg)
        messages = [controller.make_broadcast(0) for controller in controllers]
        self.assertEqual(mesh.exchange(messages, controllers, positions, np.ones(2, dtype=bool), 0.), 0)
        for controller in controllers:
            self.assertFalse(controller.resolve_round())
            controller.finalize_allocation()
            self.assertEqual(controller.peer_buffer, {})
        self.assertEqual([controller.target_id for controller in controllers], [0, 0])

    def test_reconnecting_components_renegotiate_before_the_periodic_deadline(self):
        cfg = Config(n_chasers=2, n_targets=1, broadcast_range=.1,
                     drift_std=0., position_noise=0., velocity_noise=0.)
        scenario = harness.Scenario(cfg, seed=1, mode='coordinated')
        scenario.sim.chaser_pos[:] = [[0., 0., 10.], [10., 0., 10.]]
        scenario.sim.chaser_vel[:] = 0.
        scenario.sim.target_pos[:] = [[5., 0., 30.]]
        scenario.sim.target_vel[:] = 0.
        scenario.step()
        self.assertEqual(sum(c.target_id == 0 for c in scenario.controllers), 1)
        self.assertEqual(scenario.duplicate_commitments, 0)
        previous_epoch = scenario.allocation_epoch
        # Sensor visibility stays unchanged, but the two radio components join.
        scenario.sim.chaser_pos[:] = [[0., 0., 10.], [.01, 0., 10.]]
        scenario.step()
        self.assertLess(scenario.sim.time, cfg.allocation_period)
        self.assertGreater(scenario.allocation_epoch, previous_epoch)
        self.assertEqual(scenario.duplicate_commitments, 0)
        chosen = [c.target_id for c in scenario.controllers if c.target_id is not None]
        self.assertEqual(chosen, [0])

    def test_plant_limits_thrust_without_clipping_physical_velocity(self):
        cfg = Config(n_chasers=2, n_targets=2)
        sim = Simulation(cfg, seed=4)
        sim.target_pos[:] = [[1000., 1000., 1000.], [1100., 1100., 1100.]]
        for _ in range(100):
            sim.step(np.full((2, 3), 1e6))
            self.assertTrue(sim.chaser_active.all())
            self.assertTrue(np.isfinite(sim.chaser_pos).all())
            self.assertTrue(np.isfinite(sim.chaser_vel).all())
            self.assertLessEqual(np.linalg.norm(sim.world.thrust_forces, axis=1).max(), cfg.thrust_max+1e-9)
        # v_max limits the guidance reference. Wind, contacts and forced inputs
        # must not be hidden by clipping MuJoCo's integrated physical state.
        self.assertGreater(np.linalg.norm(sim.chaser_vel, axis=1).max(), cfg.v_max)

    def test_force_response_and_actuator_lag(self):
        cfg = Config(n_chasers=1, n_targets=1, drag_area=0.)
        sim = Simulation(cfg, 2)
        sim.chaser_vel[:] = 0.
        sim.target_pos[:] = [[1000., 1000., 1000.]]
        # Hover-compensated thrust builds gradually after a 1.5 m/s² request.
        sim.step(np.array([[1.5, 0., 0.]]))
        self.assertGreater(sim.chaser_vel[0, 0], 0.)
        self.assertLess(sim.chaser_vel[0, 0], 1.5*cfg.dt)
        self.assertAlmostEqual(sim.world.model.opt.timestep, cfg.physics_dt)
        np.testing.assert_allclose(sim.world.model.opt.gravity, [0., 0., -cfg.gravity])

    def test_nonclosing_guidance_can_recover_in_z(self):
        cfg = Config(n_chasers=1, n_targets=1, drift_std=0.)
        sim = Simulation(cfg, 1)
        sim.chaser_pos[:] = [[20., 0., 10.]]
        sim.chaser_vel[:] = [[0., 0., -3.]]
        sim.target_pos[:] = [[20., 0., 30.]]
        sim.target_vel[:] = [[0., 0., 1.]]
        controller = guidance.SwarmController(0, cfg)
        controller.update_telemetry(sim.observe(0))
        negotiate([controller])
        command = controller.get_control_commands()
        velocity = controller.obs.chaser_vel[0]
        # The plant handles force and gravity compensation. Guidance outputs
        # desired net acceleration in SI units, opposing descent.
        self.assertGreater(command[2], 0.)

    def test_replay_independent_of_control_and_observation_calls(self):
        cfg = Config(n_chasers=2, n_targets=2)
        a, b = Simulation(cfg, 42), Simulation(cfg, 42)
        for _ in range(20):
            a.observe(0)
            a.observe(0)
            a.step(np.zeros((2, 3)))
            b.step(np.ones((2, 3)))
        np.testing.assert_array_equal(a.target_pos, b.target_pos)
        np.testing.assert_array_equal(a.target_vel, b.target_vel)
        first, second = a.observe(0), a.observe(0)
        np.testing.assert_array_equal(first.target_pos, second.target_pos)
        first.target_pos[:] = -999
        self.assertFalse(np.any(a.target_pos == -999))



    def test_local_sensor_range_and_no_targets(self):
        cfg = Config(n_chasers=2, n_targets=2, sensor_range=.01)
        sim = Simulation(cfg, 3)
        obs = sim.observe(0)
        self.assertTrue(obs.active)
        self.assertEqual(obs.chaser_ids.tolist(), [0])
        self.assertEqual(len(obs.target_ids), 0)
        controller = guidance.SwarmController(0, cfg, 'coordinated')
        controller.update_telemetry(obs)
        command = controller.get_control_commands()
        self.assertEqual(command.shape, (3,))
        self.assert_braking(controller)
        self.assertIsNone(controller.target_id)

    def test_controller_commands_are_bounded(self):
        cfg = Config(n_chasers=2, n_targets=3)
        sim = Simulation(cfg, 7)
        for mode in ('greedy', 'coordinated'):
            controller = guidance.SwarmController(0, cfg, mode)
            controller.update_telemetry(sim.observe(0))
            if mode == 'coordinated':
                negotiate([controller])
            command = controller.get_control_commands()
            self.assertEqual(command.shape, (3,))
            self.assertTrue(np.isfinite(command).all())
            self.assertLessEqual(np.linalg.norm(command), cfg.a_max+1e-9)

    def test_reacquisition_immediately_requests_fresh_negotiation(self):
        cfg = Config(n_chasers=1, n_targets=1)
        obs = packet(0, [0], [0])
        controller = guidance.SwarmController(0, cfg)
        controller.update_telemetry(obs)
        negotiate([controller])
        controller.update_telemetry(replace(obs, time=.02,
            target_ids=np.empty(0, dtype=int), target_pos=np.empty((0, 3)),
            target_vel=np.empty((0, 3))))
        self.assertIsNone(controller.target_id)
        controller.update_telemetry(replace(obs, time=.04))
        self.assertTrue(controller.allocation_due)
        negotiate([controller], epoch=2)
        self.assertEqual(controller.target_id, 0)

    def test_isometric_projection_keeps_depth_and_altitude_distinct(self):
        points = np.array([[0., 0., 5.], [0., 0., 6.], [0., 1., 5.]])
        projected = project_isometric(points, ground_z=5., scale=2., origin=(100., 100.))
        np.testing.assert_allclose(projected,
            [[100., 100.], [100., 98.], [98.2679491924, 101.]], atol=1e-9)


    def test_plant_rejects_planar_or_nonfinite_commands(self):
        sim = Simulation(Config(n_chasers=1, n_targets=1), 1)
        for invalid in (np.zeros((1, 2)), np.full((1, 3), np.nan)):
            with self.assertRaises(ValueError):
                sim.step(invalid)


class TestObservationPlant(unittest.TestCase):
    def fixture(self, chasers=1, targets=1, dwell=.06):
        cfg = Config(n_chasers=chasers, n_targets=targets, service_duration=dwell,
                     position_noise=0., velocity_noise=0.)
        sim = Simulation(cfg, seed=12)
        sim.chaser_pos[:] = [20., 30., 30.]
        sim.chaser_vel[:] = 0.
        sim.target_pos[:] = [30., 30., 30.]
        sim.target_vel[:] = 0.
        frozen = patch.object(sim, '_advance_targets', return_value=None)
        frozen.start()
        self.addCleanup(frozen.stop)
        return sim

    def test_observation_requires_dwell_and_craft_can_service_again(self):
        sim = self.fixture(targets=2)
        sim.target_pos[1] = [70., 30., 30.]
        for _ in range(2):
            sim.step(np.zeros((1, 3)))
            self.assertEqual(sim.stats()['serviced'], 0)
        sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.status.tolist(), [1, 0])
        self.assertTrue(sim.chaser_active.all())
        self.assertEqual(sim.stats()['retired_chasers'], 0)
        sim.chaser_pos[0] = [60., 30., 30.]
        for _ in range(3):
            sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.stats()['serviced'], 2)
        self.assertEqual(len(sim.service_times), 2)
        self.assertTrue(sim.chaser_active.all())

    def test_breaking_observation_band_resets_continuous_dwell(self):
        sim = self.fixture(dwell=.1)
        for _ in range(4):
            sim.step(np.zeros((1, 3)))
        sim.chaser_pos[:] = [0., 30., 30.]
        sim.step(np.zeros((1, 3)))
        np.testing.assert_array_equal(sim.service_dwell, 0.)
        sim.chaser_pos[:] = [20., 30., 30.]
        for _ in range(4):
            sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.stats()['serviced'], 0)
        sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.stats()['serviced'], 1)

    def test_contact_never_counts_as_service_or_consumes_craft(self):
        sim = self.fixture()
        for _ in range(10):
            # Reintroduce contact each tick. MuJoCo otherwise pushes overlapping
            # bodies apart, legitimately allowing fresh dwell after recovery.
            sim.chaser_pos[:] = sim.target_pos
            sim.chaser_vel[:] = 0.
            sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.stats()['serviced'], 0)
        self.assertTrue(sim.chaser_active.all())
        np.testing.assert_array_equal(sim.service_dwell, 0.)
        self.assertGreater(sim.stats()['near_contact_violations'], 0)

    def test_dwell_is_not_pooled_between_observers(self):
        sim = self.fixture(chasers=2)
        sim.chaser_pos[1] = [0., 30., 30.]
        for _ in range(2):
            sim.step(np.zeros((2, 3)))
        sim.chaser_pos[0], sim.chaser_pos[1] = [0., 30., 30.], [20., 30., 30.]
        for _ in range(2):
            sim.step(np.zeros((2, 3)))
        self.assertEqual(sim.stats()['serviced'], 0)
        sim.step(np.zeros((2, 3)))
        self.assertEqual(sim.stats()['serviced'], 1)
        self.assertEqual(sim.service_agent[0], 1)

    def test_each_craft_observes_one_target_and_service_count_is_unique(self):
        sim = self.fixture(chasers=2, targets=2, dwell=.02)
        sim.target_pos[1] = [31., 30., 30.]
        sim.step(np.zeros((2, 3)))
        self.assertEqual(sim.stats()['serviced'], 1)
        self.assertEqual(len(sim.service_times), 1)
        sim.step(np.zeros((2, 3)))
        self.assertEqual(sim.stats()['serviced'], 2)
        self.assertTrue(sim.chaser_active.all())

    def test_completed_target_remains_visible_for_separation_only(self):
        sim = self.fixture(dwell=.02)
        sim.step(np.zeros((1, 3)))
        obs = sim.observe(0)
        self.assertTrue(obs.active)
        self.assertEqual(obs.target_ids.tolist(), [0])
        self.assertEqual(obs.target_available.tolist(), [False])
        controller = guidance.SwarmController(0, sim.config, 'greedy')
        controller.update_telemetry(obs)
        self.assertIsNone(controller.target_id)

    def test_perimeter_crossing_is_escape_and_not_ground_crossing(self):
        sim = self.fixture(dwell=.1)
        sim.target_pos[:] = [sim.config.width-.05, 30., 30.]
        sim.chaser_pos[:] = [sim.config.width-10., 30., 30.]
        with patch.object(sim, '_advance_targets', side_effect=lambda: sim.target_pos.__iadd__([.1, 0., 0.])):
            sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.status.tolist(), [2])
        self.assertEqual(sim.stats()['serviced'], 0)
        self.assertAlmostEqual(sim.target_event_pos[0, 0], sim.config.width)

    def test_strict_outer_boundary_resets_dwell(self):
        sim = self.fixture(dwell=.1)
        sim.service_dwell[0, 0] = .08
        sim.chaser_pos[:] = [14., 30., 30.]
        sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.stats()['serviced'], 0)
        self.assertEqual(sim.service_dwell[0, 0], 0.)

    def test_swept_inner_crossing_cannot_hide_between_safe_endpoints(self):
        cfg = Config(n_chasers=1, n_targets=1, a_max=2000., v_max=2000.,
                     service_duration=.1)
        sim = Simulation(cfg, 5)
        sim.chaser_pos[:] = [20., 30., 30.]
        sim.chaser_vel[:] = [1000., 0., 0.]
        sim.target_pos[:] = [30., 30., 30.]
        sim.service_dwell[0, 0] = .09
        with patch.object(sim, '_advance_targets', return_value=None):
            sim.step(np.array([[1000., 0., 0.]]))
        self.assertEqual(sim.stats()['serviced'], 0)
        self.assertGreaterEqual(np.linalg.norm(sim.chaser_pos[0]-sim.target_pos[0]), cfg.min_separation)
        self.assertLess(sim.service_dwell[0, 0], cfg.dt)
        self.assertGreater(sim.stats()['near_contact_violations'], 0)

    def test_escape_wins_a_simultaneous_dwell_completion(self):
        sim = self.fixture(dwell=.04)
        sim.target_pos[:] = [99.875, 30., 30.]
        sim.chaser_pos[:] = [90., 30., 30.]
        sim.service_dwell[0, 0] = .03
        with patch.object(sim, '_advance_targets', side_effect=lambda: sim.target_pos.__iadd__([.25, 0., 0.])):
            sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.status.tolist(), [2])
        self.assertEqual(sim.stats()['serviced'], 0)

    def test_target_speed_altitude_and_smooth_maneuvers(self):
        sim = Simulation(Config(n_chasers=1, n_targets=4), 23)
        initial_velocity = sim.target_vel.copy()
        for _ in range(750):
            previous_velocity = sim.target_vel.copy()
            sim.step(np.zeros((1, 3)))
            speed = np.linalg.norm(sim.target_vel, axis=1)
            self.assertGreaterEqual(speed.min(), 3.-1e-8)
            self.assertLessEqual(speed.max(), 7.+1e-8)
            self.assertGreaterEqual(sim.target_pos[:, 2].min(), sim.config.ground_z+20.-1e-8)
            self.assertLessEqual(sim.target_pos[:, 2].max(), sim.config.ground_z+60.+1e-8)
            self.assertLess(np.linalg.norm(sim.target_vel-previous_velocity, axis=1).max(), .25)
        self.assertFalse(np.allclose(sim.target_vel, initial_velocity))

    def test_observation_reachability_uses_band_and_perimeter(self):
        result = estimate_pairs(np.array([[0., 0., 30.]]), np.array([[2., 0., 0.]]),
                                np.array([[20., 0., 30.]]), np.array([[0., 0., 0.]]),
                                r_dock=10., sigma_base=1., zem_scale=.1,
                                ground_z=0., min_separation=5., perimeter_x=100.)
        self.assertAlmostEqual(result.closing[0, 0], 2.)
        self.assertAlmostEqual(result.t_go[0, 0], 5.)
        self.assertAlmostEqual(result.zem[0, 0], 10.)
        self.assertAlmostEqual(result.probability[0, 0], 1-np.exp(-100./8.))
        stationary = estimate_pairs(np.array([[0., 0., 30.]]), np.zeros((1, 3)),
                                    np.array([[8., 0., 30.], [2., 0., 30.]]), np.zeros((2, 3)),
                                    r_dock=10., sigma_base=1., zem_scale=.1,
                                    ground_z=0., min_separation=5., perimeter_x=100.)
        self.assertGreater(stationary.probability[0, 0], 0.)
        self.assertEqual(stationary.probability[0, 1], 0.)
        escape = estimate_pairs(np.array([[0., 0., 30.]]), np.array([[6., 0., 0.]]),
                                np.array([[99., 0., 30.]]), np.array([[5., 0., 0.]]),
                                r_dock=10., sigma_base=1., zem_scale=.1,
                                ground_z=0., min_separation=5., perimeter_x=100.)
        self.assertEqual(escape.probability[0, 0], 0.)


class TestGNNPolicy(unittest.TestCase):
    def setUp(self):
        import gnn_model
        self.gnn = gnn_model
        torch.manual_seed(19)
        self.model = gnn_model.SwarmGNN(hidden_dim=16).eval()
        self.features = torch.randn(3, 4, 10)
        self.features[..., -1] = torch.rand(3, 4)
        self.edges = torch.ones((3, 3), dtype=torch.bool)

    def test_exact_leakage_math_masks_and_finite_gradient(self):
        logits = torch.zeros((2, 2), requires_grad=True)
        self.assertAlmostEqual(float(self.gnn.service_leakage_loss(logits, torch.full((2, 2), .5)).detach()), 1.125)
        mask = torch.tensor([[True, False], [False, False]])
        x = self.gnn.masked_softmax(logits, mask)
        torch.testing.assert_close(x, torch.tensor([[1., 0.], [0., 0.]]))
        self.assertAlmostEqual(float(self.gnn.service_leakage_loss(logits, torch.ones(2, 2), mask).detach()), 0.)
        out = self.model(self.features, self.edges)
        loss = self.gnn.service_leakage_loss(out, self.features[..., -1])
        loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.parameters()), 0.)

    def test_two_layers_and_craft_target_permutation_equivariance(self):
        self.assertEqual(len(self.model.message_layers), 2)
        original = self.model(self.features, self.edges)
        craft_order = torch.tensor([2, 0, 1])
        target_order = torch.tensor([3, 1, 0, 2])
        changed = self.model(self.features[craft_order][:, target_order],
                             self.edges[craft_order][:, craft_order])
        torch.testing.assert_close(changed, original[craft_order][:, target_order])

    def test_disconnected_nodes_cannot_send_messages_but_neighbors_can(self):
        changed = self.features.clone()
        changed[1] += 2.
        isolated = torch.eye(3, dtype=torch.bool)
        torch.testing.assert_close(self.model(self.features, isolated)[0],
                                   self.model(changed, isolated)[0], rtol=0, atol=0)
        difference = self.model(self.features, self.edges)[0]-self.model(changed, self.edges)[0]
        self.assertGreater(float(difference.abs().max().detach()), 1e-6)

    def test_sensor_graph_uses_noisy_track_geometry_and_keeps_column_alignment(self):
        obs = packet(0, [0, 1, 2], [4, 2])
        obs = replace(obs, target_available=np.array([False, True]))
        cfg = Config(broadcast_range=1.1)
        features, edges, estimates = self.gnn.build_sensor_graph(obs, cfg)
        self.assertEqual(tuple(features.shape), (3, 2, 10))
        torch.testing.assert_close(edges, torch.tensor([[True, True, False],
                                   [True, True, True], [False, True, True]]))
        self.assertTrue(torch.all(features[:, 0, -1] == 0))
        np.testing.assert_array_equal(estimates.probability[:, 0], 0.)
        expected_r = (obs.target_pos[1]-obs.chaser_pos[0])/100.
        np.testing.assert_allclose(features[0, 1, :3].numpy(), expected_r)
        _, disconnected, _ = self.gnn.build_sensor_graph(obs, replace(cfg, broadcast_range=.5))
        torch.testing.assert_close(disconnected, torch.eye(3, dtype=torch.bool))
        choices = self.gnn.masked_softmax(self.model(features, edges), self.gnn.sensor_pair_mask(obs))
        torch.testing.assert_close(choices[:, 0], torch.zeros(3))

    def test_checkpoint_roundtrip_and_invalid_or_missing_weights_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'output') as folder:
            path = Path(folder)/'unit-policy.pt'
            with self.assertRaises(FileNotFoundError):
                self.gnn.load_policy(path)
            self.gnn.save_checkpoint(self.model, path, trained_epochs=2, metadata={'task': 'observation'})
            loaded = self.gnn.load_policy(path)
            self.assertFalse(loaded.training)
            self.assertEqual(loaded.checkpoint_metadata['trained_epochs'], 2)
            torch.testing.assert_close(loaded(self.features, self.edges), self.model(self.features, self.edges))
            checkpoint = torch.load(path, weights_only=True)
            checkpoint['architecture'] = 'incompatible'
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                self.gnn.load_policy(path)

    def test_adam_training_changes_real_weights_and_reduces_fixed_holdout_loss(self):
        from train_gnn import train_policy
        model, report = train_policy(epochs=8, seed=19, batch_size=12, holdout_size=32, hidden_dim=16)
        self.assertEqual(report['optimizer_steps'], 8)
        self.assertEqual(report['training_snapshots'], 96)
        self.assertGreater(report['weight_delta_l2'], 0.)
        self.assertEqual(report['changed_parameter_tensors'], report['total_parameter_tensors'])
        self.assertTrue(report['weights_finite'])
        self.assertLess(report['final_holdout']['soft_leakage_loss'],
                        report['initial_holdout']['soft_leakage_loss'])
        self.assertFalse(model.training)

    def test_controller_runs_one_forward_per_due_cycle_and_never_bids(self):
        cfg = Config(n_chasers=3, n_targets=4)
        sim = Simulation(cfg, 7)
        controller = guidance.SwarmController(0, cfg, 'gnn')
        # Hooks observe real forwards rather than replacing inference outputs.
        policy = controller.model
        calls = []
        handle = policy.register_forward_hook(lambda *args: calls.append(True))
        self.addCleanup(handle.remove)
        obs = sim.observe(0)
        controller.update_telemetry(obs)
        self.assertEqual(len(calls), 1)
        self.assertIn(controller.target_id, obs.target_ids)
        controller.update_telemetry(replace(obs, time=.02))
        self.assertEqual(len(calls), 1)
        controller.update_telemetry(replace(obs, time=cfg.allocation_period))
        self.assertEqual(len(calls), 2)
        controller.update_telemetry(replace(obs, time=cfg.allocation_period+.02,
                                           target_available=np.zeros(cfg.n_targets, dtype=bool)))
        self.assertIsNone(controller.target_id)
        with self.assertRaises(RuntimeError):
            controller.begin_allocation(1)

    def test_three_mode_benchmark_reconciles_service_and_uses_real_inference(self):
        cfg = Config()
        results = {r['mode']: r for r in run_headless(cfg, seed=7)}
        self.assertEqual(set(results), {'greedy', 'auction', 'gnn'})
        self.assertEqual(results['auction']['max_duplicate_commitments'], 0)
        self.assertGreater(results['auction']['allocation_rounds'], 0)
        self.assertEqual(results['greedy']['max_duplicate_commitments'], 0)
        self.assertGreater(results['gnn']['gnn_forward_calls'], 0)
        self.assertEqual(results['gnn']['allocation_rounds'], 0)
        for result in results.values():
            self.assertEqual(result['active_chasers'], cfg.n_chasers)
            self.assertEqual(result['retired_chasers'], 0)
            self.assertEqual(result['serviced']+result['escaped']+result['pending'], cfg.n_targets)
            self.assertGreater(result['serviced'], 0)
            self.assertEqual(result['max_capacity_violations'], 0)
            self.assertEqual(result['max_duplicate_commitments'], 0)
            self.assertEqual(result['near_contact_violations'], 0)
            self.assertGreaterEqual(result['minimum_separation'], cfg.min_separation)


class TestContinuousScenario(unittest.TestCase):
    def test_minimum_separation_keeps_observation_outside_body_contact(self):
        for minimum in (.5, .7):
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, 'body diameter'):
                Config(min_separation=minimum)
        self.assertEqual(Config(min_separation=1.).min_separation, 1.)

    def test_cli_safe_distance_updates_export_and_service_band(self):
        import contextlib
        import io
        import json
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'output') as folder:
            output = Path(folder)/'safe-distance.json'
            arguments = ['main.py', '--headless', '--mode', 'greedy', '--duration', '.04',
                         '--chasers', '1', '--targets', '1', '--safe-distance', '12', '--json', str(output)]
            with patch('sys.argv', arguments), contextlib.redirect_stdout(io.StringIO()):
                harness.main()
            cfg = Config(**json.loads(output.read_text())['config'])
        self.assertEqual((cfg.min_separation, cfg.stand_off_distance, cfg.r_dock), (12., 17., 23.))
        cfg = replace(cfg, service_duration=.04, thrust_tau=0., drag_area=0.)
        for distance, expected_service in ((11., 0), (17., 1)):
            sim = Simulation(cfg, 7)
            sim.chaser_pos[:] = [20., 30., 40.]
            sim.target_pos[:] = [20.+distance, 30., 40.]
            sim.chaser_vel[:] = sim.target_vel[:] = 0.
            with patch.object(sim, '_advance_targets', return_value=None):
                sim.step(np.zeros((1, 3)))
                sim.step(np.zeros((1, 3)))
            self.assertEqual(sim.stats()['serviced'], expected_service)

    def test_cli_explicit_observation_distances_and_invalid_bands(self):
        import contextlib
        import io
        import json
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'output') as folder:
            output = Path(folder)/'band.json'
            arguments = ['main.py', '--headless', '--mode', 'greedy', '--duration', '.02',
                         '--safe-distance', '7', '--stand-off-distance', '12',
                         '--observation-radius', '19', '--json', str(output)]
            with patch('sys.argv', arguments), contextlib.redirect_stdout(io.StringIO()):
                harness.main()
            cfg = json.loads(output.read_text())['config']
            self.assertEqual((cfg['min_separation'], cfg['stand_off_distance'], cfg['r_dock']), (7., 12., 19.))
        for flags in (['--safe-distance', '0'], ['--safe-distance', 'nan'],
                      ['--safe-distance', '12', '--stand-off-distance', '10'],
                      ['--stand-off-distance', '12', '--observation-radius', '11']):
            with self.subTest(flags=flags), patch('sys.argv', ['main.py', '--headless', *flags]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    harness.main()
                self.assertEqual(error.exception.code, 2)

    def test_auction_relaunch_joins_current_epoch_before_status_broadcast(self):
        cfg = Config(continuous=True, duration=5., recycle_on_service=True,
                     v_max=14., cruise_speed=14.,
                     profile_speeds=(('weave', 4., 5.), ('constant-height', 6., 6.)))
        scenario = harness.Scenario(cfg, seed=7, mode='auction')
        while not scenario.sim.done:
            scenario.step()
        self.assertGreater(scenario.sim.relaunches, 0)
        self.assertEqual(scenario.max_capacity_violations, 0)
        for i in np.flatnonzero(scenario.sim.chaser_generation):
            if scenario.sim.chaser_active[i]:
                self.assertEqual(scenario.controllers[i]._epoch, scenario.allocation_epoch)

    def test_cli_exposes_observer_and_per_profile_object_speeds(self):
        arguments = ['main.py', '--headless', '--mode', 'greedy', '--duration', '.1',
                     '--chaser-speed', '9', '--profile-speed', 'weave', '4', '5',
                     '--profile-speed', 'constant-height', '6', '6']
        with patch('sys.argv', arguments), patch.object(harness, 'run_headless', return_value=[]) as runner:
            harness.main()
        cfg = runner.call_args.args[0]
        self.assertTrue(cfg.recycle_on_service)
        self.assertEqual(cfg.chasers_per_target, 1)
        self.assertEqual((cfg.v_max, cfg.cruise_speed), (9., 9.))
        self.assertEqual(cfg.speed_range('weave'), (4., 5.))
        self.assertEqual(cfg.speed_range('constant-height'), (6., 6.))
        self.assertEqual(cfg.speed_range('random'), (3., 7.))

    def test_relaunched_observer_rejoins_control_and_radio_with_fresh_assignment(self):
        for mode in ('greedy', 'auction', 'gnn'):
            with self.subTest(mode=mode):
                cfg = Config(n_chasers=2, n_targets=2, continuous=True, duration=1.,
                             max_tracks=2, recycle_on_service=True, relaunch_delay=.08,
                             service_duration=.04, arrival_interval=5.,
                             position_noise=0., velocity_noise=0., drag_area=0., thrust_tau=0.)
                scenario = harness.Scenario(cfg, seed=7, mode=mode)
                sim = scenario.sim
                sim.chaser_pos[:] = [[20., 10., 40.], [20., 40., 40.]]
                sim.chaser_vel[:] = 0.
                sim.target_pos[:] = [[30., 10., 40.], [70., 40., 40.]]
                sim.target_vel[:] = 0.
                # Seed a legitimate incumbent pair so this lifecycle test is
                # independent of the three policies' initial preferences.
                for controller in scenario.controllers:
                    controller.update_telemetry(sim.observe(controller.agent_id))
                    controller.accept_reservation(controller.agent_id)
                scenario.reservations.assign(scenario.controllers)
                with patch.object(sim, '_advance_targets', return_value=None):
                    for _ in range(2):
                        scenario.step()
                    self.assertFalse(sim.chaser_active[0])
                    self.assertFalse(sim.target_present[0])
                    self.assertIsNone(scenario.controllers[0].target_id)
                    np.testing.assert_array_equal(scenario.controllers[0].get_control_commands(), 0.)
                    self.assertIn(0, scenario.mesh.withdrawn_senders)
                    model = scenario.controllers[0].model
                    previous_forwards = scenario.controllers[0].gnn_forward_calls
                    for _ in range(8):
                        scenario.step()
                    self.assertTrue(sim.chaser_active[0])
                    self.assertEqual(sim.chaser_generation[0], 1)
                    self.assertNotIn(0, scenario.mesh.withdrawn_senders)
                    self.assertFalse(scenario.controllers[0]._retired)
                    self.assertIs(scenario.controllers[0].model, model)
                    self.assertGreaterEqual(scenario.controllers[0].gnn_forward_calls, previous_forwards)
                    self.assertTrue(scenario.controllers[1].peer_buffer[0].active)
                    self.assertEqual(scenario.capacity_violations, 0)

    def test_export_records_effective_settings_after_interactive_restart(self):
        import json
        from dataclasses import asdict
        initial = Config(continuous=True, duration=None)
        effective = replace(initial, chasers_per_target=2, trajectory='parabolic')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'output') as folder:
            path = Path(folder)/'session.json'
            harness.write_results(path, initial, [{'mode': 'gnn', 'config': asdict(effective)}])
            data = json.loads(path.read_text())
        self.assertEqual(data['config']['chasers_per_target'], 2)
        self.assertEqual(data['config']['trajectory'], 'parabolic')
        self.assertEqual(data['launch_config']['chasers_per_target'], 1)

    def test_service_latency_is_measured_from_new_object_birth(self):
        cfg = Config(n_chasers=1, n_targets=1, continuous=True, max_tracks=2,
                     arrival_interval=.04, service_duration=.04, drag_area=0., thrust_tau=0.)
        sim = Simulation(cfg, 7)
        sim.target_pos[0] = [500., 30., 40.]
        for _ in range(3):
            sim.step(np.zeros((1, 3)))
        slot = sim.target_index(1)
        self.assertIsNotNone(slot)
        sim.chaser_pos[:] = [[10., 10., 40.]]
        sim.chaser_vel[:] = 0.
        sim.target_pos[slot] = [20., 10., 40.]
        sim.target_vel[slot] = 0.
        sim.set_observation_assignments({0: 1})
        with patch.object(sim, '_advance_targets', return_value=None):
            sim.step(np.zeros((1, 3)))
            sim.step(np.zeros((1, 3)))
        self.assertEqual(sim.stats()['serviced'], 1)
        self.assertAlmostEqual(sim.service_times[-1], .10)
        self.assertAlmostEqual(sim.stats()['mean_service_time'], .06)

    def test_every_policy_obeys_configurable_capacity_even_without_radio_links(self):
        for mode in ('greedy', 'auction', 'gnn'):
            for capacity in (1, 2):
                with self.subTest(mode=mode, capacity=capacity):
                    cfg = Config(n_chasers=4, n_targets=2, broadcast_range=.01,
                                 chasers_per_target=capacity)
                    scenario = harness.Scenario(cfg, seed=7, mode=mode)
                    for _ in range(8):
                        scenario.step()
                        from collections import Counter
                        counts = Counter(scenario.active_commitments)
                        self.assertTrue(all(n <= capacity for n in counts.values()))
                        self.assertEqual(scenario.capacity_violations, 0)
                    self.assertEqual(scenario.result()['max_capacity_violations'], 0)

    def test_new_global_ids_are_allocated_during_a_continuous_session(self):
        cfg = Config(n_chasers=3, n_targets=1, continuous=True, duration=1.,
                     arrival_interval=.2, max_tracks=8, trajectory='constant-height')
        scenario = harness.Scenario(cfg, seed=7, mode='gnn')
        observed_ids = set()
        while not scenario.sim.done:
            scenario.step()
            for controller in scenario.controllers:
                observed_ids.update(controller.obs.target_ids.tolist())
                if controller.target_id is not None:
                    self.assertIsNotNone(scenario.sim.target_index(controller.target_id))
            self.assertEqual(scenario.capacity_violations, 0)
        self.assertGreater(max(observed_ids), cfg.n_targets-1)
        self.assertGreater(scenario.sim.stats()['arrivals'], cfg.n_targets)
        self.assertEqual(scenario.result()['auction_rounds'], 0)

    def test_indefinite_sessions_are_interactive_only(self):
        cfg = Config(continuous=True, duration=None)
        sim = Simulation(cfg, 7)
        sim.status[:] = 1
        self.assertFalse(sim.done)
        with self.assertRaisesRegex(ValueError, 'finite duration'):
            run_headless(cfg, 7)


class TestOrbitCamera(unittest.TestCase):
    def test_orbit_keeps_volume_center_fixed_and_uses_both_axes(self):
        cfg = Config(ground_z=-4.)
        camera = harness.OrbitCamera(cfg, viewport=(38, 128, 870, 585))
        center = np.array([50., 30., 36.])
        np.testing.assert_allclose(camera.center, center)
        np.testing.assert_allclose(camera.project(center), [473., 420.5])
        point = center + [11., 7., 13.]
        initial = camera.project(point)
        camera.orbit(60., 0.)
        horizontal = camera.project(point)
        self.assertFalse(np.allclose(horizontal, initial))
        camera.orbit(0., 35.)
        self.assertFalse(np.allclose(camera.project(point), horizontal))
        np.testing.assert_array_equal(camera.center, center)
        np.testing.assert_allclose(camera.project(center), [473., 420.5])

    def test_pan_moves_world_by_screen_pixels_without_rotating_vectors(self):
        camera = harness.OrbitCamera(Config())
        points = np.array([[5., 8., 11.], [23., 31., 49.]])
        vectors = np.array([[3., -4., 5.], [-1., 8., 2.]])
        saved_points, saved_vectors = points.copy(), vectors.copy()
        before, vector_before = camera.project(points), camera.project_vector(vectors)
        depths = camera.depth(points)
        camera.pan(37., -22.)
        np.testing.assert_allclose(camera.project(points)-before, [[37., -22.], [37., -22.]])
        np.testing.assert_allclose(camera.project_vector(vectors), vector_before)
        np.testing.assert_allclose(camera.depth(points), depths)
        np.testing.assert_allclose(camera.project(points+vectors)-camera.project(points), vector_before)
        np.testing.assert_array_equal(points, saved_points)
        np.testing.assert_array_equal(vectors, saved_vectors)

    def test_zoom_scales_about_center_and_camera_bounds_remain_finite(self):
        camera = harness.OrbitCamera(Config())
        points = camera.center + np.array([[15., 3., 4.], [-8., 2., 11.]])
        center_pixel = camera.project(camera.center)
        before = camera.project(points)-center_pixel
        initial_scale, initial_zoom = camera.scale, camera.zoom
        camera.zoom_by(3)
        self.assertGreater(camera.zoom, initial_zoom)
        np.testing.assert_allclose(camera.project(points)-center_pixel,
                                   before*(camera.scale/initial_scale))
        np.testing.assert_allclose(camera.project(camera.center), center_pixel)
        camera.zoom_by(1000)
        self.assertAlmostEqual(camera.zoom, 5.)
        camera.zoom_by(-1000)
        self.assertAlmostEqual(camera.zoom, .2)
        camera.orbit(0., 1e6)
        first_limit = camera.elevation
        camera.orbit(0., -1e6)
        self.assertAlmostEqual(min(first_limit, camera.elevation), np.deg2rad(5.))
        self.assertAlmostEqual(max(first_limit, camera.elevation), np.deg2rad(85.))
        self.assertTrue(np.isfinite(camera.project(points)).all())

    def test_projection_basis_is_orthonormal_and_depth_follows_view(self):
        camera = harness.OrbitCamera(Config())
        projected_basis = camera.project_vector(np.eye(3))
        np.testing.assert_allclose(projected_basis.T @ projected_basis,
                                   np.eye(2)*camera.scale**2, atol=1e-10)
        points = camera.center + np.array([[15., 0., 0.], [-15., 0., 0.]])
        depth_difference = np.diff(camera.depth(points))[0]
        self.assertNotAlmostEqual(depth_difference, 0.)
        camera.orbit(np.pi/.006, 0.)
        self.assertAlmostEqual(np.diff(camera.depth(points))[0], -depth_difference)

    def test_reset_restores_original_projection_after_all_camera_controls(self):
        camera = harness.OrbitCamera(Config())
        points = np.array([[0., 0., 0.], [100., 60., 80.], [35., 12., 48.]])
        original = camera.project(points)
        original_center = camera.center.copy()
        camera.orbit(77., 23.)
        camera.pan(-52., 31.)
        camera.zoom_by(4)
        self.assertFalse(np.allclose(camera.project(points), original))
        camera.reset()
        np.testing.assert_allclose(camera.project(points), original)
        np.testing.assert_array_equal(camera.center, original_center)
        self.assertEqual(camera.zoom, 1.)


# Native renderer input/layout regressions live in test_world_renderer.py.

if __name__ == '__main__':
    unittest.main()
