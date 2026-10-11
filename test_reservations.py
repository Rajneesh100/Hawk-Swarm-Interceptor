"""Capacity reservations and net-acceleration controller regressions."""
from dataclasses import replace
import unittest

import numpy as np

from estimator import Observation
from reservations import ReservationManager
from sim_engine import Config
from swarm_controller import PeerBroadcast, SwarmController


def controller(agent_id, count, target_ids=(101, 205), distances=(20., 30.), *, available=None):
    cfg = Config(n_chasers=count, n_targets=1, position_noise=0., velocity_noise=0.)
    craft = SwarmController(agent_id, cfg, 'greedy')
    obs = Observation(0., agent_id, np.array([agent_id]), np.array([[0., 0., 30.]]),
        np.array([[2., 0., 0.]]), np.array(target_ids, dtype=int),
        np.array([[distance, 0., 30.] for distance in distances]).reshape(-1, 3),
        np.zeros((len(target_ids), 3)), target_available=(None if available is None else np.array(available, dtype=bool)))
    craft.update_telemetry(obs)
    return craft


class TestReservations(unittest.TestCase):
    def test_default_capacity_assigns_alternatives_and_leaves_extra_craft_idle(self):
        craft = [controller(i, 3, distances=(20.+i, 30.+i)) for i in range(3)]
        manager = ReservationManager()
        self.assertEqual(manager.assign(craft), {0: 101, 1: 205, 2: None})
        self.assertEqual([c.target_id for c in craft], [101, 205, None])
        self.assertEqual(manager.peak_capacity_violations, 0)

    def test_capacity_two_and_stable_distinct_observation_slots(self):
        craft = [controller(i, 3, target_ids=(101,), distances=(20.+i,)) for i in range(3)]
        manager = ReservationManager(capacity=2)
        self.assertEqual(manager.assign(craft), {0: 101, 1: 101, 2: None})
        self.assertEqual([craft[i].reservation_slot for i in (0, 1)], [0, 1])
        for c in craft[:2]:
            c.obs.chaser_vel[:] = 0.
        first, second = craft[0].get_control_commands(), craft[1].get_control_commands()
        self.assertFalse(np.allclose(first, second))
        self.assertEqual(manager.assign(list(reversed(craft))), {0: 101, 1: 101, 2: None})
        self.assertEqual([craft[i].reservation_slot for i in (0, 1)], [0, 1])

    def test_valid_incumbent_survives_policy_rank_changes(self):
        craft = [controller(i, 2, distances=(20.+i, 30.+i)) for i in range(2)]
        manager = ReservationManager()
        initial = manager.assign(craft)
        changed = replace(craft[0].obs, time=1., target_pos=np.array([[50., 0., 30.], [10., 0., 30.]]))
        craft[0].update_telemetry(changed)
        self.assertEqual(craft[0].target_id, 205)  # Raw nearest proposal changed.
        self.assertEqual(manager.assign(craft), initial)
        self.assertEqual(craft[0].target_id, 101)

    def test_unavailable_and_unobserved_targets_release_without_reusing_old_ids(self):
        craft = [controller(0, 1)]
        manager = ReservationManager()
        self.assertEqual(manager.assign(craft), {0: 101})
        craft[0].update_telemetry(replace(craft[0].obs, time=1., target_available=np.array([False, True])))
        self.assertEqual(manager.assign(craft), {0: 205})
        craft[0].update_telemetry(replace(craft[0].obs, time=2.,
            target_ids=np.array([9999]), target_pos=np.array([[25., 0., 30.]]),
            target_vel=np.zeros((1, 3)), target_available=np.array([True])))
        self.assertEqual(manager.assign(craft), {0: 9999})
        craft[0].update_telemetry(replace(craft[0].obs, time=3., target_ids=np.empty(0, dtype=int),
            target_pos=np.empty((0, 3)), target_vel=np.empty((0, 3)), target_available=np.empty(0, dtype=bool)))
        self.assertEqual(manager.assign(craft), {0: None})

    def test_disconnected_local_views_still_share_explicit_mission_capacity(self):
        first = controller(0, 2, target_ids=(999,), distances=(20.,))
        second = controller(1, 2, target_ids=(999,), distances=(20.,))
        self.assertEqual(first.obs.chaser_ids.tolist(), [0])
        self.assertEqual(second.obs.chaser_ids.tolist(), [1])
        self.assertFalse(first.peer_buffer or second.peer_buffer)
        self.assertEqual(ReservationManager().assign([second, first]), {0: 999, 1: None})

    def test_more_targets_than_craft_uses_each_craft_once(self):
        craft = [controller(i, 2, target_ids=(50, 70, 90, 110), distances=(20., 30., 40., 50.)) for i in range(2)]
        result = ReservationManager().assign(craft)
        self.assertEqual(result, {0: 50, 1: 70})

    def test_gnn_reservations_use_cached_id_scores_without_new_forward(self):
        craft = [controller(i, 2) for i in range(2)]
        for c in craft:
            c.mode = 'gnn'
            c._gnn_scores = {101: 4.-c.agent_id, 205: 2.+c.agent_id}
            c.gnn_forward_calls = 1
        self.assertEqual(ReservationManager().assign(craft), {0: 101, 1: 205})
        self.assertEqual([c.gnn_forward_calls for c in craft], [1, 1])
        # A changed physical-track layout cannot shift cached logits to new IDs.
        c = craft[0]
        c.obs = replace(c.obs, target_ids=np.array([205, 777]))
        candidates = c.reservation_candidates()
        self.assertEqual([item[0] for item in candidates], [205])

    def test_equal_policy_scores_arbitrate_by_craft_id(self):
        craft = [controller(0, 2, target_ids=(101,), distances=(40.,)),
                 controller(1, 2, target_ids=(101,), distances=(20.,))]
        for c in craft:
            c.mode = 'gnn'
            c._gnn_scores = {101: 3.}
        self.assertEqual(ReservationManager().assign(list(reversed(craft))), {0: 101, 1: None})

    def test_monotonic_target_id_broadcast_accepts_ids_beyond_initial_population(self):
        c = controller(0, 2, target_ids=(10000,), distances=(20.,))
        c.receive_broadcast(PeerBroadcast(1, 0, 0, 0., 10000, .5))
        self.assertEqual(c.peer_buffer[1].target_id, 10000)
        c.receive_broadcast(PeerBroadcast(1, 0, 1, 0., -1, .5))
        self.assertEqual(c.peer_buffer[1].target_id, 10000)

    def test_unassigned_command_is_net_braking_acceleration(self):
        c = controller(0, 1, target_ids=(), distances=())
        c.obs.chaser_vel[:] = [[1., 0., 0.]]
        np.testing.assert_allclose(c.get_control_commands(), [-3.5, 0., 0.], atol=1e-12)

    def test_invalid_capacity_and_duplicate_controller_ids_are_rejected(self):
        for capacity in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ReservationManager(capacity=capacity)
        c = controller(0, 1)
        with self.assertRaises(ValueError):
            ReservationManager().assign([c, c])

    def test_launch_reset_preserves_loaded_policy_and_cumulative_metrics(self):
        source = controller(0, 2)
        c = SwarmController(0, source.config, 'gnn')
        c.update_telemetry(source.obs)
        policy = c.model
        totals = (c.gnn_forward_calls, c.gnn_forward_seconds, c.gnn_compute_seconds)
        c.make_broadcast(9)
        c.update_telemetry(replace(source.obs, time=1., active=False))

        c.reset_for_launch()
        self.assertIs(c.model, policy)
        self.assertEqual((c.gnn_forward_calls, c.gnn_forward_seconds, c.gnn_compute_seconds), totals)
        self.assertIsNone(c.obs)
        self.assertIsNone(c.estimates)
        self.assertIsNone(c.target_id)
        self.assertIsNone(c.gnn_last_logits)
        self.assertEqual(c.reservation_candidates(), [])

        fresh = replace(source.obs, time=2., target_ids=np.array([777]),
                        target_pos=np.array([[25., 0., 30.]]), target_vel=np.zeros((1, 3)))
        c.update_telemetry(fresh)
        self.assertEqual(c.target_id, 777)
        self.assertEqual([candidate[0] for candidate in c.reservation_candidates()], [777])
        self.assertEqual(c.gnn_forward_calls, totals[0]+1)
        with self.assertRaises(ValueError):
            c.make_broadcast(8)
        packet = c.make_broadcast(10)
        self.assertTrue(packet.active)
        self.assertEqual(packet.target_id, 777)

    def test_launch_reset_preserves_monotonic_auction_epoch(self):
        c = controller(0, 1)
        c.mode = 'auction'
        active_obs = c.obs
        c.begin_allocation(4)
        c.make_broadcast(6)
        c.update_telemetry(replace(active_obs, time=1., active=False))
        c.reset_for_launch()
        c.update_telemetry(replace(active_obs, time=2.))
        self.assertTrue(c.allocation_due)
        with self.assertRaises(ValueError):
            c.begin_allocation(4)
        c.begin_allocation(5)
        self.assertEqual(c.make_broadcast(0).epoch, 5)

    def test_relaunch_can_join_newer_transport_epoch_before_next_auction(self):
        c = controller(1, 2)
        c.mode = 'auction'
        active_obs = c.obs
        c.begin_allocation(7)
        c.make_broadcast(3)
        c.update_telemetry(replace(active_obs, time=1., active=False))
        c.reset_for_launch(epoch=12)
        c.update_telemetry(replace(active_obs, time=2.))
        c.accept_reservation(101)

        peer = controller(0, 2)
        peer.mode = 'auction'
        peer.update_telemetry(replace(peer.obs, time=2.))
        peer.begin_allocation(12)
        # Round 2 in epoch 12 is newer than round 3 in epoch 7.
        message = c.make_broadcast(2)
        self.assertEqual(message.epoch, 12)
        peer.receive_broadcast(message)
        self.assertEqual(peer.peer_buffer[1].target_id, 101)
        with self.assertRaises(ValueError):
            c.reset_for_launch(epoch=11)
        self.assertIsNotNone(c.obs)

    def test_trusted_peer_launch_forgets_only_that_incarnation(self):
        c = controller(0, 3)
        c.receive_broadcast(PeerBroadcast(1, 0, 5, 0., None, 0., False))
        c.receive_broadcast(PeerBroadcast(2, 0, 7, 0., 205, .4))
        c.update_telemetry(c.obs)
        self.assertFalse(c.allocation_due)
        c.receive_broadcast(PeerBroadcast(1, 0, 6, 0., 101, .5))
        self.assertNotIn(1, c.peer_buffer)

        c.forget_peer_incarnation(1)
        self.assertTrue(c.allocation_due)
        self.assertNotIn(1, c.retired_peer_ids)
        self.assertNotIn(1, c._last_peer_rounds)
        self.assertEqual(c.peer_buffer[2].target_id, 205)
        c.receive_broadcast(PeerBroadcast(1, 0, 0, 0., 101, .5))
        self.assertEqual(c.peer_buffer[1].target_id, 101)

    def test_inactive_bay_cannot_allocate_or_command_until_launch_reset(self):
        c = controller(0, 2)
        obs = replace(c.obs, active=False, chaser_ids=np.array([0, 1]),
                      chaser_pos=np.array([[0., 0., 30.], [0., 20., 30.]]),
                      chaser_vel=np.zeros((2, 3)))
        c.update_telemetry(obs)
        self.assertFalse(c.allocation_due)
        self.assertEqual(c.reservation_candidates(), [])
        self.assertEqual(ReservationManager().assign([c]), {0: None})
        np.testing.assert_array_equal(c.get_control_commands(), np.zeros(3))
        c.update_telemetry(replace(obs, time=1., active=True))
        self.assertIsNone(c.target_id)
        self.assertFalse(c.make_broadcast(1).active)
        c.reset_for_launch()
        c.update_telemetry(replace(obs, time=2., active=True))
        self.assertEqual(c.target_id, 101)

    def test_forgetting_relaunched_bay_releases_incumbent_priority(self):
        craft = [controller(0, 2, target_ids=(101,), distances=(20.,)),
                 controller(1, 2, target_ids=(101,), distances=(30.,))]
        manager = ReservationManager()
        self.assertEqual(manager.assign(craft), {0: 101, 1: None})
        craft[0].update_telemetry(replace(craft[0].obs, time=1., target_pos=np.array([[50., 0., 30.]])))
        craft[1].update_telemetry(replace(craft[1].obs, time=1., target_pos=np.array([[10., 0., 30.]])))
        manager.forget_agent(0)
        self.assertNotIn(0, manager.assignments)
        self.assertNotIn(0, manager.slots)
        self.assertEqual(manager.assign(craft), {0: None, 1: 101})

    def test_configured_speed_caps_approach_correction_for_both_slot_modes(self):
        # A target approaching at 3 m/s plus a capped 4 m/s radial correction
        # requests 1 m/s forward, not the old 10-3=7 m/s reference.
        for capacity in (1, 2):
            with self.subTest(capacity=capacity):
                outputs = []
                for speed in (4., 12.):
                    c = controller(0, 1, target_ids=(101,), distances=(40.,))
                    c.config = replace(c.config, v_max=speed, cruise_speed=10., a_max=40.)
                    c.obs.chaser_vel[:] = 0.
                    c.obs.target_vel[:] = [-3., 0., 0.]
                    c.accept_reservation(101, capacity=capacity)
                    outputs.append(c.get_control_commands())
                if capacity == 1:
                    np.testing.assert_allclose(outputs[0], [3.5, 0., 0.], atol=1e-10)
                    np.testing.assert_allclose(outputs[1], [24.5, 0., 0.], atol=1e-10)
                else:
                    # Unit direction to offset point (40,10) is (4,1)/sqrt(17).
                    np.testing.assert_allclose(outputs[0], [3.082,
                        3.3955, 0.], atol=.001)
                self.assertLess(np.linalg.norm(outputs[0]), np.linalg.norm(outputs[1]))


if __name__ == '__main__':
    unittest.main()
