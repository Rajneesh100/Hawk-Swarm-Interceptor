"""Display-independent input, settings and read-only world-view regressions."""
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from sim_engine import Config
from main import Scenario
from world_renderer import Renderer


class TestWorldRenderer(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, SDL_VIDEODRIVER='dummy')
        self.environment.start()
        # These tests exercise the supported display-less path deterministically.
        self.no_gl = patch('mujoco.Renderer', side_effect=RuntimeError('test without OpenGL'))
        self.no_gl.start()
        self.cfg = Config(n_chasers=2, n_targets=2, max_tracks=4, continuous=True)
        self.ui = Renderer(self.cfg, fullscreen=False)
        self.pg = self.ui.pg
        self.scenario = Scenario(self.cfg, 7, 'auction')
        self.scenario.step()

    def tearDown(self):
        self.ui.close()
        self.no_gl.stop()
        self.environment.stop()

    def event(self, kind, **attrs):
        return self.ui.handle_event(self.pg.event.Event(kind, attrs))

    def drag(self, button=1, mod=0, delta=(40, 20)):
        origin = self.ui.viewport.center
        self.event(self.pg.MOUSEBUTTONDOWN, button=button, pos=origin, mod=mod)
        buttons = (button == 1, False, button == 3)
        self.event(self.pg.MOUSEMOTION, rel=delta, pos=(origin[0]+delta[0], origin[1]+delta[1]), buttons=buttons)

    def click_action(self, action):
        rect = next(rect for rect, value in self.ui._buttons if value == action)
        self.event(self.pg.MOUSEBUTTONDOWN, button=1, pos=rect.center)

    def test_orbit_pan_zoom_reset_and_focus_release(self):
        initial = (self.ui.camera.azimuth, self.ui.camera.elevation)
        center = self.ui.camera.lookat.copy()
        self.drag()
        self.assertNotEqual(initial, (self.ui.camera.azimuth, self.ui.camera.elevation))
        np.testing.assert_array_equal(self.ui.camera.lookat, center)
        self.event(self.pg.WINDOWFOCUSLOST)
        self.assertIsNone(self.ui._drag_mode)
        after_focus = self.ui.camera.azimuth
        self.event(self.pg.MOUSEMOTION, rel=(30, 20), pos=(600, 300), buttons=(True, False, False))
        self.assertEqual(self.ui.camera.azimuth, after_focus)
        self.drag(button=3)
        self.assertFalse(np.array_equal(center, self.ui.camera.lookat))
        self.event(self.pg.MOUSEBUTTONUP, button=3, pos=(0, 0))
        self.assertIsNone(self.ui._drag_mode)
        center = self.ui.camera.lookat.copy()
        self.drag(mod=self.pg.KMOD_SHIFT)
        self.assertFalse(np.array_equal(center, self.ui.camera.lookat))
        self.event(self.pg.MOUSEWHEEL, y=1000)
        self.assertEqual(self.ui.camera.distance, 15.)
        self.event(self.pg.MOUSEWHEEL, y=-1000)
        self.assertEqual(self.ui.camera.distance, 500.)
        self.event(self.pg.KEYDOWN, key=self.pg.K_HOME)
        self.assertEqual(self.ui.camera.distance, 130.)
        self.assertEqual((self.ui.camera.azimuth, self.ui.camera.elevation), initial)
        self.assertIsNone(self.ui._drag_mode)

    def test_hud_and_outside_click_do_not_capture_camera(self):
        for position in ((20, 20), self.ui.telemetry_rect.center, (20, self.ui.screen.get_height()-10)):
            self.assertFalse(self.event(self.pg.MOUSEBUTTONDOWN, button=1, pos=position))
            self.assertIsNone(self.ui._drag_mode)
        self.drag()
        self.event(self.pg.MOUSEMOTION, rel=(3, 4), pos=(5, 5), buttons=(False, False, False))
        self.assertIsNone(self.ui._drag_mode)

    def test_camera_moves_world_without_mutating_plant_or_fixed_hud(self):
        sim = self.scenario.sim
        saved = [array.copy() for array in (sim.chaser_pos, sim.target_pos,
                                            sim.world.data.qpos, sim.world.data.qvel)]
        self.ui.draw(self.scenario, 1., [])
        before = self.pg.surfarray.array3d(self.ui.screen)
        self.drag()
        self.ui.draw(self.scenario, 1., [])
        after = self.pg.surfarray.array3d(self.ui.screen)
        # Exclude the live FPS/RTF counter from the fixed top-bar comparison.
        np.testing.assert_array_equal(before[:600, :105], after[:600, :105])
        self.assertGreater(np.count_nonzero(before[300:, 150:600] != after[300:, 150:600]), 10000)
        for old, current in zip(saved, (sim.chaser_pos, sim.target_pos, sim.world.data.qpos, sim.world.data.qvel)):
            np.testing.assert_array_equal(old, current)
        self.assertEqual(self.ui.render_backend, 'Software 3D')

    def test_settings_are_staged_and_applied_once(self):
        self.ui.draw(self.scenario, 1., [])
        self.click_action('settings')
        self.ui.draw(self.scenario, 1., [])
        self.click_action(('n_chasers', 1))
        self.click_action(('n_targets', 1))
        self.click_action(('chasers_per_target', 1))
        self.click_action(('arrival_interval', -1))
        self.click_action(('trajectory', 1))
        self.assertEqual(self.cfg.n_chasers, 2)
        self.assertIsNone(self.ui.consume_settings())
        self.click_action('apply')
        self.assertEqual(self.ui.consume_settings(), dict(n_chasers=3, n_targets=3,
                          chasers_per_target=2, arrival_interval=3.5, trajectory='random',
                          v_max=12., cruise_speed=12., target_speed_min=3., target_speed_max=7.,
                          min_separation=5., stand_off_distance=10., r_dock=16., profile_speeds=()))
        self.assertIsNone(self.ui.consume_settings())
        self.assertFalse(self.ui.settings_open)
        self.event(self.pg.KEYDOWN, key=self.pg.K_TAB)
        self.assertTrue(self.ui.settings_open)

    def test_speed_settings_support_independent_mixed_profiles_and_valid_ranges(self):
        self.ui.settings_open = True
        self.ui.draw(self.scenario, 1., [])
        self.click_action(('v_max', 1))
        self.click_action(('target_speed_min', 1))
        self.click_action(('target_speed_max', -1))
        self.click_action(('speed_profile', 1))
        self.click_action(('speed_profile', 1))
        self.assertEqual(self.ui._speed_profile, 'weave')
        self.click_action(('profile_speed_min', 1))
        self.click_action(('profile_speed_max', 1))
        self.click_action(('speed_profile', 1))
        self.click_action(('profile_speed_min', 1))
        self.click_action('apply')
        settings = self.ui.consume_settings()
        self.assertEqual(settings['trajectory'], 'mixed')
        self.assertNotIn('speed_profile', settings)
        self.assertEqual(settings['v_max'], settings['cruise_speed'])
        self.assertEqual(settings['v_max'], 12.5)
        self.assertEqual(settings['profile_speeds'], (('weave', 4., 7.), ('parabolic', 4., 6.5)))
        config = replace(self.cfg, **settings)
        self.assertEqual(config.speed_range('weave'), (4., 7.))
        self.assertEqual(config.speed_range('random'), (3.5, 6.5))
        for _ in range(100):
            self.ui._change_setting('target_speed_max', -1)
            self.ui._change_setting('profile_speed_max', -1)
            self.ui._change_setting('v_max', -1)
        config = replace(self.cfg, **self.ui._settings)
        self.assertGreater(config.target_speed_min, 0.)
        self.assertLessEqual(config.target_speed_min, config.target_speed_max)
        self.assertEqual(config.v_max, config.cruise_speed)
        self.assertLessEqual(*config.speed_range('parabolic'))

    def test_relaunch_capacity_is_locked_and_settings_fit_small_window(self):
        self.ui.cfg = replace(self.cfg, recycle_on_service=True)
        self.ui.settings_open = True
        self.ui.draw(self.scenario, 1., [])
        actions = [action for _, action in self.ui._buttons]
        self.assertNotIn(('chasers_per_target', 1), actions)
        self.assertNotIn(('chasers_per_target', -1), actions)
        self.ui._change_setting('chasers_per_target', 1)
        self.assertEqual(self.ui._settings['chasers_per_target'], 1)
        for rect, action in self.ui._buttons:
            if action != 'settings':
                self.assertTrue(self.ui.settings_rect.contains(rect))
        self.assertLessEqual(self.ui.settings_rect.bottom, self.ui.viewport.bottom)

    def test_distance_controls_apply_valid_values_and_expand_crossed_limits(self):
        self.ui.settings_open = True
        self.ui.draw(self.scenario, 1., [])
        for _ in range(5):
            self.click_action(('min_separation', 1))
        self.assertEqual([self.ui._settings[key] for key in ('min_separation', 'stand_off_distance', 'r_dock')],
                         [10., 15., 21.])
        for _ in range(6):
            self.click_action(('stand_off_distance', 1))
        self.assertEqual(self.ui._settings['stand_off_distance'], 21.)
        self.assertEqual(self.ui._settings['r_dock'], 27.)
        self.click_action(('r_dock', 1))
        self.assertEqual(self.cfg.min_separation, 5.)  # Staged until Apply.
        self.assertIsNone(self.ui.consume_settings())
        self.click_action('apply')
        applied = replace(self.cfg, **self.ui.consume_settings())
        self.assertEqual((applied.min_separation, applied.stand_off_distance, applied.r_dock), (10., 21., 28.))

    def test_distance_decrease_buttons_clamp_without_invalidating_configuration(self):
        self.ui.settings_open = True
        self.ui.draw(self.scenario, 1., [])
        for key in ('r_dock', 'stand_off_distance', 'min_separation'):
            for _ in range(30):
                self.click_action((key, -1))
                cfg = replace(self.cfg, **self.ui._settings)
                self.assertLess(cfg.min_separation, cfg.stand_off_distance)
                self.assertLess(cfg.stand_off_distance, cfg.r_dock)
        self.assertEqual(self.ui._settings['min_separation'], 1.)
        self.assertEqual(self.ui._settings['stand_off_distance'], 6.)
        self.assertEqual(self.ui._settings['r_dock'], 11.)
        for key in ('stand_off_distance', 'r_dock'):
            for _ in range(30):
                self.click_action((key, -1))
        self.click_action('apply')
        cfg = replace(self.cfg, **self.ui.consume_settings())
        self.assertEqual((cfg.min_separation, cfg.stand_off_distance, cfg.r_dock), (1., 2., 3.))

    def test_distance_floor_respects_configured_aircraft_body_size(self):
        self.ui.cfg = replace(self.cfg, body_radius=3., min_separation=7., stand_off_distance=12., r_dock=18.)
        self.ui._settings = self.ui._current_settings()
        self.ui.settings_open = True
        self.ui.draw(self.scenario, 1., [])
        for _ in range(20):
            self.click_action(('min_separation', -1))
        self.click_action('apply')
        cfg = replace(self.ui.cfg, **self.ui.consume_settings())
        self.assertAlmostEqual(cfg.min_separation, 6.1)
        self.assertGreater(cfg.min_separation, 2.*cfg.body_radius)

    def test_completion_bubble_uses_event_position_and_simulation_time(self):
        sim = self.scenario.sim
        self.ui.draw(self.scenario, 1., [])
        event = dict(type='completion', time=sim.time-.2, chaser_id=0, generation=0,
                     target_id=int(sim.target_ids[0]), position=(45., 30., 38.))
        sim.events.append(event)
        def pixels():
            self.ui.screen.fill((0, 0, 0))
            self.ui._event_effects(sim)
            return self.pg.surfarray.array3d(self.ui.screen)
        first = pixels()
        self.assertGreater(np.count_nonzero(first[:, :, 1] > first[:, :, 0]), 100)
        # Recycled/moved slots cannot move the existing immutable event halo.
        sim.target_pos[0] += [30., 20., 10.]
        sim.chaser_pos[0] += [20., 10., 10.]
        np.testing.assert_array_equal(pixels(), first)
        # Pausing means the simulation clock stays fixed, so the effect does too.
        np.testing.assert_array_equal(pixels(), first)
        sim.tick += int(1.2/sim.config.dt)
        self.assertEqual(np.count_nonzero(pixels()), 0)
        sim.events.append(dict(type='launch', time=sim.time-.2, chaser_id=0, generation=1,
                               target_id=None, position=tuple(sim.base_positions[0])))
        self.assertGreater(np.count_nonzero(pixels()[:, :, 1]), 20)
        sim.tick += int(.8/sim.config.dt)
        self.assertEqual(np.count_nonzero(pixels()), 0)

    def test_inactive_pair_disappears_and_generation_starts_a_new_trail(self):
        sim = self.scenario.sim
        self.ui.draw(self.scenario, 1., [])
        self.assertIn(('c', 0, 0), self.ui._trails)
        target_id = int(sim.target_ids[0])
        sim.chaser_active[0] = False
        sim.target_present[0] = False
        with patch.object(self.ui, '_software_world', wraps=self.ui._software_world) as world, \
             patch.object(self.ui, '_text', wraps=self.ui._text) as text:
            self.ui.draw(self.scenario, 1., [])
        self.assertEqual(len(world.call_args.args[0]), 1)  # Only active shadows.
        labels = [str(call.args[0]) for call in text.call_args_list]
        self.assertNotIn('C0', labels)
        self.assertNotIn(f'T{target_id}', labels)
        self.assertLess(self.ui.data.mocap_pos[self.ui.craft_mocap[0], 2], self.cfg.ground_z-10.)
        self.assertLess(self.ui.data.mocap_pos[self.ui.target_mocap[0], 2], self.cfg.ground_z-10.)
        self.assertNotIn(('c', 0, 0), self.ui._trails)
        self.assertNotIn(('t', target_id), self.ui._trails)
        sim.chaser_active[0] = True
        sim.chaser_generation[0] += 1
        sim.previous_chaser_pos[0] = [80., 40., 35.]
        sim.chaser_pos[0] = sim.base_positions[0]
        sim.tick += int(.1/sim.config.dt)
        cp, *_ = self.ui._snapshot(self.scenario, .3)
        np.testing.assert_array_equal(cp[0], sim.base_positions[0])
        self.assertNotIn(('c', 0, 0), self.ui._trails)
        self.assertEqual(len(self.ui._trails[('c', 0, 1)]), 1)
        np.testing.assert_array_equal(self.ui._trails[('c', 0, 1)][0], sim.base_positions[0])

    def test_static_cache_preserves_dynamic_world_and_rebuilds_for_camera(self):
        self.ui.draw(self.scenario, 1., [])
        ground = self.ui._ground_cache
        self.scenario.step()
        self.ui.draw(self.scenario, 1., [])
        self.assertIs(self.ui._ground_cache, ground)
        cached = self.pg.surfarray.array3d(self.ui.screen)
        self.ui._ground_cache_key = None
        self.ui.draw(self.scenario, 1., [])
        rebuilt = self.pg.surfarray.array3d(self.ui.screen)
        np.testing.assert_array_equal(cached[:, 105:-66], rebuilt[:, 105:-66])
        ground = self.ui._ground_cache
        self.drag()
        self.ui.draw(self.scenario, 1., [])
        self.assertIsNot(self.ui._ground_cache, ground)

    def test_fullscreen_toggle_resize_and_visual_toggles(self):
        self.event(self.pg.KEYDOWN, key=self.pg.K_F11)
        self.assertTrue(self.ui.fullscreen)
        self.event(self.pg.KEYDOWN, key=self.pg.K_F11)
        self.assertFalse(self.ui.fullscreen)
        self.event(self.pg.VIDEORESIZE, w=1200, h=800)
        self.assertEqual(self.ui.screen.get_size(), (1200, 800))
        self.assertEqual(self.ui.viewport.size, (1200, 629))
        for key, attr in ((self.pg.K_l, 'show_labels'), (self.pg.K_v, 'show_trails'), (self.pg.K_c, 'show_links')):
            self.event(self.pg.KEYDOWN, key=key)
            self.assertFalse(getattr(self.ui, attr))


if __name__ == '__main__':
    unittest.main()
