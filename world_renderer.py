"""Fullscreen observation-world viewer: MuJoCo shading inside a Pygame HUD.

The scene is a separate, read-only visual replica. Snapshot positions move mocap
bodies; neither camera input nor drawing writes to the simulated physical world.
"""
from collections import deque
import math
import time
import numpy as np


PROFILES = ('mixed', 'random', 'constant-height', 'weave', 'parabolic', 'orbit')
SPEED_PROFILES = PROFILES[1:]
PROFILE_LABELS = {'mixed': 'Mixed traffic', 'random': 'Random waypoints',
                  'constant-height': 'Level cruise', 'weave': 'Weaving',
                  'parabolic': 'Parabolic arc', 'orbit': 'Orbit'}
BG = (10, 18, 24)
PANEL = (16, 29, 36)
TEXT = (222, 235, 235)
MUTED = (124, 153, 162)
TEAL = (76, 222, 194)
AMBER = (255, 192, 104)


def _scene_xml(cfg):
    """Build metric ground landmarks and visual-only quad/beacon bodies."""
    width, depth, ground = cfg.width, cfg.depth, cfg.ground_z
    tracks = max(cfg.n_targets, int(getattr(cfg, 'max_tracks', cfg.n_targets)))
    geoms = []
    for x in np.arange(0., width+.1, 10.):
        geoms.append(f'<geom type="capsule" fromto="{x} 0 {ground+.025} {x} {depth} {ground+.025}" size=".025" rgba=".22 .34 .34 1"/>')
    for y in np.arange(0., depth+.1, 10.):
        geoms.append(f'<geom type="capsule" fromto="0 {y} {ground+.025} {width} {y} {ground+.025}" size=".025" rgba=".22 .34 .34 1"/>')
    edges = [((0, 0), (width, 0)), ((0, depth), (width, depth)), ((0, 0), (0, depth))]
    for a, b in edges:
        geoms.append(f'<geom type="capsule" fromto="{a[0]} {a[1]} {ground+.05} {b[0]} {b[1]} {ground+.05}" size=".09" rgba=".22 .63 .55 1"/>')
    geoms.append(f'<geom type="capsule" fromto="{width} 0 {ground+.07} {width} {depth} {ground+.07}" size=".19" rgba=".95 .58 .18 1"/>')
    for y in np.arange(0., depth+.1, 10.):
        geoms.append(f'<geom type="cylinder" pos="{width} {y} {ground+1.1}" size=".16 1.1" rgba=".18 .25 .28 1"/>')
        geoms.append(f'<geom type="sphere" pos="{width} {y} {ground+2.25}" size=".22" rgba="1 .65 .2 1"/>')
    craft = []
    for i in range(cfg.n_chasers):
        pad_y = depth*(i+1)/(cfg.n_chasers+1)
        geoms.append(f'<geom name="pad_outer_{i}" type="cylinder" pos="9 {pad_y} {ground+.035}" size="2.4 .03" rgba=".10 .21 .23 1"/>')
        geoms.append(f'<geom name="pad_inner_{i}" type="cylinder" pos="9 {pad_y} {ground+.073}" size="1.55 .012" rgba=".16 .35 .34 1"/>')
        parts = ['<geom type="box" size=".62 .39 .16" material="craft_shell"/>',
                 '<geom type="box" pos=".35 0 .19" size=".2 .22 .07" rgba=".22 .85 .71 1"/>',
                 '<geom type="sphere" pos=".69 0 .01" size=".13" rgba=".7 1 .94 1"/>']
        for x in (-.87, .87):
            for y in (-.87, .87):
                parts.extend([f'<geom type="capsule" fromto="0 0 0 {x} {y} 0" size=".075" rgba=".13 .22 .25 1"/>',
                              f'<geom type="cylinder" pos="{x} {y} .045" size=".43 .025" rgba=".15 .26 .29 .85"/>',
                              f'<geom type="cylinder" pos="{x} {y} .09" size=".09 .065" rgba=".36 .74 .68 1"/>'])
        craft.append(f'<body name="craft_{i}" mocap="true" pos="0 0 -50">'+''.join(parts)+'</body>')
    targets = []
    for j in range(tracks):
        targets.append(f'''<body name="target_{j}" mocap="true" pos="0 0 -50">
          <geom type="sphere" size=".72" material="target_shell"/>
          <geom type="box" size="1.35 .15 .055" rgba=".75 .42 .12 1"/>
          <geom type="box" size=".15 1.35 .055" rgba=".75 .42 .12 1"/>
          <geom type="sphere" pos="0 0 .78" size=".14" rgba="1 .87 .52 1"/>
        </body>''')
    return f'''<mujoco model="hawk_observation_view">
      <compiler angle="radian"/>
      <option gravity="0 0 -9.81"/>
      <statistic center="{width*.5} {depth*.5} {ground+cfg.height*.25}" extent="{max(width, depth, cfg.height)}"/>
      <visual><global offwidth="1920" offheight="1200" fovy="48"/>
        <quality shadowsize="2048" offsamples="4"/>
        <headlight ambient=".3 .35 .38" diffuse=".35 .35 .35" specular=".15 .15 .15"/>
        <map znear=".002" zfar="8" fogstart="3" fogend="7"/>
        <rgba haze=".08 .13 .17 1"/></visual>
      <asset>
        <texture name="sky" type="skybox" builtin="gradient" rgb1=".035 .065 .09" rgb2=".19 .25 .27" width="512" height="3072"/>
        <texture name="floor" type="2d" builtin="checker" rgb1=".115 .16 .17" rgb2=".13 .175 .185" width="512" height="512"/>
        <material name="ground" texture="floor" texrepeat="30 30" texuniform="true" reflectance=".04" shininess=".08"/>
        <material name="craft_shell" rgba=".61 .79 .78 1" specular=".5" shininess=".6"/>
        <material name="target_shell" rgba=".97 .57 .17 1" specular=".45" shininess=".45"/>
      </asset>
      <default><geom contype="0" conaffinity="0" group="0"/></default>
      <worldbody>
        <light name="sun" pos="{width*.2} {depth*.1} {ground+150}" dir=".25 .2 -1" directional="true" castshadow="true" diffuse=".85 .86 .8" ambient=".18 .22 .22" specular=".3 .3 .3"/>
        <geom name="ground" type="plane" pos="{width/2} {depth/2} {ground}" size="220 220 .1" material="ground"/>
        {''.join(geoms)}{''.join(craft)}{''.join(targets)}
      </worldbody></mujoco>'''


class Renderer:
    """Adaptive fullscreen world with native 3D camera and fixed HUD/settings."""
    def __init__(self, cfg, fullscreen=True):
        import pygame
        import mujoco
        self.pg, self.mj, self.cfg = pygame, mujoco, cfg
        pygame.init()
        self.fullscreen = bool(fullscreen)
        self._closed = False
        self._render = None
        self._native_unavailable = False
        self.render_backend = 'MuJoCo OpenGL'
        self.native_error = None
        self.model = mujoco.MjModel.from_xml_string(_scene_xml(cfg))
        self.data = mujoco.MjData(self.model)
        self._software_scene = mujoco.MjvScene(self.model, maxgeom=5000)
        self._scene_option = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._drag_mode = self._drag_button = self._drag_position = None
        self.settings_open = False
        self.pending_settings = None
        self._settings = self._current_settings()
        self._speed_profile = SPEED_PROFILES[0]
        self._buttons = []
        self.show_labels, self.show_links, self.show_trails = True, True, True
        self._trails = {}
        self._seen_generations = None
        self._last_trail_time = -math.inf
        self._last_wall = self._last_sim = None
        self.fps = self.rtf = 0.
        self._set_display()
        self.reset_camera()
        self.craft_mocap = [self.model.body(f'craft_{i}').mocapid[0] for i in range(cfg.n_chasers)]
        self._pad_geoms = [(self.model.geom(f'pad_outer_{i}').id, self.model.geom(f'pad_inner_{i}').id)
                           for i in range(cfg.n_chasers)]
        self.target_mocap = [self.model.body(f'target_{j}').mocapid[0]
                             for j in range(max(cfg.n_targets, int(getattr(cfg, 'max_tracks', cfg.n_targets))))]
        self._base_rgba = self.model.geom_rgba.copy()
        self._target_geom_ids = []
        for j in range(len(self.target_mocap)):
            body = self.model.body(f'target_{j}').id
            self._target_geom_ids.append(np.flatnonzero(self.model.geom_bodyid == body))

    def _current_settings(self):
        return dict(trajectory=getattr(self.cfg, 'trajectory', 'mixed'),
                    n_chasers=int(self.cfg.n_chasers), n_targets=int(self.cfg.n_targets),
                    chasers_per_target=int(getattr(self.cfg, 'chasers_per_target', 1)),
                    arrival_interval=float(getattr(self.cfg, 'arrival_interval', 4.)),
                    v_max=float(self.cfg.v_max), cruise_speed=float(self.cfg.v_max),
                    target_speed_min=float(self.cfg.target_speed_min),
                    target_speed_max=float(self.cfg.target_speed_max),
                    min_separation=float(self.cfg.min_separation),
                    stand_off_distance=float(self.cfg.stand_off_distance),
                    r_dock=float(self.cfg.r_dock),
                    profile_speeds=tuple(tuple(entry) for entry in getattr(self.cfg, 'profile_speeds', ())))

    def _settings_geometry(self):
        width, height = self.screen.get_size()
        panel_width = min(976, width-48)
        return self.pg.Rect(width-panel_width-24, 119, panel_width, min(510, height-192))

    def _set_display(self):
        pg = self.pg
        info = pg.display.Info()
        desktop = (max(1024, info.current_w), max(720, info.current_h))
        flags = pg.DOUBLEBUF | (pg.FULLSCREEN if self.fullscreen else pg.RESIZABLE)
        size = desktop if self.fullscreen else (min(1500, desktop[0]), min(930, desktop[1]))
        self.screen = pg.display.set_mode(size, flags)
        pg.display.set_caption('HAWK | Continuous Observation World')
        self._resize_resources()

    def _resize_resources(self):
        width, height = self.screen.get_size()
        self.viewport = self.pg.Rect(0, 105, width, max(100, height-171))
        self.telemetry_rect = self.pg.Rect(23, 127, 216, 378)
        self.settings_rect = self._settings_geometry()
        if self._render is not None:
            self._render.close()
        scale = min(1., 1600/self.viewport.width, 1000/self.viewport.height)
        self.render_size = (max(64, int(self.viewport.width*scale)), max(64, int(self.viewport.height*scale)))
        self._render = None
        if not self._native_unavailable:
            try:
                self._render = self.mj.Renderer(self.model, width=self.render_size[0], height=self.render_size[1], max_geom=5000)
            except Exception as error:
                # A display-less macOS executor cannot create a CGL connection.
                # Preserve the real MuJoCo simulation and render its visual
                # replica in software; do not pretend this is OpenGL output.
                self._native_unavailable = True
                self.native_error = str(error)
                self.render_backend = 'Software 3D'
        self._scene = self._render.scene if self._render is not None else self._software_scene
        self.font = self.pg.font.SysFont('Arial', 16)
        self.small = self.pg.font.SysFont('Arial', 13)
        self.micro = self.pg.font.SysFont('Arial', 11)
        self.title = self.pg.font.SysFont('Arial', 29, bold=True)
        self.number = self.pg.font.SysFont('Arial', 25, bold=True)
        self._buttons = []
        self._ground_cache = None
        self._ground_cache_key = None
        self._static_polygons = []

    def reset_camera(self):
        self.camera.lookat[:] = [self.cfg.width*.44, self.cfg.depth*.5,
                                 self.cfg.ground_z+self.cfg.height*.30]
        self.camera.distance = max(self.cfg.width, self.cfg.depth, self.cfg.height)*1.30
        self.camera.azimuth = 135.
        self.camera.elevation = -31.

    def _release_drag(self):
        self._drag_mode = self._drag_button = self._drag_position = None

    def consume_settings(self):
        result, self.pending_settings = self.pending_settings, None
        return result

    def _change_setting(self, key, direction):
        if key == 'trajectory':
            self._settings[key] = PROFILES[(PROFILES.index(self._settings[key])+direction) % len(PROFILES)]
        elif key == 'speed_profile':
            self._speed_profile = SPEED_PROFILES[(SPEED_PROFILES.index(self._speed_profile)+direction) % len(SPEED_PROFILES)]
        elif key == 'v_max':
            value = round(float(np.clip(self._settings[key]+direction*.5, .5, max(40., self._settings[key]))), 1)
            self._settings['v_max'] = self._settings['cruise_speed'] = value
        elif key in ('target_speed_min', 'target_speed_max', 'profile_speed_min', 'profile_speed_max'):
            is_profile = key.startswith('profile_')
            low, high = self._profile_range() if is_profile else (self._settings['target_speed_min'], self._settings['target_speed_max'])
            if key.endswith('min'):
                low = round(float(np.clip(low+direction*.5, .5, max(40., low))), 1)
                high = max(low, high)
            else:
                high = round(float(np.clip(high+direction*.5, .5, max(40., high))), 1)
                low = min(low, high)
            if is_profile:
                overrides = {name: (minimum, maximum) for name, minimum, maximum in self._settings['profile_speeds']}
                overrides[self._speed_profile] = (low, high)
                self._settings['profile_speeds'] = tuple((name, *overrides[name]) for name in SPEED_PROFILES if name in overrides)
            else:
                self._settings['target_speed_min'], self._settings['target_speed_max'] = low, high
        elif key == 'arrival_interval':
            self._settings[key] = round(float(np.clip(self._settings[key]+direction*.5, .5, 30.)), 1)
        elif key in ('min_separation', 'stand_off_distance', 'r_dock'):
            minimum = self._settings['min_separation']
            stand_off = self._settings['stand_off_distance']
            outer = self._settings['r_dock']
            if key == 'min_separation':
                minimum = max(1., 2.*self.cfg.body_radius+.1, minimum+direction)
                if minimum >= stand_off:
                    stand_off = minimum+5.
                    outer = max(outer, stand_off+6.)
            elif key == 'stand_off_distance':
                stand_off = max(minimum+1., stand_off+direction)
                if stand_off >= outer:
                    outer = stand_off+6.
            else:
                outer = max(stand_off+1., outer+direction)
            self._settings.update(min_separation=minimum, stand_off_distance=stand_off, r_dock=outer)
        else:
            if key == 'chasers_per_target' and getattr(self.cfg, 'recycle_on_service', False):
                self._settings[key] = 1
                return
            maximum = {'n_chasers': max(32, self.cfg.n_chasers), 'n_targets': max(self.cfg.n_targets, int(getattr(self.cfg, 'max_tracks', 24))),
                       'chasers_per_target': self._settings['n_chasers']}[key]
            self._settings[key] = int(np.clip(self._settings[key]+direction, 1, maximum))
            self._settings['chasers_per_target'] = min(self._settings['chasers_per_target'], self._settings['n_chasers'])

    def _profile_range(self):
        overrides = {name: (minimum, maximum) for name, minimum, maximum in self._settings['profile_speeds']}
        return overrides.get(self._speed_profile,
                             (self._settings['target_speed_min'], self._settings['target_speed_max']))

    def handle_event(self, event):
        pg = self.pg
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_F11:
                self.fullscreen = not self.fullscreen
                self._release_drag()
                self._set_display()
                return True
            if event.key == pg.K_HOME:
                self.reset_camera()
                self._release_drag()
                return True
            if event.key == pg.K_TAB:
                self.settings_open = not self.settings_open
                self._release_drag()
                return True
            if event.key in (pg.K_l, pg.K_v, pg.K_c):
                attr = {pg.K_l: 'show_labels', pg.K_v: 'show_trails', pg.K_c: 'show_links'}[event.key]
                setattr(self, attr, not getattr(self, attr))
                return True
        if event.type == pg.VIDEORESIZE and not self.fullscreen:
            self.screen = pg.display.set_mode((max(900, event.w), max(640, event.h)), pg.RESIZABLE | pg.DOUBLEBUF)
            self._resize_resources()
            return True
        if (event.type == getattr(pg, 'WINDOWFOCUSLOST', None) or
                (event.type == pg.ACTIVEEVENT and not event.gain and event.state & (pg.APPINPUTFOCUS | pg.APPACTIVE))):
            self._release_drag()
            return True
        if event.type == pg.MOUSEWHEEL:
            steps = getattr(event, 'precise_y', event.y)
            if getattr(event, 'flipped', False):
                steps = -steps
            extent = max(self.cfg.width, self.cfg.depth, self.cfg.height)
            self.camera.distance = float(np.clip(self.camera.distance*np.exp(np.clip(-steps*.12, -10., 10.)), extent*.15, extent*5.))
            return True
        if event.type == pg.MOUSEBUTTONDOWN and event.button in (1, 3):
            if event.button == 1:
                for rect, action in self._buttons:
                    if rect.collidepoint(event.pos):
                        if action == 'settings':
                            self.settings_open = not self.settings_open
                        elif action == 'apply':
                            self.pending_settings = dict(self._settings)
                            self.settings_open = False
                        elif action == 'close':
                            self.settings_open = False
                        else:
                            self._change_setting(*action)
                        self._release_drag()
                        return True
            if self.settings_open and self.settings_rect.collidepoint(event.pos):
                return True
            if not self.viewport.collidepoint(event.pos) or self.telemetry_rect.collidepoint(event.pos):
                return False
            mod = getattr(event, 'mod', pg.key.get_mods())
            self._drag_mode = 'pan' if event.button == 3 or mod & pg.KMOD_SHIFT else 'orbit'
            self._drag_button = event.button
            self._drag_position = np.asarray(event.pos, dtype=float)
            return True
        if event.type == pg.MOUSEBUTTONUP and event.button == self._drag_button:
            self._release_drag()
            return True
        if event.type == pg.MOUSEMOTION and self._drag_mode:
            if hasattr(event, 'buttons') and not event.buttons[self._drag_button-1]:
                self._release_drag()
                return True
            delta = np.asarray(event.rel, dtype=float) if hasattr(event, 'rel') else np.asarray(event.pos)-self._drag_position
            if hasattr(event, 'pos'):
                self._drag_position = np.asarray(event.pos, dtype=float)
            action = self.mj.mjtMouse.mjMOUSE_MOVE_V if self._drag_mode == 'pan' else self.mj.mjtMouse.mjMOUSE_ROTATE_V
            self.mj.mjv_moveCamera(self.model, action, float(delta[0]/self.viewport.height),
                                  float(delta[1]/self.viewport.height), self.camera)
            self.camera.elevation = float(np.clip(self.camera.elevation, -85., -8.))
            self.camera.azimuth %= 360.
            return True
        return False

    def _text(self, text, pos, color=TEXT, font=None):
        self.screen.blit((font or self.font).render(str(text), True, color), pos)

    def _panel(self, rect, alpha=226):
        surface = self.pg.Surface(rect.size, self.pg.SRCALPHA)
        self.pg.draw.rect(surface, (*PANEL, alpha), surface.get_rect(), border_radius=12)
        self.pg.draw.rect(surface, (68, 99, 106, 150), surface.get_rect(), 1, border_radius=12)
        self.screen.blit(surface, rect.topleft)

    def _button(self, rect, text, action, filled=False):
        self.pg.draw.rect(self.screen, (36, 90, 83) if filled else (28, 44, 52), rect, border_radius=6)
        label = self.small.render(text, True, TEAL if filled else TEXT)
        self.screen.blit(label, label.get_rect(center=rect.center))
        self._buttons.append((rect, action))

    def _connector(self, a, b, color, width=1.):
        if self._render is None:
            self._extra_lines.append((np.asarray(a), np.asarray(b), color, width))
            return
        scene = self._scene
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        self.mj.mjv_initGeom(geom, self.mj.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color))
        self.mj.mjv_connector(geom, self.mj.mjtGeom.mjGEOM_LINE, width, np.asarray(a), np.asarray(b))
        geom.category = self.mj.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1

    def _project(self, point):
        projected = self._project_unclipped(point)
        if projected is None:
            return None
        x, y = projected[0]
        if not self.viewport.collidepoint(x, y):
            return None
        return int(x), int(y)

    def _prepare_projection(self):
        """Compute one pinhole basis per frame, shared by all projected vertices."""
        left, right = self._scene.camera
        eye = (np.asarray(left.pos)+np.asarray(right.pos))*.5
        forward = np.asarray(left.forward, dtype=float)
        forward /= max(np.linalg.norm(forward), 1e-9)
        up = np.asarray(left.up, dtype=float)
        up /= max(np.linalg.norm(up), 1e-9)
        lateral = np.cross(forward, up)
        frustum_height = left.frustum_top-left.frustum_bottom
        focal = left.frustum_near/frustum_height*self.viewport.height
        center_y = self.viewport.bottom+left.frustum_bottom/frustum_height*self.viewport.height
        self._projection = eye, forward, up, lateral, focal, center_y, max(.1, left.frustum_near)

    def _project_unclipped(self, point):
        """Project through MuJoCo's CPU camera state, with camera-space depth."""
        eye, forward, up, lateral, focal, center_y, near = self._projection
        delta = np.asarray(point)-eye
        depth = float(delta @ forward)
        if depth <= near:
            return None
        x = self.viewport.centerx+float(delta @ lateral)/depth*focal
        y = center_y-float(delta @ up)/depth*focal
        return np.array([x, y]), depth, focal/depth

    def _software_world(self, cp, tp, present):
        """Lambert-shaded primitive rasterization when native CGL is unavailable.

        It uses MuJoCo's transformed geom poses and camera projection. Shadows
        are directional ground projections, not a substitute physics model.
        """
        pg = self.pg
        old_clip = self.screen.get_clip()
        self.screen.set_clip(self.viewport)
        pg.draw.rect(self.screen, (27, 43, 53), self.viewport)
        ground = self.cfg.ground_z
        camera_key = (tuple(self.viewport), tuple(self.camera.lookat), self.camera.distance,
                      self.camera.azimuth, self.camera.elevation)
        rebuild_static = camera_key != self._ground_cache_key
        polygons = []
        light = np.array([-.25, -.2, 1.])
        light /= np.linalg.norm(light)
        def polygon(vertices, color, normal=None):
            projected = [self._project_unclipped(vertex) for vertex in vertices]
            if any(point is None for point in projected):
                return
            coordinates = [tuple(np.clip(point[0], -30000, 30000).astype(int)) for point in projected]
            if not pg.Rect(min(p[0] for p in coordinates), min(p[1] for p in coordinates),
                           max(p[0] for p in coordinates)-min(p[0] for p in coordinates)+1,
                           max(p[1] for p in coordinates)-min(p[1] for p in coordinates)+1).colliderect(self.viewport):
                return
            shade = 1.
            if normal is not None:
                unit = normal/max(np.linalg.norm(normal), 1e-9)
                shade = .42+.58*max(0., float(unit @ light))
            rgb = tuple(int(np.clip(component*shade, 0, 255)) for component in color[:3])
            polygons.append((sum(p[1] for p in projected)/len(projected), 'polygon', coordinates, rgb))
        # The floor raster and projected stationary landmarks change only with
        # the camera. Keep landmark depths so moving craft still sort correctly.
        if rebuild_static:
            for ix, x in enumerate(np.arange(-120., self.cfg.width+120., 20.)):
                for iy, y in enumerate(np.arange(-120., self.cfg.depth+120., 20.)):
                    color = (33, 48, 50) if (ix+iy) % 2 else (37, 53, 54)
                    polygon([[x,y,ground], [x+20,y,ground], [x+20,y+20,ground], [x,y+20,ground]], color)
            for _, _, points, color in sorted(polygons, reverse=True, key=lambda item:item[0]):
                pg.draw.polygon(self.screen, color, points)
            self._ground_cache = self.screen.copy()
        else:
            self.screen.blit(self._ground_cache, (0, 0))
        dynamic_polygons = []
        static_polygons = [] if rebuild_static else self._static_polygons
        # Directional shadows live on the same ground plane as pads and grid.
        shade_surface = pg.Surface(self.screen.get_size(), pg.SRCALPHA)
        for position, radius in [(p, 1.7) for p in cp]+[(tp[j], 1.1) for j in np.flatnonzero(present)]:
            altitude = max(0., position[2]-ground)
            center = np.array([position[0]+.25*altitude, position[1]+.2*altitude, ground+.01])
            points = [self._project_unclipped(center+radius*np.array([math.cos(t),math.sin(t),0.]))
                      for t in np.linspace(0., 2.*math.pi, 13)[:-1]]
            if all(p is not None for p in points):
                pg.draw.polygon(shade_surface, (1, 8, 12, 115), [p[0] for p in points])
        self.screen.blit(shade_surface, (0, 0))
        # Low-poly solids retain lighting, perspective, and occlusion without GL.
        angles = np.arange(8)*np.pi/4.
        for index in range(self.model.ngeom):
            is_static = self.model.geom_bodyid[index] == 0
            if is_static and not rebuild_static:
                continue
            polygons = static_polygons if is_static else dynamic_polygons
            kind = int(self.model.geom_type[index])
            if kind == int(self.mj.mjtGeom.mjGEOM_PLANE):
                continue
            position = self.data.geom_xpos[index]
            if position[2] < ground-2.:
                continue
            rotation = self.data.geom_xmat[index].reshape(3,3)
            size = self.model.geom_size[index]
            rgba = self.model.geom_rgba[index]
            color = np.asarray(rgba[:3])*255.
            if kind == int(self.mj.mjtGeom.mjGEOM_CAPSULE):
                a, b = position-rotation[:,2]*size[1], position+rotation[:,2]*size[1]
                pa, pb = self._project_unclipped(a), self._project_unclipped(b)
                if pa is not None and pb is not None:
                    polygons.append(((pa[1]+pb[1])*.5, 'line', (pa[0],pb[0],max(1,int(2*size[0]*pa[2]))), tuple(color.astype(int))))
                continue
            if kind == int(self.mj.mjtGeom.mjGEOM_BOX):
                vertices = np.array([[x,y,z] for x in (-size[0],size[0]) for y in (-size[1],size[1]) for z in (-size[2],size[2])])
                faces = [(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
            elif kind == int(self.mj.mjtGeom.mjGEOM_CYLINDER):
                vertices = np.array([[size[0]*np.cos(a),size[0]*np.sin(a),z] for z in (-size[1],size[1]) for a in angles])
                faces = [tuple(range(7,-1,-1)),tuple(range(8,16))]+[(k,(k+1)%8,(k+1)%8+8,k+8) for k in range(8)]
            else:
                r=size[0]
                vertices = r*np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
                faces = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
            transformed = vertices @ rotation.T+position
            for face in faces:
                world = transformed[list(face)]
                normal = np.cross(world[1]-world[0],world[2]-world[0])
                polygon(world,color,normal)
        self._static_polygons = static_polygons
        self._ground_cache_key = camera_key
        polygons = static_polygons+dynamic_polygons
        for a,b,color,width in self._extra_lines:
            pa,pb=self._project_unclipped(a),self._project_unclipped(b)
            if pa is not None and pb is not None:
                polygons.append(((pa[1]+pb[1])*.5,'line',(pa[0],pb[0],max(1,round(width))),tuple(int(v*255) for v in color[:3])))
        for _,kind,points,color in sorted(polygons,reverse=True,key=lambda item:item[0]):
            if kind=='polygon':
                pg.draw.polygon(self.screen,color,points)
            else:
                pg.draw.line(self.screen,color,points[0],points[1],points[2])
        self.screen.set_clip(old_clip)

    def _snapshot(self, scenario, alpha):
        sim = scenario.sim
        cp = (1-alpha)*sim.previous_chaser_pos+alpha*sim.chaser_pos
        tp = (1-alpha)*sim.previous_target_pos+alpha*sim.target_pos
        active = np.asarray(getattr(sim, 'chaser_active', np.ones(len(cp), dtype=bool)), dtype=bool)
        generations = np.asarray(getattr(sim, 'chaser_generation', np.zeros(len(cp), dtype=int)), dtype=int)
        if self._seen_generations is not None:
            changed = generations != self._seen_generations
            cp[changed] = sim.chaser_pos[changed]
        self._seen_generations = generations.copy()
        self._active_craft = active
        for i, base in enumerate(getattr(sim, 'base_positions', ())):
            for geom in self._pad_geoms[i]:
                if not np.array_equal(self.model.geom_pos[geom, :2], base[:2]):
                    self.model.geom_pos[geom, :2] = base[:2]
                    self._ground_cache_key = None
        present = np.asarray(getattr(sim, 'target_present', np.ones(len(tp), dtype=bool)), dtype=bool)
        ids = np.asarray(getattr(sim, 'target_ids', np.arange(len(tp))))
        for i, position in enumerate(cp[:len(self.craft_mocap)]):
            self.data.mocap_pos[self.craft_mocap[i]] = position if active[i] else [0., 0., self.cfg.ground_z-50.]
            velocity = sim.chaser_vel[i]
            yaw = math.atan2(velocity[1], velocity[0]) if np.linalg.norm(velocity[:2]) > .1 else 0.
            self.data.mocap_quat[self.craft_mocap[i]] = [math.cos(yaw/2.), 0., 0., math.sin(yaw/2.)]
        self.model.geom_rgba[:] = self._base_rgba
        for j, mocap in enumerate(self.target_mocap):
            visible = j < len(tp) and present[j]
            self.data.mocap_pos[mocap] = tp[j] if visible else [0., 0., self.cfg.ground_z-50.]
            if visible and sim.status[j] == 1:
                self.model.geom_rgba[self._target_geom_ids[j], :3] = [.27, .58, .45]
        self.mj.mj_forward(self.model, self.data)
        if sim.time < self._last_trail_time:
            self._trails.clear()
            self._last_trail_time = -math.inf
        craft_indices = np.flatnonzero(active)[:12]
        target_indices = np.flatnonzero(present)[:12]
        live_keys = {('c', int(i), int(generations[i])) for i in craft_indices}
        live_keys.update(('t', int(ids[j])) for j in target_indices)
        # Remove completed bodies and old generations immediately, including
        # frames between trail samples and the frame a replacement launches.
        self._trails = {key: value for key, value in self._trails.items() if key in live_keys}
        if sim.time-self._last_trail_time >= .08:
            for i in craft_indices:
                self._trails.setdefault(('c', int(i), int(generations[i])), deque(maxlen=36)).append(cp[i].copy())
            for j in target_indices:
                key = ('t', int(ids[j]))
                self._trails.setdefault(key, deque(maxlen=36)).append(tp[j].copy())
            self._last_trail_time = sim.time
        return cp, tp, present, ids

    def _event_effects(self, sim):
        """Draw fading observation bubbles from immutable events and sim time.

        These purely visual halos survive object-slot reuse and freeze when the
        simulation is paused. Their radius has no physical or service meaning.
        """
        pg = self.pg
        overlay = pg.Surface(self.screen.get_size(), pg.SRCALPHA)
        overlay.set_clip(self.viewport)
        angles = np.linspace(0., 2.*math.pi, 49)
        for event in getattr(sim, 'events', ()):
            kind = event.get('type')
            duration = 1.1 if kind == 'completion' else .8
            age = float(sim.time)-float(event['time'])
            if kind not in ('completion', 'launch') or not 0. <= age < duration:
                continue
            center = np.asarray(event['position'], dtype=float)
            projected = self._project_unclipped(center)
            if projected is None:
                continue
            progress, fade = age/duration, 1.-age/duration
            radius = (1.4+5.2*progress) if kind == 'completion' else (1.+2.6*progress)
            if kind == 'completion':
                pixels = max(2, round(radius*projected[2]))
                point = tuple(projected[0].astype(int))
                pg.draw.circle(overlay, (73, 244, 148, round(30*fade)), point, pixels)
                pg.draw.circle(overlay, (108, 255, 182, round(185*fade)), point, pixels, 2)
                planes = ((0, 1), (0, 2), (1, 2))
            else:
                # Launch pulses stay close to the documented base pad plane.
                center = center.copy()
                center[2] = self.cfg.ground_z+.12
                planes = ((0, 1),)
            for a, b in planes:
                points = np.tile(center, (len(angles), 1))
                points[:, a] += radius*np.cos(angles)
                points[:, b] += radius*np.sin(angles)
                projected_ring = [self._project_unclipped(point) for point in points]
                if all(point is not None for point in projected_ring):
                    pg.draw.lines(overlay, (81, 237, 158, round(165*fade)), False,
                                  [point[0] for point in projected_ring], 2)
        self.screen.blit(overlay, (0, 0))

    def _world(self, scenario, alpha):
        cp, tp, present, ids = self._snapshot(scenario, alpha)
        self._extra_lines = []
        if self._render is not None:
            self._render.update_scene(self.data, camera=self.camera)
            self._scene.flags[self.mj.mjtRndFlag.mjRND_SHADOW] = True
        else:
            self.mj.mjv_updateScene(self.model,self.data,self._scene_option,None,self.camera,
                                    self.mj.mjtCatBit.mjCAT_ALL,self._scene)
        self._prepare_projection()
        if self.show_trails:
            for key, trail in self._trails.items():
                points = list(trail)
                color = (.24, .65, .58, .65) if key[0] == 'c' else (.78, .49, .18, .45)
                for a, b in zip(points, points[1:]):
                    self._connector(a, b, color, .8)
        if self.show_links:
            slots = {int(value): j for j, value in enumerate(ids) if present[j]}
            for i, controller in enumerate(scenario.controllers[:16]):
                slot = slots.get(controller.target_id)
                if self._active_craft[i] and slot is not None and scenario.sim.status[slot] == 0:
                    self._connector(cp[i], tp[slot], (.32, .72, .64, .45), 1.)
        if self._render is None:
            self._software_world(cp[self._active_craft],tp,present)
        else:
            pixels = self._render.render()
            surface = self.pg.image.frombuffer(pixels.tobytes(), self.render_size, 'RGB')
            if surface.get_size() != self.viewport.size:
                surface = self.pg.transform.smoothscale(surface, self.viewport.size)
            self.screen.blit(surface, self.viewport.topleft)
        self._event_effects(scenario.sim)
        # Sparse, fixed-size screen labels use the actual perspective camera.
        if self.show_labels:
            for i in np.flatnonzero(self._active_craft)[:8]:
                position = cp[i]
                point = self._project(position+np.array([0., 0., 1.5]))
                if point:
                    self._text(f'C{i}', (point[0]+6, point[1]-9), TEAL, self.micro)
            shown = 0
            for j in np.flatnonzero(present):
                if scenario.sim.status[j] != 0:
                    continue
                point = self._project(tp[j]+np.array([0., 0., 1.3]))
                if point:
                    self._text(f'T{int(ids[j])}', (point[0]+5, point[1]-7), AMBER, self.micro)
                    shown += 1
                if shown >= 8:
                    break
        for label, point in [('LAUNCH / BASE', [8., self.cfg.depth*.5, self.cfg.ground_z]),
                             ('EXIT PERIMETER', [self.cfg.width, self.cfg.depth*.5, self.cfg.ground_z]),
                             ('10 m GRID', [self.cfg.width*.5, 0., self.cfg.ground_z])]:
            screen_point = self._project(point)
            if screen_point:
                self._text(label, (screen_point[0]-35, screen_point[1]+10), (147, 171, 168), self.micro)

    def _hud(self, scenario, paused):
        pg, width, height = self.pg, *self.screen.get_size()
        stats, cfg = scenario.sim.stats(), self.cfg
        pg.draw.rect(self.screen, BG, (0, 0, width, 105))
        pg.draw.line(self.screen, (45, 66, 72), (24, 104), (width-24, 104))
        self._text('HAWK', (27, 17), TEXT, self.title)
        self._text('OBSERVATION WORLD', (29, 54), MUTED, self.micro)
        mode = str(scenario.mode).upper()
        self._text(mode, (200, 23), TEAL, self.number)
        self._text('OBSERVE / RELAUNCH / NONCONTACT' if getattr(cfg, 'recycle_on_service', False)
                   else 'REUSABLE FLEET / NONCONTACT', (200, 57), MUTED, self.micro)
        self._text('PAUSED' if paused else 'LIVE', (width-322, 25), AMBER if paused else TEAL, self.font)
        self._text(f'{scenario.sim.time:07.2f} s', (width-246, 21), TEXT, self.number)
        self._button(pg.Rect(width-112, 22, 87, 34), 'Settings', 'settings')
        profile = PROFILE_LABELS.get(getattr(cfg, 'trajectory', 'mixed'), getattr(cfg, 'trajectory', 'mixed'))
        self._text(f'{profile}   /   K {getattr(cfg, "chasers_per_target", 1)} craft per object   /   '
                   f'Arrival every {getattr(cfg, "arrival_interval", 4.):g}s', (29, 80), MUTED, self.small)
        self._text(f'{self.fps:4.0f} FPS   {self.rtf:4.2f}x real time', (width-217, 79), MUTED, self.small)
        self.telemetry_rect = pg.Rect(23, 127, 216, 378)
        self._panel(self.telemetry_rect)
        x = 41
        self._text('MISSION TELEMETRY', (x, 145), MUTED, self.micro)
        self._text(str(stats.get('active_chasers', cfg.n_chasers)), (x, 169), TEAL, self.number)
        self._text('active aircraft', (x+45, 177), TEXT, self.small)
        pending = stats.get('pending', 0)
        observed = stats.get('serviced', stats.get('observed', 0))
        escaped = stats.get('escaped', 0)
        rows = [('Arrivals', stats.get('arrivals', pending+observed+escaped)),
                ('Pending', pending), ('Observed', observed), ('Escaped', escaped)]
        for index, (label, value) in enumerate(rows):
            y = 215+index*27
            self._text(label, (x, y), MUTED, self.small)
            self._text(str(value), (x+148, y-2), AMBER if label == 'Pending' else TEXT, self.font)
        pg.draw.line(self.screen, (48, 71, 77), (x, 329), (218, 329))
        minimum = stats.get('minimum_separation')
        minimum_text = f'{minimum:.1f} m' if minimum is not None else '--'
        self._text('Closest separation', (x, 345), MUTED, self.small)
        self._text(minimum_text, (x, 368), TEAL if not stats.get('near_contact_violations', 0) else AMBER, self.font)
        self._text(f'Cap violations {getattr(scenario, "capacity_violations", 0)}', (x+90, 371), MUTED, self.micro)
        self._text(f'Reserved objects   {stats.get("reserved", 0)}', (x, 402), MUTED, self.small)
        self._text(f'Allocation loss    {getattr(scenario, "local_loss", 0.):.3f}', (x, 425), MUTED, self.small)
        self._text(f'Configured minimum   {cfg.min_separation:g} m', (x, 451), MUTED, self.small)
        self._text(f'Track {cfg.stand_off_distance:g} m / observe <{cfg.r_dock:g} m', (x, 474), MUTED, self.small)
        # Physics quantities are actual configuration values, in SI units.
        pg.draw.rect(self.screen, BG, (0, height-66, width, 66))
        pg.draw.line(self.screen, (45, 66, 72), (24, height-66), (width-24, height-66))
        self._text('Drag orbit / Right-drag pan / Scroll zoom / Home reset / Tab settings / F11 fullscreen / Space pause / R reset / Esc exit',
                   (26, height-52), TEXT, self.small)
        wind = np.asarray(getattr(cfg, 'wind', (0., 0., 0.)))
        physics = (f'MUJOCO   g {getattr(cfg, "gravity", 9.81):g} m/s²   mass {getattr(cfg, "mass", 1.5):g} kg   '
                   f'air {getattr(cfg, "air_density", 1.225):g} kg/m³   wind {np.linalg.norm(wind):.1f} m/s   '
                   f'physics {1/getattr(cfg, "physics_dt", .002):.0f} Hz   control {1/cfg.dt:.0f} Hz   {self.render_backend}')
        self._text(physics, (26, height-28), MUTED, self.micro)

    def _settings_panel(self):
        pg = self.pg
        self.settings_rect = self._settings_geometry()
        if not self.settings_open:
            return
        self._panel(self.settings_rect, 255)
        x, y = self.settings_rect.x+19, self.settings_rect.y+19
        self._text('WORLD SETTINGS', (x, y), TEXT, self.font)
        self._text('Speeds in m/s. Distances in metres. Apply restarts the observation world.', (x, y+27), MUTED, self.small)
        column_width = (self.settings_rect.width-38)//3
        for index, heading in enumerate(('WORLD', 'SPEEDS', 'DISTANCES')):
            self._text(heading, (x+index*column_width, y+60), TEAL, self.micro)
        world_rows = [('trajectory', 'Trajectory'), ('n_chasers', 'Fleet size'),
                ('n_targets', 'Initial objects'), ('chasers_per_target', 'Craft per object K'),
                ('arrival_interval', 'Arrival interval')]
        speed_rows = [('v_max', 'Observer speed'), ('target_speed_min', 'Default target minimum'),
                      ('target_speed_max', 'Default target maximum'), ('speed_profile', 'Edit profile (also in Mixed)'),
                      ('profile_speed_min', 'Profile minimum'), ('profile_speed_max', 'Profile maximum')]
        distance_rows = [('min_separation', 'Minimum separation'), ('stand_off_distance', 'Tracking stand-off'),
                         ('r_dock', 'Observation radius')]
        row_step = min(52, (self.settings_rect.height-160)//6)
        for column, rows in enumerate((world_rows, speed_rows, distance_rows)):
            col_x = x+column*column_width
            for index, (key, label) in enumerate(rows):
                line_y = y+81+index*row_step
                self._text(label, (col_x, line_y), MUTED, self.small)
                if key == 'speed_profile':
                    value = PROFILE_LABELS[self._speed_profile]
                elif key.startswith('profile_speed_'):
                    value = f'{self._profile_range()[0 if key.endswith("min") else 1]:g} m/s'
                else:
                    value = self._settings[key]
                    value = PROFILE_LABELS.get(value, value) if key == 'trajectory' else value
                    if key == 'arrival_interval':
                        value = f'{value:g} s'
                    elif key in ('v_max', 'target_speed_min', 'target_speed_max'):
                        value = f'{value:g} m/s'
                    elif key in ('min_separation', 'stand_off_distance', 'r_dock'):
                        value = f'{value:g} m'
                locked = key == 'chasers_per_target' and getattr(self.cfg, 'recycle_on_service', False)
                self._text('1 / relaunch cycle' if locked else str(value), (col_x, line_y+18), TEXT, self.small)
                if not locked:
                    self._button(pg.Rect(col_x+column_width-88, line_y+5, 29, 29), '−', (key, -1))
                    self._button(pg.Rect(col_x+column_width-51, line_y+5, 29, 29), '+', (key, 1))
        distance_x = x+2*column_width
        hint_y = y+91+3*row_step
        self._text('Minimum < stand-off < radius', (distance_x, hint_y), TEAL, self.small)
        self._text('Increasing a limit can expand the next.', (distance_x, hint_y+24), MUTED, self.micro)
        self._text('All three values are shown before Apply.', (distance_x, hint_y+43), MUTED, self.micro)
        footer = self.settings_rect.bottom-54
        apply_width = min(190, column_width-108)
        self._button(pg.Rect(x, footer, apply_width, 35), 'Apply and restart', 'apply', True)
        self._button(pg.Rect(x+apply_width+11, footer, 79, 35), 'Close', 'close')
        self._text('Unedited profiles inherit the default range.', (x+column_width, footer+9), MUTED, self.micro)

    def _comparison(self, results):
        pg, width, height = self.pg, *self.screen.get_size()
        rect = pg.Rect(max(20, (width-820)//2), max(116, (height-340)//2), min(820, width-40), 340)
        self._panel(rect, 247)
        x, y = rect.x+27, rect.y+23
        self._text('OBSERVATION BENCHMARK', (x, y), TEXT, self.number)
        self._text('Same physical world and disturbances / independent policy runs', (x, y+36), MUTED, self.small)
        for label, offset in (('POLICY', 0), ('OBSERVED', 187), ('ESCAPED', 311),
                              ('MIN RANGE', 419), ('CAP VIOLATIONS', 617)):
            self._text(label, (x+offset, y+79), MUTED, self.small)
        for index, result in enumerate(results[:3]):
            yy = y+110+index*39
            minimum = result.get('minimum_separation')
            distance = f'{minimum:.2f} m' if minimum is not None else '--'
            self._text(str(result['mode']).upper(), (x, yy), TEAL, self.font)
            self._text(str(result.get('serviced', 0)), (x+187, yy), TEXT, self.font)
            self._text(str(result.get('escaped', 0)), (x+311, yy), TEXT, self.font)
            self._text(distance, (x+419, yy), TEXT, self.font)
            self._text(str(result.get('max_capacity_violations', 0)), (x+617, yy), AMBER, self.font)
        calls = sum(r.get('gnn_forward_calls', 0) for r in results)
        forward_ms = 1000*sum(r.get('gnn_forward_wall_seconds', 0.) for r in results)
        self._text(f'GNN {calls} forwards / {forward_ms:.1f} ms inference. Explicit reservations enforce configured capacity K.',
                   (x, y+240), MUTED, self.micro)
        self._text('R replay    /    ESC close', (x, y+278), TEAL, self.small)

    def draw(self, scenario, alpha, results, paused=False, expected_results=1):
        wall = time.perf_counter()
        if self._last_wall is not None:
            elapsed = max(wall-self._last_wall, 1e-6)
            instant = 1./elapsed
            rtf = max(0., scenario.sim.time-self._last_sim)/elapsed
            self.fps = instant if not self.fps else .9*self.fps+.1*instant
            self.rtf = .85*self.rtf+.15*rtf
        self._last_wall, self._last_sim = wall, scenario.sim.time
        self.screen.fill(BG)
        self._buttons = []
        self._world(scenario, float(np.clip(alpha, 0., 1.)))
        self._hud(scenario, paused)
        self._settings_panel()
        if results and len(results) == expected_results:
            self._comparison(results)
        self.pg.display.flip()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._render is not None:
            self._render.close()
        self.pg.quit()
