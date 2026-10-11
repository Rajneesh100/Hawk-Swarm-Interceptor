# HAWK — Physical observation world: specification and status

Project: HAWK (`hawk-swarm-simulator`). Updated 2026-10-12.

The active application is a continuous, fullscreen, **non-contact observation**
world. Safe observation completion removes the object and its observer, emits
a green bubble, and schedules a replacement at that observer's base pad.
Contact never awards service. New aerial tracks arrive throughout a session.
The default relaunch cycle requires one observer per object. A separate benchmark option compares
Greedy, Auction, and a trained PyTorch GNN on identical seeded conditions.

## Scope and files

All source, tests, project documentation, weights, caches and outputs stay inside
HAWK. `speedy` supplied read-only inspiration: centralized physical constants,
MuJoCo integration, a lit metric world and compact telemetry. No `speedy` code or
files were changed. Unrelated documents, invoices, PDFs, `tmp/pdfs/` and
`output/pdf/` must not be created. `.gitignore` is a tracking safeguard, not a
filesystem permission mechanism.

| Component | Responsibility |
|---|---|
| `physics_constants.py` | SI reference constants and trajectory names |
| `physics_world.py` | MuJoCo model/data, force application, actuator lag, drag, gravity and contacts |
| `sim_engine.py` | 50 Hz mission clock, prescribed track motion, bounded arrival pool, sensor payloads, observation dwell and statistics |
| `estimator.py` | Relative kinematics, line-of-sight rate and observation reachability |
| `swarm_controller.py` | Local policy preferences and acceleration-based stand-off/separation control |
| `reservations.py` | Explicit shared mission authority enforcing configurable per-object capacity |
| `gnn_model.py` | Two learned graph layers, sensor encoding, leakage loss, checkpoint validation |
| `train_gnn.py` | Seeded randomized sensor snapshots, Adam training and held-out reports |
| `world_renderer.py` | Read-only 3D scene, perspective camera, fullscreen Pygame HUD and settings |
| `main.py` | SITL wiring, peer transport, reservation application, live/benchmark CLI and pacing |
| `test_testbed.py` | Integrated sensing, estimation, observation, protocol, learning and scenario regressions |
| `test_physical_world.py` | Newtonian force response, ground contact, profiles and arrival lifecycle |
| `test_observer_cycle.py` | Completion removal, inactive physics/sensors, base relaunch and profile speed overrides |
| `test_reservations.py` | Capacity, stable ownership, global IDs and cached neural scores |
| `test_world_renderer.py` | Camera, fullscreen/settings, fixed HUD and read-only rendering |

## Interfaces and authority

| Interface | Contract |
|---|---|
| `Simulation.observe(agent_id)` | Copied noisy local `Observation`; global IDs plus aligned availability mask |
| `Simulation.step(commands)` | Finite `(N,3)` net acceleration requests, m/s²; advances one control tick |
| `Simulation.set_observation_assignments(mapping)` | Mission metadata controlling dwell ownership; separate from actuation |
| `Simulation.target_index(id)` | Resolve a currently present global object ID to a pool slot, or `None` |
| `Simulation.chaser_generation` | Monotonic incarnation number per fixed fleet bay |
| `Simulation.lifecycle_events` / `events` | Bounded 128-event buffer with copied completion/launch IDs, positions and simulation times |
| `PhysicsWorld.step(...)` | Ten 0.002 s MuJoCo substeps; returns integrated position/velocity arrays |
| `PhysicsWorld.reset_craft(i, position, active=...)` | Reset motor, linear/angular, orientation and contact state for one bay |
| `SwarmController.reset_for_launch(epoch=None)` | Clear prior mission/sensor state, retain weights/counters, optionally join a newer trusted epoch |
| `SwarmController.forget_peer_incarnation(id)` | Clear the replaced peer's claims and withdrawal tombstone |
| `ReservationManager.forget_agent(id)` | Release a previous incarnation's ownership before reassignment |
| `SwarmController.update_telemetry(obs)` | Refresh estimates; perform one due Greedy/GNN policy evaluation |
| `SwarmController.reservation_candidates()` | Local eligible `(target_id, score, distance)` candidates |
| `ReservationManager(K).assign(controllers)` | Retain valid incumbents, allocate free slots, return/set accepted commitments |
| `SwarmController.get_control_commands()` | Bounded own `(3,)` net acceleration vector |
| `build_sensor_graph(obs,cfg)` | Features `(N_visible,M_visible,10)`, adjacency `(N_visible,N_visible)`, estimates |
| `SwarmGNN.forward(features,adjacency,mask)` | Learned target logits, preserving track-column alignment |
| `Renderer.handle_event(event)` | Camera/display/settings input, without modifying simulation arrays |
| `Renderer.consume_settings()` | Optional configuration changes; runner applies them by explicit restart |
| `run_headless(...)` | Finite-horizon simulation; rejects indefinite duration |

Controllers never receive a plant reference. The GNN graph uses sensed noisy
peer/target geometry, not hidden true poses. Peer transport uses true craft
positions only to determine radio connectivity. The reservation manager sees
local candidate scores and IDs, not true positions, but is still a **shared
central mission service**. That service is what guarantees the configured cap
across disconnected radio components; the GNN and mesh alone do not.

Accepted commitments are broadcast after reservation enforcement. Local loss
uses current accepted choices visible through the mesh. All modes publish these
status packets once per control tick; Auction adds negotiation packets. Message
counts therefore include both telemetry and negotiation, not measured radio time.

## Physical quantities and integration

Coordinates are `[x,y,z]`, in metres; z points upward. The arena is 100 × 60 m,
with an 80 m display height, solid ground at z=0 and exit perimeter x=100 m.

| Quantity | Default |
|---|---:|
| Control period / physics period | 0.02 s / 0.002 s |
| Gravity magnitude `g` | 9.81 m/s² |
| Craft mass `m` | 1.5 kg |
| Air density `rho` | 1.225 kg/m³ |
| Drag product `Cd*A` (`drag_area`) | 0.06 m² |
| Thrust magnitude limit | 45 N per craft |
| Thrust response time | 0.08 s |
| Wind `[wx,wy,wz]` | `[0,0,0]` m/s |
| Physical collision sphere radius | 0.35 m |
| Guidance velocity-reference / acceleration-request limits | 12 m/s / 18 m/s² |
| Track speed / altitude envelope | 3–7 m/s / 20–60 m |
| Track acceleration limit | 3 m/s² |
| Minimum separation / desired stand-off / outer observation radius | 5 m / 10 m / 16 m |
| Position / velocity sensor noise | 0.08 m / 0.025 m/s standard deviation |
| Sensor / broadcast range | 150 m / 150 m |
| Policy allocation interval | 0.5 s, with inventory/topology-triggered updates |

Let `clip_norm` limit vector magnitude. The ideal stabilized flight controller
maps the bounded acceleration request to world-axis thrust:

```math
F_{desired}=\operatorname{clip}_{F_{max}}\big[m(\operatorname{clip}_{a_{max}}(a_{cmd})-g_{vec})\big],
\qquad g_{vec}=[0,0,-g].
```

With substep h and motor time constant tau:

```math
F_{k+1}=F_k+(1-e^{-h/\tau})(F_{desired}-F_k),
\qquad
F_{drag}=-\tfrac12\rho C_d A\|v-w\|(v-w).
```

MuJoCo integrates

```math
m\dot v=F_{thrust}+F_{drag}+m g_{vec}+F_{contact},\qquad \dot p=v.
```

Hover thrust initializes at `m*g`, limited by available thrust. Zero motor time
constant means immediate force response. The model includes a solid ground and
craft/track contact geometries. Prescribed tracks are kinematic mocap bodies;
contacts affect the free craft, not their commanded track paths. Angular damping
represents ideal stabilization. World-axis vector thrust abstracts attitude and
rotor mixing; this is not a calibrated full quadrotor aerodynamic model.

`v_max` bounds the **guidance reference**, not physical state. Gravity, drag,
wind and contacts can produce acceleration/speed beyond nominal controller
limits. No position/velocity clamp hides those effects. Deprecated `tau` and
`drag_coeff` remain compatibility fields and no longer drive craft physics.
`physics_contacts` counts substep contact observations, not distinct impacts.
It equals `ground_contact_observations + body_contact_observations`; classification
uses each MuJoCo contact's geometry IDs, distinguishing ground-pad support from
craft/craft or craft/track contacts.

## Trajectories and continuous lifecycle

`--trajectory` or the settings panel selects:

| Profile | Desired path |
|---|---|
| `random` | Cosine-blended y/z waypoints, sinusoidal lateral acceleration and seeded Gaussian heading perturbations |
| `constant-height` | Straight +x flight at exactly the spawn altitude |
| `weave` | Constant-altitude forward flight with sinusoidal lateral tangent |
| `parabolic` | Controlled altitude arc over the route, remaining above ground |
| `orbit` | Drifting horizontal orbit, then a smooth transition to outbound flight |
| `mixed` | Deterministic cycling of the five profiles across new object IDs |

For a parabolic reference, normalized progress u along route length L gives
`z_ref=z_spawn+4*A*u*(1-u)` and desired tangent
`[1,0,4*A*(1-2*u)/L]`. Weave uses tangent
`[1,0.65*sin(0.55*age+phase),0]`. Orbit uses a drifting circular tangent at
0.4 rad/s and transitions outward after one revolution. All are prescribed
controlled reference paths, not falling ballistic objects.

Each track samples a speed in its profile's configured band. `--target-speed-min`
and `--target-speed-max` supply the fallback (3–7 m/s). Repeated
`--profile-speed PROFILE MIN MAX` overrides individual profiles, also inside
`mixed`. `Config.profile_speeds` stores unique `(name, min, max)` triples;
`Config.speed_range(name)` resolves the band. `--chaser-speed` sets both observer
`v_max` and `cruise_speed`; approach correction respects the smaller of these
when configured separately through Python. Bounds must be finite and positive,
and minimum must not exceed maximum. Random disturbances are keyed by seed,
control tick and global object ID, so reusing a different pool slot does not
change the same object's path. Desired heading changes are
slewed using a turn angle satisfying
`2*speed*sin(turn/2) <= target_accel_max*dt`; this removes spawn/profile-transition
velocity jumps. Predictive altitude braking and a velocity envelope preserve
altitude limits. Random disturbances affect desired motion before this limit.
Positions always integrate velocity. Profile references are approximated under
these motion limits rather than forcing the object onto a curve by teleporting.

Default live settings: six observers, six initial tracks, one arrival every
four simulated seconds, a pool of 24 physical tracks, no time limit. New tracks
enter at x=-12 m. A pending track crossing x=width becomes escaped. Successful
observations release their physical track slots immediately in the default
relaunch mode. Other objects release after x>width+40 m; freed slots can host new
arrivals with **fresh monotonic IDs**. With `--keep-observers`, served tracks keep
moving and remain sensed for avoidance until they leave the scene.
Dwell, reservations and proximity-history entries are reset on reuse, before
sweep snapshots. Old assignments cannot accidentally bind to a recycled slot.

When the pool is full, that scheduled arrival is skipped and `dropped_arrivals`
increments. There is no unbounded deferred queue. `arrivals`, `serviced` and
`escaped` are cumulative; `present`, `pending` and `reserved` are current counts.
`mean_service_time` is latency from each object's birth to observation completion,
not the absolute session timestamp. JSON preserves `launch_config` separately
from the effective `config` after a settings-panel restart; each result also
contains its own configuration.
Trajectory caches and visual/history buffers are bounded. A finite benchmark
uses only its initial population and ends when all are resolved or time expires.

### Observation completion and base relaunch

The CLI enables `Config.recycle_on_service=True`. The Python dataclass defaults
to False for existing fixtures and persistent-fleet experiments; the CLI option
`--keep-observers` also disables the cycle. Recycling requires `chasers_per_target=1`.
The capacity selector and +/- shortcuts cannot increase this during recycling.

After safe dwell completes, exactly the winning observer becomes inactive and
the observed object loses its presence flag. Both contact geometries are disabled
immediately; the observer's thrust and velocity become zero. It disappears from
assignment and other agents' sensor graphs. An inactive MuJoCo body remains at an
inert storage pose with gravity compensated and contacts disabled, not as a flying
or collidable craft. Cumulative observation statistics survive slot reuse.

The plant records a completion event with bay ID, generation, global target ID,
copied observer/target positions and interpolated event time. After 1.5 s by
default (`--relaunch-delay`), a replacement starts at its bay's ground pad only
when every active body is at least `min_separation+1 m` away. A blocked pad defers
launch. The initial roster starts airborne; only replacements start at the base.
Motor, orientation and angular/linear state reset; the incarnation counter
increments. Launch occurs before swept-path snapshots, avoiding a fictitious
flight segment from the old completion position to the base. Commands computed
for the previous incarnation are ignored on the launch tick.

The runner clears the old ledger reservation, radio withdrawal and peer caches
before accepting fresh telemetry. The GNN model and cumulative inference metrics
are retained. Auction restarts its allocation epoch as needed. This synchronous
simulated transport has no delayed packets crossing incarnation changes; a real
asynchronous radio would need generation IDs carried in its wire protocol.

`completed_observers` and `relaunches` count cumulative departures and launches;
`active_chasers` and `retired_chasers` are current roster counts. A finite benchmark
may finish before the last replacements launch. A green expanding/fading bubble
lasts 1.1 simulation seconds at the copied event position; base launch pulses last
0.8 s. Pausing freezes these effects. Generation-keyed trails prevent a visual
line from the old position to the base. Animations never affect physics/service.

## Observation and control mathematics

Service requires one **reserved** craft to maintain
`min_separation <= ||p_c-p_t|| < r_dock` continuously for two seconds.
Defaults are 5 m inner exclusion, 16 m outer radius and 10 m desired stand-off.
`--safe-distance` (alias `--min-separation`) configures the exclusion distance;
`--stand-off-distance` and `--observation-radius` optionally configure the other
two. Omitted CLI values use `max(10, minimum+5)` and `max(16, stand-off+6)` metres.
All values must be finite and positive with `minimum < stand-off < radius`;
minimum must exceed `2*body_radius`, so physical contact is outside the service
band. The settings panel exposes the same values under Distances. It preserves
displayed values until an increase crosses the next bound, then expands
stand-off to minimum+5 or radius to stand-off+6. Decreases stop at valid bounds.
Changes restart the seeded world and are included in effective configuration
exports. Guidance, reachability, dwell, measured violations and base-launch
clearance all consume the configured minimum. The controller's bounded force
response cannot guarantee a requested separation under every initial layout.
`r_dock` is a legacy parameter name. Dwell cannot pool between craft, survive an
inner-radius crossing, or follow a different target. One craft observes one
object at a time. With capacity above one, the first observer completing dwell
services the object once and releases every reservation on that object.

Swept relative segments resolve safe annulus intervals within each 50 Hz tick.
Escape wins simultaneous service/exit events. Minimum separations and entries
below 5 m are measured independently of service. Contact never earns service.
`ACTIVE=0`, `SERVICED=1`, `ESCAPED=2`; absent pool slots are filtered by a presence
mask. The tracker is omnidirectional; camera occlusion,
field of view and real EKF/camera/OAK-D hardware are not modeled.

```math
r=p_t-p_c,\quad v_{rel}=v_t-v_c,\quad d=\|r\|,\quad V_c=-r\cdot v_{rel}/\max(d,\epsilon),
```
```math
t_{go}=\max(d-r_{dock},0)/\max(V_c,10^{-3}),\quad
ZEM=\|r+v_{rel}t_{go}\|,\quad
P_{reach}=1-\exp[-r_{dock}^2/(2(\sigma_{base}+k ZEM)^2)].
```

Outside-band nonclosing pairs, unsafe inner-band pairs, and predicted
perimeter-before-arrival pairs have zero reachability. Already-in-band stationary
pairs may be reachable. This is a heuristic, not a calibrated safety probability.

```math
\Omega_{los}=(r\times v_{rel})/\max(d^2,\epsilon),\qquad
 a_{PN}=-N V_c(\hat r\times\Omega_{los}).
```

Target-velocity following plus radial stand-off regulation produces a velocity
reference; bounded PN assists acquisition. A velocity servo emits net
acceleration. Predictive braking uses actuator lag and sensed peers/all physical
tracks. Multiple observers receive stable angular stand-off slots. These limits
and reactive avoidance do not guarantee separation for arbitrary crowding,
noise, wind or initial conditions; measured violations are retained in results.

## Policies, reservation cap and GNN

Greedy proposes its nearest eligible object. Auction uses capability
`P_reach/(1+d/cruise_speed)`, synchronous proposal/rejection rounds, and craft-ID
ties across the simulated mesh. `coordinated` remains an alias for `auction`.
Raw Auction uniqueness applies only within connected components.

GNN pair features are `[r3,v_rel3,Vc,t_go,ZEM,P_reach]`. Dynamic edges join sensed
peers within broadcast range, with self-edges. Two 32-wide learned EdgeConv layers
and a shared head output target logits. IDs do not act as learned ownership
features. Each craft performs one forward call per due local cycle; no extra
network calls or auction rounds are inserted by the reservation manager. Cached
logits are keyed by global target ID, preventing column-shift errors on arrivals.

For every policy, the explicit ledger preserves observable/available incumbents,
then arbitrates new proposals and alternatives by score, sensed distance and ID.
Each craft gets at most one object, and each object at most K craft. This is a
stable capacity rule, not an optimal matching solver. K is a maximum, not a
promise that enough observers or geometrical clearance exist. Consequently,
Greedy no longer deliberately dogpiles in the default benchmark. Neural-only
argmax still has no global uniqueness guarantee; the shared ledger provides it.

Training uses Adam with the service leakage objective:

```math
X=\operatorname{Softmax}(W,dim=targets),\qquad
L=\sum_j\prod_i(1-X_{ij}P_{reach}(i,j)).
```

Masked unavailable/padded target columns are excluded. Training operates on
random sensor snapshots, not closed-loop reinforcement learning. The existing
17,505-parameter checkpoint remains in use; this UI/physics update does not
retrain it or establish superiority over the classical policies. The previous
300-update seed-7 training used 9,600 snapshots and reduced fixed held-out soft
loss from 3.535874 to 1.559009. Checkpoint metadata and weights load strictly on
CPU; missing/incompatible weights produce an actionable error.

## Display and controls

Default launch is fullscreen. The view uses a perspective camera, ground grid,
launch pads, metre-scaled models, perimeter markers, trails, shadows and fixed
telemetry/settings overlays. Native MuJoCo OpenGL rendering is preferred; an
explicit software perspective view is available when the process has no graphics
context. Both views read the same state; MuJoCo force integration is unchanged.
The renderer backend is shown in the HUD. No decorative collision behavior is
inferred from the visual scene.

Left drag orbits; wheel zooms; right/Shift-drag pans; Home resets; F11 toggles
fullscreen; Tab opens settings. Space pauses, R restarts, Esc exits. In continuous
mode 1/2/3 selects policy, T cycles profiles, +/- changes K only with
`--keep-observers`, and [/] changes the arrival interval. The three-column settings
panel also edits observer speed, default object speed bounds, and per-profile
speed bounds (the profile editor works independently of the active Mixed mode).
Its Distances column edits minimum separation, preferred tracking stand-off and
outer observation radius in metres. The HUD displays the active distance band.
Applying changed settings starts a fresh seeded world. HUD and
benchmark text remain in screen coordinates. Fixed physics steps do not depend
on frame rate or playback speed.

## Verification record

Reproduce the current cycle checks with:

```bash
cd hawk-swarm-simulator
./.venv/bin/python -m unittest discover -v
./.venv/bin/python main.py --headless --benchmark --seed 7 \
  --json output/benchmark-observer-cycle.json
./.venv/bin/python main.py --headless --duration 120 --trajectory mixed \
  --json output/continuous-observer-cycle.json
./.venv/bin/python main.py --headless --duration 40 --mode auction \
  --chaser-speed 14 --profile-speed weave 4 5 \
  --profile-speed constant-height 6 6 --json output/configurable-speeds-cycle.json
```

### Measured observation/relaunch scenarios

Seed 7, mixed profiles, six observers/six initial objects, K=1, CLI defaults:

| Mode | Observed | Completed at | Escaped | Peak cap violations | Minimum separation | Relaunches before end |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 6/6 | 4.28 s | 0 | 0 | 6.714 m | 0 |
| Auction | 6/6 | 4.28 s | 0 | 0 | 6.749 m | 0 |
| GNN | 6/6 | 8.24 s | 0 | 0 | 6.697 m | 5 |

Each mode completed six pair removals. Finite runs stop as soon as all objects
resolve, so they need not wait for pending base launches. No run recorded a
below-5-m separation entry or craft/craft or craft/target contact. GNN recorded
85 ground-contact substep observations from replacement craft on launch pads;
these are not 85 distinct collisions. Auction used 14 negotiation rounds; GNN
used 136 forwards and no auction rounds. GNN was slower in this seeded case.

**Continuous GNN, 120 s:** 35 admitted and observed objects, 35 pair removals,
34 replacements launched, five currently active observers and one waiting to
relaunch. Zero escaped, pending or dropped objects; zero cap violations,
below-5-m separation entries or body contacts. Minimum separation 6.684 m;
mean birth-to-service latency 5.266 s; 1,812 neural forwards. There were 561
ground-contact substep observations. `physics_contacts` is the total of
`ground_contact_observations` and `body_contact_observations`, classified from
actual MuJoCo contact geometry IDs. Counts include repeated support contacts.

**Custom-speed Auction, 40 s:** observer reference 14 m/s, weave 4–5 m/s,
constant-height 6 m/s and default bands for other profiles. All 15 admitted
objects were observed, with 15 pair removals and 14 relaunches. Zero escapes,
pending objects, capacity/separation violations or body contacts; minimum
separation 6.758 m. There were 221 ground-contact substep observations and 213
auction rounds. Evidence: `output/configurable-speeds-cycle.json`.
This run also covers replacement craft
joining the current allocation epoch after several auctions occurred while their
bay was inactive. A regression rejects backward epochs and preserves same-epoch
round monotonicity; the new incarnation advances to the trusted current epoch
before its first status packet.

Current evidence: `output/benchmark-observer-cycle.json`,
`output/continuous-observer-cycle.json`, `output/configurable-speeds-cycle.json`,
and `output/test-results-publish.txt`. Earlier local-only visual smoke captures
covered the six-second fullscreen-default CLI under SDL dummy, with three pair
removals/three relaunches.

Published media shows actual tracking, completion and base relaunch:
`docs/media/hawk-observation-demo.mp4` (20 s, 1280×800, 25 FPS, no audio),
`docs/media/hawk-observation-preview.gif` (first 12 s, 800×500, 10 FPS),
`docs/media/hawk-tracking.png` and `docs/media/hawk-completion.png`.
The recording uses seed 7, the GNN and unchanged default continuous physics.
It finishes with 10 arrivals, 9 observed, 1 pending, zero escaped, 9 relaunches,
minimum separation 6.684 m and zero separation/capacity/body-contact violations.
`docs/media/capture-provenance.json` records configuration, event timeline,
source/checkpoint hashes and encoding details. The full MP4 decoded successfully
as exactly 500 frames; the GIF has 120 frames. Recorded FPS describes playback,
not a live performance measurement.

Offscreen MuJoCo OpenGL succeeded for this publication recording. Earlier
images and automated input/fullscreen checks used SDL dummy and the software
3D view. The settings screenshot is `docs/images/configurable-safe-distance.png`.
Tests check animation expiry/pause, event-position snapshots, immediate hiding,
generation-specific trails, speed bounds, settings application, fixed HUD and
read-only rendering. Native interactive fullscreen on a desktop remains
unverified; successful offscreen capture does not establish desktop input behavior.

The model was not retrained by this lifecycle/UI change. Checkpoint SHA-256:
`4da263e0450e030d024132b64da6d19e2965e5ad6d2d4bbd0768b74eea5e9853`.

### Test scope and historical results

**111 tests passed in 11.530 s** during publication preparation, with full output in
`output/test-results-publish.txt`. Changed production modules passed
`py_compile`.

Configurable-distance validation: `--headless --benchmark --safe-distance 6`
resolves to a 6 m exclusion / 11 m stand-off / 17 m observation radius.
Greedy/Auction/GNN each observed 6/6, finishing at 4.12/4.12/8.52 s; minimum
separations were 7.672/7.672/7.614 m. All had zero separation entries, body contacts
and capacity violations. Evidence: `output/benchmark-safe-distance.json`.
Tests additionally verify explicit CLI overrides, derived defaults, invalid bands,
the service exclusion boundary, body-diameter exclusion, UI clicks/apply,
dependent bound adjustments and exported settings. The three-column panel was
rendered and inspected at 1024×768: `docs/images/configurable-safe-distance.png`.

The current suite covers force/mass response, hover, lag/thrust saturation,
wind/drag and classified contacts; trajectory speed/altitude/acceleration;
identity-stable pool reuse; safe continuous dwell; physical removal and inactive
sensor/control exclusion; delayed and blocked-pad relaunch; controller/radio
incarnation reset; capacity one/two and disconnected views; neural gradients,
checkpoints/inference; settings export, speed overrides and renderer behavior.
Production modules are also checked with `py_compile`.

Previous `*physical*`, `benchmark-capacity-two.json` and
`trajectory-validation.json` outputs belong to the persistent-fleet version
(81 tests). That version observed 35/35 in its continuous check, 6/6 in all three
capacity-two runs, and 70/72 in the two-seed profile sweep. They are historical,
not a fresh validation of the new cycle. Random-profile disturbances now use
identity-keyed noise, so old paths need not match exactly. Earlier
`benchmark-observation-*` and contact-era results are also superseded. These
older raw outputs remain local and are excluded from the published repository;
the repository contains selected current reports, the trained model and demo media.

The prior software rendering profile measured 39.65 ms per draw at 1024×768
(static camera, six craft/six tracks, excluding physics/control). This is about
25 draw FPS, not a 60 FPS guarantee; the new animation was not performance-profiled.
Stationary-scene caches remain covered by pixel equivalence and camera invalidation
checks. Native rendering is capped at 60 FPS but was not timed without a display.
