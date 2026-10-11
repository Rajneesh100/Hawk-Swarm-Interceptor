# HAWK architecture

[SPEC_AND_STATUS.md](SPEC_AND_STATUS.md) is the authoritative specification and
verification record.

```text
Trajectory scheduler → bounded physical track pool with unique IDs
                              ↓
MuJoCo force plant → noisy local Observation → kinematic estimator
       ↑                                      ↓
       │                        Greedy / mesh Auction / learned GNN
       │                                      ↓
       │                 explicit shared capacity-reservation ledger
       │                                      ↓
       └──── acceleration commands ← stand-off and separation control

Mission reservation metadata → continuous safe-range observation dwell
World snapshots → read-only shaded MuJoCo scene → Pygame HUD and settings
Random sensor snapshots → Adam leakage training → versioned GNN checkpoint
```

Physics uses SI units and fixed substeps; view timing never drives the plant.
Controllers receive copied local observations, never a plant reference. The
reservation ledger sees local candidate scores and IDs, not true world poses.
It is nevertheless a shared mission service: global capacity enforcement across
radio partitions is not a decentralized guarantee from the neural policy.

A craft follows at most one track, and each track accepts at most the configured
number of simultaneous observers. Mission metadata constrains dwell credit;
physical integration continues to accept only acceleration commands. After the
same observer maintains the configured safe observation band for two seconds,
the default mode removes that observer and track from physics and allocation.
A green completion bubble marks the event. A fresh observer launches from its
base pad after the relaunch delay and clearance check; generation IDs reset
mission and radio state without reloading the learned model. With
`--keep-observers`, completed tracks remain physical until leaving the scene
and observers stay available for reuse. Contact never awards service.

Continuous mode recycles a bounded pool after completion or physical exit, gives each new
arrival a fresh ID, and continues until quit or a configured horizon. Benchmark
mode disables arrivals and replays the same seeded initial tracks for all three
policies. Physics, reservations, motion profiles, and service rules are shared.

The renderer owns its camera and visual scene. Orbit, pan, zoom, fullscreen,
and HUD layout do not mutate simulation state. Applying changed world settings
explicitly starts a fresh scenario rather than changing physics midway through
a recorded benchmark.
