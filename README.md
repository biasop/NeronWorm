# NeuroWorm

A runnable, connectome-inspired worm survival simulation for Python 3.11+.
Twenty-four spiking neurons drive a segmented worm through a 2D Pygame world.
Food replenishes energy; a steering predator damages the worm. The right panel
shows the neural circuit, membrane activity, recent spikes, and motor/sensory
activity. No PyTorch or TensorFlow is used.

## Run locally

From this repository's root (the project is deliberately placed here rather
than inside a second `neuroworm/` directory):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Windows activation: `.venv\Scripts\activate`. After activation, the same
`pip install` and `python main.py` commands work. Pygame needs a desktop display
for normal graphical mode. Audio is not required or initialized.

```bash
# Repeatable seed and optional sensor/field overlays
python main.py --seed 7 --debug

# Full Pygame initialization and rendering without a visible window.
# Stops after 30 simulated seconds, runs as fast as it can, and saves a frame.
python main.py --headless --seconds 30 --seed 7 \
  --results results/headless.csv --screenshot results/headless.png

# Run tests, including neural timing, body steering, feeding and rendering
python -m pytest -q

# Print measured neural and body responses to bilateral cues
python tools/verify_behavior.py --output results/behavior.json

# Plot aggregate results with Matplotlib (no display required)
python tools/plot_results.py results/episodes.csv --output results/episodes.png
```

`requirements.txt` declares compatible dependency ranges; `requirements-lock.txt`
records the versions used for verification in this workspace. Install from that
file for the same dependency versions. Rendering performance varies by machine;
headless fixed-duration runs give repeatable physics for the same seed,
configuration, dependency versions, and elapsed simulation time.

## Controls

| Key | Action |
| --- | --- |
| ESC / close window | Quit and record the current partial episode |
| SPACE | Pause/resume all simulation clocks |
| R | Record the current episode, reset worm/world/brain, keep aggregate history |
| B | Toggle neural circuit display |
| D | Toggle sensor positions, illustrative field rings, and diagnostic text |
| UP / DOWN | Double/halve simulation speed, between 0.25x and 8x |

`--seconds` stops a run after simulated time, `--fps` controls rendering,
`--speed` sets initial speed, `--connectome PATH` supplies another JSON network,
and `--results PATH` changes the CSV destination. Run `python main.py --help` for
all options. Headless runs default to ten simulated seconds if no duration is
provided. Durations are quantized down to whole physics ticks (10 ms by default).

## Architecture

```text
main.py                       CLI, Pygame events, render accumulator
config.py                     immutable simulation configuration
simulation.py                 orchestrates sensing → brain → muscles → world
episode_stats.py              episode aggregates and CSV persistence
brain/
  neuron.py                   Leaky Integrate-and-Fire dynamics
  synapse.py                  chemical connections, gap junctions, heap queue
  connectome.py               structural metadata, validation, JSON I/O
  brain.py                    one neural clock, state, spike-rate readout
organism/
  sensors.py                  bilateral local concentration/touch sensing
  muscles.py                  filtered neural motor readout
  worm.py                     segmented kinematic body and energy
environment/
  food.py, predator.py        environment objects
  world.py                    seeded food respawn and collision interactions
visualization/
  brain_view.py               cached NetworkX layout, Pygame activity display
  hud.py                      survival, clock and neural activity readouts
data/demo_connectome.json     24 neurons, 40 chemical connections, 5 gaps
tools/
  generate_demo.py            regenerate the artificial demo wiring
  verify_behavior.py          reproducible cue-response diagnostic
  plot_results.py             static Matplotlib aggregate charts
tests/                        neural, structural, behavioral and render tests
```

`brain/neurons.py` and `brain/synapses.py` remain as import shims for the original
plural filenames; new code uses the singular module names. The neural and
physical models can run without importing Pygame. Neurons use simple dataclasses,
connections use IDs, and NumPy stores activity/body arrays for later vectorization.
There are no per-neuron loops running independently and no neuron objects stored
inside connections.

### Neural timing and units

The default neural timestep is **1 ms**. Each **10 ms** physics tick samples local
sensors, applies their currents for ten neural steps, then updates muscles,
movement, energy, predator, food and collisions. The render loop uses an
accumulator, so 30/60/120 FPS do not change neural dynamics for the same elapsed
simulation time. Speed changes how quickly simulated time advances. Graphical
wall-clock intervals are capped at 100 ms to avoid a catch-up spiral after long
stalls; headless mode advances a fixed interval per rendered frame.

Each brain step delivers due heap events, computes gap currents from a shared
voltage snapshot, integrates all neurons, timestamps spikes at the end of the
interval, schedules their outgoing events, and updates activity statistics.
A spike at `t` arrives at `t + delay_ms`; events are delivered at the first step
boundary at or after that arrival. Zero-delay events are delivered at the next
step boundary, never recursively within a neuron update.

The LIF equation is `tau * dV/dt = -(V - V_rest) + I`, using Euler integration.
Defaults are rest/reset −70 mV, threshold −55 mV, tau 20 ms, refractory period
2 ms. Input current is **mV-equivalent drive**, with resistance absorbed into its
scale; this is not a calibrated physical current in amperes. Input sums clear
after each step, including refractory steps. Sustained stimulation must be
reapplied each neural step.

Chemical connections transmit signed spike-triggered current impulses, not
voltage. A weight specifies the current for a 1 ms impulse; for smaller steps the
brain divides it by `dt_ms`, preserving the integrated impulse area. A weight of
340 changes an otherwise resting tau=20 ms neuron's voltage by approximately
17 mV. Positive weights excite and negative weights inhibit. The demo weights
are deliberately strong relay impulses, not biological conductance estimates.
Gap junctions apply `g*(V_a - V_b)` as opposite continuous currents every step.
The demo uses small conductances; arbitrary imported conductances may require
additional stability checks and a smaller timestep.

Motor activity is an exponential spike-rate estimate with a 100 ms time constant,
normalized by 180 Hz, then clipped to [0, 1] and filtered by an 80 ms muscle
time constant. Raw activity bars may exceed 1; only muscle activation is clipped.
`turn_signal = right - left`; `forward_signal = left + right`. Angle increases
clockwise in screen coordinates, so a stronger left motor turns toward the left.
Body speed is proportional to the sum of the two muscle activations.

### Sensory and reflex behavior

Each probe samples `exp(-alpha * distance)` from food or danger sources. Food
contributions sum and clip to [0, 1]; danger comes from the predator. Sensors sit
ahead of the head and on either side. Concentrations map to bounded currents
(food 65, danger 105 mV-equivalent at concentration 1), so normal sensory inputs
take several milliseconds to produce a spike. Only scalar readings enter the
brain. The environment knows object locations for drawing/collision; muscles and
the locomotor calculation never inspect food or predator coordinates.

The demo has eight sensory neurons, twelve interneurons, and four motor neurons:

- Food pathways excite the motor on their own side and weakly inhibit the other,
  producing turns toward food.
- Danger pathways excite the opposite motor, inhibit their own motor, and
  suppress food relays, producing turns away from danger.
- Paired artificial tonic inputs pass through forward interneurons to provide
  movement when safe. Removing neural drive stops the worm.
- Wall-touch pathways pass through interneurons. A deliberately asymmetric
  handed reflex breaks symmetry at head-on wall collisions, so the worm escapes
  rather than stopping against the wall.
- Tail motor neurons and five small gap junctions demonstrate electrical coupling;
  body steering reads the two primary motors. Segments follow kinematically.

The predator uses capped straight-line pursuit rather than a brain. Food is
stationary, respawns using the seeded random generator, and restores energy up
to 100. Energy decreases with time and movement. Predator contact anywhere on
the body causes damage with a one-second cooldown. Energy or health exhaustion
records an episode and resets the neural clock, event queue, world and body;
total simulation time and episode history continue. R resets an episode using
the continuing RNG sequence; restart with the same seed to reproduce a session.

## Logging and verification

The default `results/episodes.csv` is appended once at death, manual reset, or
session close. Columns contain seed, episode number, completion reason, survival
duration, food eaten, predator hits, total spikes, and mean firing rate per neuron
(`total_spikes / neuron_count / duration_seconds`). `session_end` and
`manual_reset` rows are partial episodes, so distinguish them when comparing
survival experiments. Episode numbers restart for a new process; use separate
`--results` files to keep experiments separate. No timestep data is written.
The HUD's spikes/s is the total network rate over the last simulated second.
Console summaries every five simulated seconds provide position and motor output.

Tests cover depolarization, reset, refractoriness, leak, positive/negative inputs,
chemical delay/order, gap currents, JSON round trips and validation, sensory →
interneuron → motor propagation, reset, deterministic render batching, actual
food/danger steering, wall escape, eating/respawn, predator cooldown, starvation,
CSV aggregation, and the real headless Pygame entry point. A longer seeded survival
test also requires the demo worm to eat food. Rendering smoke checks save PNGs;
the normal graphical window has also been launched in this workspace.

## Biological assumptions and limits

**This is NOT a complete simulation of a real C. elegans nervous system.**
It uses LIF neurons, weighted chemical synapses, simplified gap junctions,
artificial sensor mappings, artificial motor mappings, and manually designed
reflex wiring. Many worm neurons use graded potentials; this model deliberately
uses spikes throughout. It omits neurotransmitter kinetics, neuromodulation,
muscle physiology, realistic biomechanics and learning. The segmented body is
kinematic rather than a physics model of crawling. Primitive steering can miss
food, compete with predator avoidance, or become temporarily trapped; it is not
an optimized survival policy.

An **anatomical connectome** describes cells and their connections. **Functional
neural dynamics** additionally require excitatory/inhibitory signs, strengths,
delays, cell equations, external inputs and body coupling. Anatomical connectivity
alone does not determine behavior. Replacing the demo JSON with real connectivity
will not by itself produce a realistic worm.

Future versions may import real C. elegans connectivity from OpenWorm or other
C. elegans connectome datasets, eventually covering the 302-neuron adult
hermaphrodite nervous system. No dataset download is required for version 1.
Importers should create the structural JSON and supply explicit sensor/motor
mapping metadata for the organism adapter. Unknown per-neuron metadata such as
neurotransmitters survives JSON round trips; dynamic parameters go in the
`parameters` object. Synaptic signs and weights still need justified inference.

The structure/state split supports future neurotransmitter metadata and sign
inference, STDP, reward-modulated STDP, structural plasticity, multiple worms,
evolution, alternate environments, and a Brian2 backend. Those features are
extension points, not implemented capabilities. The recommended next step is
to compare the measured demo cue responses across several seeds, then add a
dataset importer while retaining these neural timing and behavior checks.
