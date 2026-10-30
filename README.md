# DodgeAI

A toolchain for training a reinforcement-learning agent to play the PICO-8 game
`dodge`, and for watching and playing it locally.

This repository contains **no game content**. The game runs on Taylor Turner's
native Rust port of the cartridge, installed separately as the `dodge_native`
Python module; this repository ships the environment, the policies, the
training dashboard and the tools around them.

## Setup

    pip install -r requirements-train.txt

Then build the engine. It lives in its own repository and is installed as a
Python extension with [maturin](https://www.maturin.rs/) and a Rust toolchain:

    git clone https://github.com/TaylorTurnerIT/dodge-ai-3 ../dodge-ai-3
    cd ../dodge-ai-3 && git checkout velocity-flow   # until the PR is merged
    pip install maturin
    cd native/crates/dodge-python && maturin build --release
    pip install ../../target/wheels/dodge_native-*.whl

The `velocity-flow` branch adds what this project needs on top of Turner's
engine, all opt-in: the VelocityFlow observation computed natively, one-frame
decisions, death telemetry, Taylor Turner's gameplay trace (his death
diagnostic, recorder and enemy identities from `dodge-native`, read per lane
with `lane_trace`), and the training options (lives, enemy toggles, spawn
rate). Training never turns the trace on. Without the module the tooling still imports, and the tests that
need a game skip themselves.

## Play it

    python play.py --difficulty hard

Arrows or WASD to move, X/V or Z/C to start another game after a death, Escape
to quit.

## Web dashboard

Build the page once, then train from the browser:

    cd web && pnpm install && pnpm run build && cd ..
    python dashboard.py --architecture velocity-flow-v2 --envs 32

It opens <http://127.0.0.1:8765>, with the same controls as the window below
(pause, watch, game settings, reward weights, save and load), and three views:

* **Training**: the run's metrics, and the worst quarter, mean and best quarter
  of recent games at every update.
* **Evaluation**: the paper's figures (survival curve, what ended each game by
  window, deaths with and without a pattern, the deadliest patterns), from
  recent training games, or from **Evaluate**, which freezes the live policy
  and plays it on fixed seeds from 1,000,000 while training carries on.
* **Death paths**: on request, plays the live policy on fixed seeds with the
  engine's trace on and records every pattern appearance, survived or died.
  Pick a pattern to see all its appearances at once over its rectangles
  (deaths coloured by cause, survived ones grey); click one to watch it as the
  real game, from the pattern's start to its end or the death. Replays are
  re-simulated from the seed and the recorded decisions, and say so if they
  ever stop matching the recording.

Every chart has a table view. For frontend work, run `pnpm run dev` in `web/`
alongside `dashboard.py`; it proxies `/api` to the server.

## Training dashboard

Start PPO training in the Dodge dashboard window:

    python training_gui.py

Training runs on a background thread so the window stays responsive. Pause/Resume controls
rollout collection, Rewards opens the reward weights, and Game Config stages difficulty,
powerups, patterns, normal/kamikaze enemy switches, spawn rate, and decision-interval
changes. Applying game settings checkpoints the
live policy, rebuilds its environments, resumes the same model, and saves the
settings to [`game-config.json`](#game-settings) so the next launch starts where
this one left off. Models saves or loads real
Stable-Baselines3 `.zip` checkpoints, and Watch Agent opens the current policy in a second
rendered game window with the active game configuration. Each watch launch saves an immutable snapshot, chooses a new random
episode seed, and labels the game window with its exact PPO update and seed. The active
model name is shown in Learning Statistics and carried into the watch snapshot
filename and window title. Saving names the active in-memory policy; loading changes the
active policy to the selected model. Training continues until the dashboard is closed;
`--steps` controls how often the current policy is automatically saved over the active
model's own checkpoint (a fresh model takes its architecture's name, numbered if that file
already exists), not when the session ends. Every PPO update also appends one line of
training metrics to `<model>.metrics.jsonl` beside it.

Every chart spans the whole run rather than a recent window, so improvement stays visible
as training goes on. Each draws the raw per-episode values faintly beneath a smoothed
trend line, keeps its y-axis anchored to the run's full range, and reports the change
since the run started. The Progress panel states the same thing in numbers. History is
held in memory only, so restarting the dashboard starts a fresh curve.

Use `python training_gui.py --screenshot out.png` to render one frame of sample telemetry
without training — useful for working on the dashboard itself.

The dashboard is meant to cost a fraction of a core, since it competes with eight
environment processes for four of them. Watching it is not free, but it is close:

| | Before | Now |
| --- | --- | --- |
| Render | 45.7 ms | 7.5 ms |
| Telemetry lock held | the whole render | 25 us |
| Redraw rate | 60 attempted, ~22 achieved | 20 |
| Share of a core | ~100% | ~15% |

The lock mattered more than the drawing. The PPO callback takes the same lock on every
step, so rendering while holding it stalled the trainer — and every environment
subprocess behind it — for the length of a frame. The dashboard now copies telemetry
under the lock and draws from the copy, which is what `DashboardSnapshot.copy()` and
`RunSeries.copy()` exist for; both series are capacity-bounded, so the copy is cheap by
construction. The render itself was dominated by allocating and blitting a full-window
alpha surface for each of eleven chart fills, and by redrawing 2,560 background dots that
never change; the overlays are now plot-sized and the backdrop is cached per window size.
Input is still polled at 60Hz so clicks and drags stay responsive — only the redraw is
throttled.

### Dead units

Learning Statistics reports the share of the policy trunk that produced zero for
an entire rollout, amber past 10% and red past a third. It is there because a
2.3M-step run died silently: 56% of the shared features and **every** unit in the
policy trunk went permanently zero, so the network emitted one fixed action
distribution no matter what it saw. Under `deterministic=True` — what Watch Agent
uses — that distribution's argmax never changes, and the agent stands perfectly
still.

Nothing else on the dashboard showed it. Reward, steps/second and update count
all looked normal, and entropy stayed near uniform the whole time, because a
constant distribution is still a high-entropy one. Only survival time drifted,
and slowly enough to read as noise.

Every activation is now a `LeakyReLU`, so a unit whose pre-activation is negative
everywhere still receives gradient and can recover. If dead units climb anyway,
the next levers are a lower or annealed learning rate and normalising the
extractor's output; the run to watch is one where the figure rises steadily
rather than one where it starts high.

### VelocityFlow

The GUI starts without a model and opens its Models panel. Load a discovered
checkpoint, or explicitly ask it to create a new policy with `--architecture`:

    python training_gui.py --architecture hybrid-cnn

The panel watches `artifacts/models` (including user-created subdirectories)
for `.zip` checkpoints. Point it at a different model library when needed:

    python training_gui.py --models-dir path/to/models

Its checkpoints save, list and load through the dashboard's Models panel
exactly like `hybrid-cnn`'s. Stable-Baselines3 records the policy class in the
checkpoint, and the input-migration check keys off layer names a field policy
does not have, so nothing special-cases it.

The Models panel lists the checkpoints on disk, and loading one switches the
running session's architecture — so an architecture with nothing saved yet
cannot be picked there, because there is nothing to click. Write a starting
point for it:

    python tools/new_policy.py field-cnn

That leaves an untrained `field-cnn.zip` in `artifacts/models`, after which the
architecture is selectable from the dashboard without relaunching.

The network's only output is a scalar danger per arena cell; it never picks an
action. Choosing is arithmetic — simulate where each of the nine actions takes
the player, read the field along those paths, and let the totals be the logits
— so every bit of learning pressure lands on the field.

| | hybrid-cnn | velocity-flow |
| --- | --- | --- |
| Parameters | 642,826 | **102,968** |
| Training throughput | 475 steps/s | 130 steps/s |
| Largest single layer | `Linear(4096, 128)`, 87% of the model | none over 37k |
| Output | 9 logits | 32x32 danger field |

Three reasons for the shape. It never flattens, so it drops the dense layer
that held 87% of the hybrid model's weights and was the layer that died.
Translation equivariance survives to the output, so "dodge a threat closing
from the left" is learned once rather than separately at each position. And the
field is a picture, which after a collapse that hid behind healthy-looking
entropy for two million steps is worth something on its own.

Early results over ten updates are mixed and should not be read as a verdict:
explained variance is markedly better (0.62-0.77 against 0.33-0.49), while
survival is behind at that point. Both were still climbing.

**Watching it.** `python watch_ppo.py MODEL` draws the field as the game's
background, viridis from dark blue (safe) to yellow (dangerous). The cart
clears to palette index 12 and draws everything over it, so replacing exactly
those pixels puts the field behind the action with no change to any drawing
primitive; only `cls()` and the score bar's backing use that index. The field
is computed once per decision and left in place for the four frames rendered
before the next one, so watching costs one extra forward pass per decision
rather than one per frame, and training is untouched. `--no-field` turns it
off.

Its known ceiling, established by the analytic version below: a scalar over
positions cannot express pursuit. "Step sideways because it is chasing you" is
a fact about trajectories, not about any cell. Sampling along each action's
path recovers part of it; if that binds, the fix is a field conditioned on
time, not a deeper trunk.

### VelocityFlow v2

    python training_gui.py --architecture velocity-flow-v2

v1 stays registered and selectable. That is the point: v2 changes the network
and the optimizer schedule together, and without a fixed baseline there is no
way to tell which one moved a result.

Two changes to the network:

* **The critic can see space.** v1 hands the value function
  `trunk.mean(dim=(2, 3))` -- 32 channel means over the whole field -- so its
  entire view of the layout is an average, while "am I about to die" is almost
  entirely a question of layout. v2 adds a 4x4 arena summary at 8 channels,
  128 numbers behind a 1x1 convolution. The policy keeps full resolution all
  the way to the field; only the critic was throwing it away.
* **The trunk's middle runs at 16x16.** Roughly 79M multiply-accumulates a
  sample down to 22M. Built with `half_resolution=False` it reverts to v1's
  full-resolution trunk, because this is the one change that trades against
  the design thesis: the field it draws is smoother, and if what makes the
  controller work is precision at a threat's edge, this is what costs it.

And a per-architecture hyperparameter table, so v1 keeps the values every
curve recorded so far was trained under:

| | velocity-flow | v2 | why |
| --- | --- | --- | --- |
| `gamma` | 0.995 | **0.99** | 0.995 is a ~13s horizon. Which enemies spawn and which pattern comes next are not predictable from the current frame, so most of that is noise in the value target, capping explained variance whatever the critic does. |
| `n_epochs` | 10 | **4** | Learning is ~5s of a ~6s cycle and this is the multiplier. |
| `target_kl` | none | **0.02** | Makes the cut adaptive: an update that has not moved the policy still gets its passes. |
| `vf_coef` | 0.5 | **1.0** | The critic is the weak half. |
| minibatch | 128 | **256** | See below. |

**Measured, GTX 1070.** Per-sample optimizer cost is flat from 128 to 256 and
then degrades badly -- 0.32 ms/sample at 256 against 0.41 at 512 and 2.88 at
2048, which is memory pressure, not compute. 256 is the floor of that curve,
and anything larger makes the learning phase slower rather than faster.

*Re-measured on torch 2.13 cu126 (rough, one optimizer step on random
observations):* ~0.24 ms/sample, flat from 256 to 2048, peaking at 3.5GiB of
GPU memory at 2048. The degradation above did not reproduce, so 256 stays as
the value every v2 curve was trained under at the default buffer, but it is
no reason to keep larger buffers below 512 -- see
[Many envs on a GPU](#many-envs-on-a-gpu).

| | batch | ms/step | steps/rollout | learning per rollout |
| --- | --- | --- | --- | --- |
| velocity-flow | 128 | 40.6 | 160 | 6.50s |
| velocity-flow-v2 | 256 | 42.9 | 32 | **1.37s** |

A 4.7x cut, against roughly 1s of collection either way -- so about 3.2x on the
whole cycle. Most of it is `n_epochs`; the trunk's contribution is that it made
the doubled batch cost the same per step as v1's, at 0.168 ms/sample against
0.318. v2 is very slightly larger in weights (111,424 against 102,968) because
of the critic summary, which is not the same thing as slower.

None of this says v2 is better at the game. It is faster per update and the
critic is better fed; whether that shows up in survival is what the two
architectures sitting side by side are for.

### Gradient bot

    python gradient_bot.py              # measure against random
    python gradient_bot.py --watch      # play one episode in a window

The same controller reading a hand-written field instead of a learned one, in
`dodge/gradient.py`. It exists to calibrate: nothing here knew what a good
survival time was, so no agent's score could be judged.

Its forward model is exact — player and enemy motion both reproduce the cart to
0.000px over 24 frames, pinned by tests — which is what makes it a useful
control. When it plays badly, the danger heuristic is at fault and nothing
else. Two findings came out of it:

* The cart punishes camping ([dodge.lua:711](dodge.lua#L711)). Linger with the
  player box overlapping 60..68 on both axes for 50 frames and it spawns an
  enemy at every spawn point. This is invisible in the observation — nothing
  about four new enemies says they were summoned — and it is why the first
  version of this bot scored *below* random while standing at the centre. It
  is also what the old centre-peaked reward was paying the agent to do.
* Homing must be predicted against the candidate path, not the player's
  current position. Freezing it there quietly rewards running directly away
  from a pursuer, because the predicted enemy carries on toward the spot the
  player has left. Against a chaser the escape is lateral.

### Training lives

    python training_gui.py --lives 3

Also a slider in **Game Config**, and off by default. It exists to widen the
range of states training sees, cheaply, without implementing random
initialisation.

With lives set, the first N lethal enemy or pattern collisions leave the player
standing. Enemies are deleted as usual; a lethal pattern is restored to the
state it had when selected so the same pattern begins again. The agent's episode
still ends at each hit, charged the full `death_penalty` — the cartridge run is
what survives, so the next episode opens on a mid-game arena instead of the
spawn point. Measured over 40 episodes at `--lives 3`, 30 of them started
somewhere between frame 145 and frame 553 of a run already in progress; at the
default they all start at frame zero.

Ending the episode is the part that matters. If the episode continued, the
value function could see that a hit buys about 100 frames of survival income
against a penalty of 1.0, and the agent would learn to spend hits to escape
tight corners. Cutting the episode makes that arithmetic unreachable: SB3
bootstraps `V = 0` at a terminal, so nothing after the collision is ever
credited to the actions that led into it. For the same reason the step is
reported as `terminated`, never `truncated` — a truncation bootstraps `V(s')`
and would leak back exactly the value being hidden.

Two consequences worth knowing about:

* **Survival time still measures the run.** An episode now ends at every
  absorbed hit, so charting episodes would quietly redefine the dashboard's
  headline metric as time between hits, against curves recorded when it meant
  time alive. The chart follows `run_frames` and is appended only when the
  cartridge run really ends. Episode reward and enemies killed remain
  per episode. At `--lives 0` the two are the same number.
* **The variety is biased, and correlated.** Every state still lies on one
  trajectory, so this is weaker than real random initialisation. Restarts also
  cluster on post-collision states — crowded, and at whatever difficulty the
  score has ramped to — which is an implicit curriculum toward danger, but it
  does mean the critic is fit on a start distribution the real game never has.

A pattern crush spends a training life just like an enemy collision and ends the
current RL episode without ending the cartridge run. Before the next episode
resumes, the active pattern is restored to the exact state it had when selected,
including randomized geometry and pattern-controlled spawn points. This removes
the lethal shape immediately and retries the same pattern without advancing the
cartridge RNG.

Watch Agent never inherits the setting. A policy that gets three free hits is
not the one being shipped, so the watch window always shows real deaths.

### Game settings

**Game Config** applies its settings to the running session *and* writes them to
`game-config.json`, so a launch resumes the curriculum the last one was tuned to
instead of dropping back to the shipped defaults:

    {
      "difficulty": 1,
      "patterns": false,
      "powerups": true,
      "normal_enemies": true,
      "kamikaze_enemies": true,
      "spawn_rate_percent": 100,
      "action_repeat": 4,
      "lives": 0
    }

Unlike `rewards.json`, the file is not read by the environments: applying a
change rebuilds the training games directly, and the file is only consulted at
startup. So an edit made while the dashboard is running takes effect on the next
launch, not the next episode. A missing or malformed file falls back to the
defaults, and out-of-range values are clamped rather than refused.

The file is written only after the session accepts the settings, so it records
what training actually ran. Command-line flags still win for the run that passes
them — `--difficulty hard` overrides a saved `easy` without changing the file —
which makes one-off experiments possible without disturbing the saved tuning.

### Reward weights

The **Rewards** button opens sliders over the reward function. They write
`rewards.json`, which is equally editable by hand:

    {
      "survival_per_frame": 0.01,
      "edge_penalty": 0.5,
      "death_penalty": 1.0,
      "score_weight": 1.0,
      "uncontrolled_score_weight": 0.0
    }

Environments re-read the file when an episode starts, so a change reaches every
training process from the next episode without restarting anything. That
indirection is deliberate: environments run in separate processes, so a slider
cannot reach them through memory. A missing or malformed file falls back to the
defaults rather than stopping a run.

`edge_penalty` is the survival credit given up against a wall. Survival pays a
flat rate across the middle 60% of the arena and tapers only in the outer band,
so the agent is discouraged from cowering in a corner — where there is nowhere
left to dodge — without being paid to hold any particular spot.

It replaces a `center_bonus` that peaked at the exact centre and fell away in
every direction. That shape had a bug in it: the agent spawns at the centre, so
*any* movement lowered its income and standing perfectly still was the
best-paying action on the board. A file still using the old key is read at its
old value; only the name and the shape changed.

`uncontrolled_score_weight` is worth knowing about. The cart credits 0.5
whenever an enemy dies to *another enemy* or to a pattern, which the agent does
not cause — measured, that was about 15% of the reward's variance and none of it
predictable from the agent's own state. At the default of 0 that score is
stripped out and only agent-caused score is paid; set it to 1.0 to restore the
cart's own scoring.

Changing weights mid-run shifts the value function's targets, so expect
explained variance to dip while the critic re-fits, and treat reward history
either side of a change as two different experiments.

### GPU

The simulation cannot use a GPU — PICO-8 is a serial interpreter on the CPU. The
policy update is ordinary torch, and it can. `--device` defaults to `auto`.

This needs a CUDA build of torch; the default wheel on PyPI is CPU-only, and a
CPU-only build cannot use a GPU even when the machine has one. The System panel
reports which device is actually in use, including that case:

    pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126

Measured on a GTX 1070 with the real training configuration — eight envs, 256
rollout steps, 2,048 samples an update, `velocity-flow-v2`, timed at
Stable-Baselines3's own rollout boundaries:

| | Collect | Learn | Full cycle | |
| --- | --- | --- | --- | --- |
| CPU | 2.85s | 18.28s | 21.12s | 97 steps/s |
| GPU | 2.01s | 2.47s | **4.48s** | **457 steps/s** |
| | 1.4x | **7.4x** | **4.7x** | |

The collect column is the one worth reading twice. Collection is dominated by
the PICO-8 simulation, which is a serial CPU interpreter and cannot move — but
each step also runs one forward pass, and `velocity-flow-v2` is fully
convolutional at 32x32, so that pass is expensive enough on a CPU that the GPU
wins collection too despite paying a per-step transfer. The obvious worry, that
shipping 8x3,076 floats across the bus each step would make collection slower
on a GPU, is measurably wrong.

An older table here reported 7.06s against 1.90s for `hybrid-cnn` at 512
samples an update. Those numbers are superseded: different architecture,
different buffer, and `velocity-flow-v2` cut its own update further with
`n_epochs=4` and `target_kl`.

`auto` only reaches for a GPU where it measurably helps: `hybrid-cnn`,
`velocity-flow` and `velocity-flow-v2`. A quick check of one 256-sample
optimizer step (rough, CPU shared with other work) put the two field policies
at about 9x and 11x faster on the GPU. The MLP update is unchanged on a GPU
(0.686s against 0.693s) because its layers are too small to cover transfer and
launch overhead, so `auto` keeps `mlp` on the CPU; the same quick check hinted
otherwise on a saturated CPU, which is worth a quiet-machine benchmark before
anything moves. An explicit `--device cuda` is always honoured.

`auto` is resolved against the architecture actually being trained. The
dashboard starts with no architecture and waits for the Models panel, and it
used to resolve `auto` once at startup, so every run loaded from a checkpoint
-- v2 included -- trained on the CPU. A loaded checkpoint now resolves it by
its own architecture, and `train_ppo.py` uses the same rule (previously it
handed `auto` straight to Stable-Baselines3, which picks any GPU for any
architecture).

Stable-Baselines3 warns that a non-CNN policy on a GPU may be slower. It reads
the policy class, which is `ActorCriticPolicy` for both architectures here, so
it cannot see that `hybrid-cnn`'s features extractor is convolutional. That
warning is suppressed for this project's models, on the strength of the numbers
above.

With the update on a GPU the balance flips: simulation becomes roughly half of
the cycle rather than a fifth, so it is the next thing worth optimising.

### Parallel environments

Training defaults to eight games stepped in parallel processes, 256 steps each:

    python training_gui.py --envs 8 --rollout-steps 256

PPO's buffer holds `envs * rollout-steps` samples per update, which the defaults
keep at 2048. The two factors are set for different reasons.

**Envs, not steps, for parallelism.** Collection makes **one forward pass per
step regardless of batch size**, and at these sizes that pass is dominated by
kernel launches rather than arithmetic, so more envs cut the number of passes
while the simulation work is unchanged. Measured at a constant 512-sample
buffer, update time identical throughout:

| Envs x steps | Collect | Full cycle | Throughput |
| --- | --- | --- | --- |
| 4 x 128 | 0.92s | 2.13s | 241 steps/s |
| 8 x 64 | 0.53s | 1.73s | 296 steps/s |
| 16 x 32 | 0.44s | 1.64s | 313 steps/s |

16 environments is fastest but is not the default, because shortening the
rollout costs advantage quality — see below — and the weak part of this system
is the critic.

**Rollout length is close to free**, as long as the minibatch scales with it.
Collection and the update both grow with rollout length, so samples per second
is roughly flat and mainly the update cadence changes:

| Envs x steps | Minibatch | Collect | Update | Cycle | Throughput |
| --- | --- | --- | --- | --- | --- |
| 8 x 64 | 64 | 0.50s | 1.15s | 1.65s | 310 steps/s |
| 8 x 256 | 64 | 1.90s | 5.02s | 6.92s | 296 steps/s |
| 8 x 256 | **128** | 1.96s | 3.18s | 5.14s | **399 steps/s** |

The minibatch is `buffer / 16`, floored at 64 and capped at 512 (v2: `buffer
/ 8`) — see `minibatch_size()`. Holding
it at a constant 64 would have put 32 minibatches in every epoch at this buffer
size, and the update is already most of the cycle; letting it grow keeps the
optimizer at 16 passes an epoch and is *faster than the old default* despite
learning from four times as much experience per update, because a bigger
minibatch covers the launch cost of a GPU pass better.

Length itself is worth spending on for two reasons. It cuts gradient noise by
drawing each update from more fresh experience. And it shrinks how much of each
advantage rests on the value estimate at the rollout boundary: with
`gamma=0.995` and `gae_lambda=0.95`, the final sample's advantage is 99.5%
bootstrap, the one before it 48%, falling below 10% about eight steps back.

| Rollout | Samples over 10% bootstrap | Buffer mean |
| --- | --- | --- |
| 64 | 8 / 64 (12%) | 5.5% |
| 128 | 8 / 128 (6%) | 2.8% |
| 256 | 8 / 256 (3%) | 1.4% |

Episodes run about 100 decisions at present, so a 256-step rollout also spans
more than one full episode per env — most of the buffer's terminals are real
deaths rather than bootstrapped guesses.

Note that rollout boundaries do not cut episodes: PPO carries `_last_obs` across
collections, so an episode simply continues into the next rollout, and a death
*inside* a rollout already contributes a true terminal through `episode_starts`.
The boundary costs an advantage estimate, not an episode.

Every environment is a lane of one Rust batch, stepped in a single call in this
process. The tables above were measured on the earlier Lua runtime, one process
per environment; their reasoning about buffers and rollouts still holds.

### Simulation speed

The Rust engine runs the game at 300,000-520,000 frames a second across a batch,
against about 4,000 per process for the Lua runtime it replaced, so simulation
is no longer what training waits on. Measured on a GTX 1070 with
velocity-flow-v2 at one frame per decision, hard difficulty with patterns, per
PPO update (four epochs at most, `target_kl` 0.02):

| Envs x steps | Minibatch | Optimizer steps (of max) | Collect | Update | Throughput |
| --- | --- | --- | --- | --- | --- |
| Lua, 8 x 256 | 256 | | 3.1s | 1.9s | 547 steps/s |
| Rust, 8 x 256 | 256 | 32 of 32 | 1.3s | 2.3s | 567 steps/s |
| Rust, 32 x 256 | 1,024 | 32 of 32 | 1.5s | 8.4s | 833 steps/s |
| Rust, 64 x 256 | 2,048 | 32 of 32 | 1.7s | 16.8s | 885 steps/s |
| Rust, 32 x 256, capped | 256 | 37-61 of 128 | 1.5s | 3.3s | 1,732 steps/s |
| Rust, 64 x 256, capped | 256 | 46-81 of 256 | 1.7s | 4.3s | 2,704 steps/s |
| Rust, 128 x 256, capped | 256 | 53-88 of 512 | 2.7s | 5.4s | 4,027 steps/s |

Collection is about fourteen times faster per sample, so the gradient update is
the cycle. Its cost is set by how many samples it processes, not by the
minibatch size: a sample costs 0.26-0.28 ms per epoch from 256 to 2,048.

Uncapped, the minibatch is an eighth of the buffer, so an update always takes
32 optimizer steps. Steps that large never reach `target_kl`, so every update runs
all four epochs and the update grows with the env count. The "capped" rows hold
the minibatch at 256 instead: more, smaller steps reach the KL target after
about one epoch and the update stops there. That is three times the throughput
at 64 envs, but each update does about a quarter of the optimization, so whether
it learns as much per hour has to be measured on a learning curve, not a stopwatch.
The default cap (below) is 512, not 256: it binds above a 4,096-sample buffer,
so 8 x 256 is unchanged and 32 x 256 or more trains on Turner's schedule.

Uncapped, 128 envs means a 4,096-sample minibatch, which ran at about 100
steps/s, so keep to 64 or fewer without the cap. The env count also sets the
buffer size.

### Many envs on a GPU

The recommended shape for a GPU run is Turner's: 64 envs x 256 steps.

    python train_ppo.py --architecture velocity-flow-v2 --preset gpu

`--preset gpu` is 64 x 256 and `--preset default` (the default) is 8 x 256;
`--envs` and `--rollout-steps` override either. The defaults stay at 8 x 256
because the env count is a dynamics change -- it sets the buffer, the
minibatch and how many games one update sees -- and every curve so far was
trained there. The run prints its schedule on startup:

    velocity-flow-v2 on cuda: 64 envs x 256 steps = 16384 samples an update,
    minibatch 512, up to 128 optimizer steps (4 epochs, target_kl 0.02)

which is Turner's schedule exactly: minibatch 512, 4 epochs, `target_kl`
early stop.

**Minibatch cap.** The minibatches-per-epoch rule used to grow the minibatch
with the buffer and hold optimizer steps flat: v2 at 64 x 256 took 2048-sample
minibatches, 32 Adam steps per 16k samples (an eighth of the steps per sample
the default buffer takes), and at 256 x 256 it asked for 8192, which does not
fit an 8GiB card. `MAX_MINIBATCH = 512` (`max_minibatch` in the hyperparameter
table) bounds it. It only binds above 4,096 samples for v2 and 8,192 for the
other architectures, so the default buffer trains exactly as before; a run at
a larger buffer gets more, smaller steps than it used to, and with
`target_kl` that can also mean an earlier stop. Per-sample cost on the GPU is
flat across the range, but more steps means more epochs before `target_kl`
can stop the update, so it is a dynamics change, not a free one.

**Periodic checkpoints** are counted in PPO updates (`--checkpoint-updates`,
default 5) and land just after an update. They used to be every ~10k env
steps, which is every five updates at 8 envs but more than once an update at
64, each landing mid-rollout. The dashboard callback banks per-lane reward and
kills in arrays rather than per-lane dicts, ~1.7ms down to ~0.3ms a step at
256 envs.

## PPO bots

### Observation

One frame, 3,076 values:

    0..3        px, py, dpx, dpy
    4..1027     categorical cell values, 32x32
    1028..2051  vx of whatever occupies each cell
    2052..3075  vy of the same

There is no frame stack. Motion is read from the game rather than inferred,
because inference was failing: everything here moves less than one 4px cell per
decision — enemies 0.54–0.74, pattern bars 0.70, the player 0.80 — so the
difference between two categorical grids was usually nothing at all. Widening
the gap to two decisions helped but could not fix the shape of the problem,
since differencing conflates everything in a region and two enemies crossing
cannot keep separate velocities.

**Incoming pattern telegraphs.** Patterns come in two shapes and only one of
them used to be visible to the agent. A `tpe==0` pattern grows in place at the
coordinates it will occupy, so its warning outline lands in the grid from the
first frame. A `tpe==1` pattern slides in from outside the arena — and the cart
grows `sh` from 0 to 2 before a rect moves at all, so it spends that whole
telegraph parked off-screen where the grid cannot reach it. Measured: the player
saw a warning line at the edge for 105 frames, 26 decisions, while the
observation held an empty arena.

The cart already computes what to draw. `r.warnings` is two segments just inside
the edge that grow outward as `sh` rises, carrying both where the bar will enter
and roughly how soon. Those segments are now painted as `HAZARD_WARNING`, with
the rect's own velocity on the cells, for exactly as long as the cart draws them
(`cp.timer < 125`) — the agent gets the cue the player gets, and not a frame
more. The blind window is 105 frames down to one.

A velocity raster has neither failure. It is exact, it is per entity, and it
costs about 120us a step. Measured against what a frame difference can recover,
in cells of error when extrapolated forward:

| Frames ahead | Velocity raster | Physics forecast | 2-frame inference |
| --- | --- | --- | --- |
| 24 | exact now | 0.12 | 0.60 |
| 48 | exact now | 0.71 | 2.22 |

Empty cells hold zero, which is unambiguous because the category channels say
separately whether a cell is occupied — no epsilon needed. Velocities are
signed, so the observation box is [-1, 1] rather than the unit interval, and
the player's own `dpx, dpy` are supplied explicitly: without a second frame
there is nothing else to derive momentum from, and it matters in a game with
friction.

`closing` is now exact too. It was the change in distance between two rounded
frames; it is now the threat's real velocity minus the player's, projected onto
the line between them.

### Architectures

Both architectures read that same pair of frames, ordered older then current.
Each frame contains the player's normalized `x, y` coordinates followed by a flattened
`32 x 32` categorical map. The arena therefore has 1,024 four-pixel cells and the complete
two-frame observation contains 2,052 values. Cells identify enemies, pickups, pattern
warnings, and lethal pattern geometry. Control is keyboard-only: PPO can hold the four
directional buttons individually or in diagonal pairs; mouse teleport control is never
exposed. Loading an older 1,026-input checkpoint migrates its analogous weights to the
current-frame inputs while normally initializing the added previous-frame connections.

The default `hybrid-cnn` policy converts each frame's categorical map into eight one-hot
spatial channels, plus one channel per frame marking the player's own cell. That player
channel matters more than its size suggests: the hazard map says nothing about where the
agent is, so without it every convolution reasoned about threats with no idea what they
were near, and player position only arrived through a separate branch after the spatial
layout had been flattened away. It costs nothing in the environment, since those
coordinates are already part of the observation. Hazard channels keep their original
order and count, so a checkpoint from before this migrates by copying into the leading
channels while the player connections start fresh.

The coordinate branch also receives six derived senses: player velocity, the
nearest lethal cell's offset and distance, and how fast that distance is
closing. Every one is a function of values already in the observation, so they
cost nothing in the environment; they are supplied because the *shape* of the
function matters. Reading "a threat is three cells right and gaining two cells
per decision" off a pair of one-hot grids is a lot to ask of three
convolutions, and basic dodging is exactly that question. Pickups and pattern
telegraphs are excluded from the threat search, since neither ends an episode. Its CNN branch uses `32 -> 64 -> 64` filters and emits 128 features;
a small MLP encodes the four player coordinates into 32 features. Separate
`128 -> 64` actor and critic heads consume the combined representation. This preserves
neighbor relationships while allowing the two frames to express movement. The original
flattened policy remains available for baseline comparisons:

    python training_gui.py --architecture mlp
    python train_ppo.py --architecture mlp --name two-frame-mlp

Install the optional training stack and start the beginner curriculum:

    pip install -r requirements-train.txt
    python train_ppo.py --steps 250000

This defaults to easy difficulty with powerups and both lethal enemy types enabled, patterns
disabled, and enemy/pattern cadence at 100%. Add patterns with `--patterns`, increase
difficulty with `--difficulty normal`, isolate enemy types with `--no-normal-enemies` or
`--no-kamikaze-enemies`, adjust cadence with `--spawn-rate-percent 50..150`, or use
`--no-powerups` for a stripped-down comparison. Models are written to `artifacts/models/`
and appear in the dashboard's Models selector.

## Layout

| Path | What | Tracked |
| --- | --- | --- |
| `dodge/` | batched environment, reference observation encoder, policies, PPO session, telemetry, game window | yes |
| `tools/` | starting a fresh checkpoint | yes |
| `play.py`, `training_gui.py`, `train_ppo.py`, `watch_ppo.py`, `gradient_bot.py` | entry points | yes |
| `docs/superpowers/` | design specs and implementation plans | yes |
| `docs/paper/` | the evaluation paper; its scripts ran on the Lua runtime, removed after commit `0b83fcc` | yes |
| `rewards.json`, `game-config.json` | per-user tuning written by the dashboard | no, ignored |

The game engine itself, `dodge_native`, is installed from Taylor Turner's
repository; see [Setup](#setup).

## Licensing

The code in this repository is original work; see [LICENSE](LICENSE).

**The game is not.** `dodge` is a third-party PICO-8 cartridge, and this project
deliberately does not distribute it — no cartridge content is tracked in git or
present in its history. The game runs on Taylor Turner's Rust port, which is
installed from its own repository under its own terms.

If you intend to redistribute a cartridge, or to publish a build that embeds
one, that is between you and the cartridge's author. Two things worth knowing
before you do:

- The default cartridge is `Dodge` by Artridge, released under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Attribution, non-commercial use, and share-alike conditions apply to that
  cartridge and adaptations of it.
- A copyleft licence such as CC BY-NC-SA can reach further than the cartridge
  itself if the combined work counts as an adaptation, which is exactly why this
  repository keeps the two apart.

This note is not legal advice.
