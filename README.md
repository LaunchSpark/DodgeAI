# DodgeAI

A toolchain for running the PICO-8 game `dodge` locally under Python, and for
training a reinforcement-learning agent to play it.

This repository contains **no game content**. It ships the tooling: a cartridge
extractor, a partial PICO-8 runtime that hosts the cart's real Lua under LuaJIT,
a Gymnasium environment, and a training dashboard. You supply the cartridge.

## Setup

    pip install -r requirements.txt

Download Artridge's public cartridge from the
[Lexaloffle release](https://www.lexaloffle.com/bbs/?pid=66443#p) and place it
at the repository root as `dodge.p8.png`. Everything else is generated from it.

Extract the cartridge sources:

    python tools/extract_cart.py

This writes `dodge.lua` (the game's Lua source) and `dodge.p8` (the full
cartridge in PICO-8's text format). Both are generated artefacts and are
git-ignored — re-running the extractor reproduces them byte-for-byte, so there
is no need to keep them under version control.

`dodge.p8.png`, `dodge.lua` and `dodge.p8` are all ignored by git. Keep it that way
unless you have the rights to redistribute the cartridge you are using; see
[Licensing](#licensing).

Without those files the tooling still imports and its own tests still pass —
the tests that need a cartridge skip themselves.

## Play it

    python play.py

Arrows or WASD to move, Z/C and X/V for the O and X buttons, Escape to quit.

## Training dashboard

Start PPO training in the Dodge dashboard:

    python training_gui.py

Training runs on a background thread so the window stays responsive. Pause/Resume controls
rollout collection, Reset creates a fresh policy, Models saves or loads real
Stable-Baselines3 `.zip` checkpoints, and Watch Agent opens the current policy in a second
rendered game window. Each watch launch saves an immutable snapshot, chooses a new random
episode seed, and labels the game window with its exact PPO update and seed. The active
model name is shown in Learning Statistics and carried into the watch snapshot
filename and window title. Saving names the active in-memory policy; loading changes the
active policy to the selected model.

Every chart spans the whole run rather than a recent window, so improvement stays visible
as training goes on. Each draws the raw per-episode values faintly beneath a smoothed
trend line, keeps its y-axis anchored to the run's full range, and reports the change
since the run started. The Progress panel states the same thing in numbers. History is
held in memory only, so restarting the dashboard starts a fresh curve.

Use `python training_gui.py --screenshot out.png` to render one frame of sample telemetry
without training — useful for working on the dashboard itself.

### Reward weights

The **Rewards** button opens sliders over the reward function. They write
`rewards.json`, which is equally editable by hand:

    {
      "survival_per_frame": 0.01,
      "center_bonus": 0.5,
      "death_penalty": 1.0,
      "score_weight": 1.0,
      "uncontrolled_score_weight": 0.0
    }

Environments re-read the file when an episode starts, so a change reaches every
training process from the next episode without restarting anything. That
indirection is deliberate: environments run in separate processes, so a slider
cannot reach them through memory. A missing or malformed file falls back to the
defaults rather than stopping a run.

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

Measured on a GTX 1070, four envs, `hybrid-cnn`, 512 samples per update:

| | Policy update | Full cycle (collect + learn) |
| --- | --- | --- |
| CPU | 7.20s | 7.06s |
| GPU | 1.40s | **1.90s** |
| | **5.1x** | **3.7x** |

`auto` only reaches for a GPU where it measurably helps. The MLP update is
unchanged on a GPU (0.686s against 0.693s) because its layers are too small to
cover transfer and launch overhead, so `auto` keeps `mlp` on the CPU. An
explicit `--device cuda` is always honoured.

Stable-Baselines3 warns that a non-CNN policy on a GPU may be slower. It reads
the policy class, which is `ActorCriticPolicy` for both architectures here, so
it cannot see that `hybrid-cnn`'s features extractor is convolutional. That
warning is suppressed for this project's models, on the strength of the numbers
above.

With the update on a GPU the balance flips: simulation becomes roughly half of
the cycle rather than a fifth, so it is the next thing worth optimising.

### Parallel environments

Training defaults to four games stepped in parallel processes:

    python training_gui.py --envs 4 --rollout-steps 128

PPO's buffer holds `envs * rollout-steps` samples per update, so those defaults keep the
update size at 512 while gathering it from four independently seeded games. Each game
advances only 128 decisions before the policy learns, which keeps openings varied.

Environments run in separate processes (`SubprocVecEnv`). This matters because every Dodge
environment hosts its own LuaJIT machine: stepping them in one process would diversify
experience without adding throughput. Measured on a 4-core machine, four environments run
about 2.5x the steps per second of one — short of 4x, since process overhead and the
policy update are not parallelised.

Raising `--envs` past the core count tends to cost more than it returns. The dashboard's
steps/second readout is the number to watch when trying other values.

## PPO bots

Both architectures observe the previous and current decision frames, ordered previous
then current. The snapshots are separated by `action_repeat` game frames, giving the
policy a clearer movement signal than two consecutive 1/60-second frames.
Each frame contains the player's normalized `x, y` coordinates followed by a flattened
`32 x 32` categorical map. The arena therefore has 1,024 four-pixel cells and the complete
two-frame observation contains 2,052 values. Cells identify enemies, pickups, pattern
warnings, and lethal pattern geometry. Control is keyboard-only: PPO can hold the four
directional buttons individually or in diagonal pairs; mouse teleport control is never
exposed. Loading an older 1,026-input checkpoint migrates its analogous weights to the
current-frame inputs while normally initializing the added previous-frame connections.

The default `hybrid-cnn` policy converts each frame's categorical map into eight one-hot
spatial channels. Its CNN branch uses `32 -> 64 -> 64` filters and emits 128 features;
a small MLP encodes the four player coordinates into 32 features. Separate
`128 -> 64` actor and critic heads consume the combined representation. This preserves
neighbor relationships while allowing the two frames to express movement. The original
flattened policy remains available for baseline comparisons:

    python training_gui.py --architecture mlp
    python train_ppo.py --architecture mlp --name two-frame-mlp

Install the optional training stack and start the beginner curriculum:

    pip install -r requirements-train.txt
    python train_ppo.py --steps 250000

This defaults to easy difficulty with patterns and powerups disabled. Enable the full game
incrementally with `--powerups`, `--patterns`, or `--difficulty normal`. Models are written
to `artifacts/models/` and appear in the dashboard's Models selector.

## Layout

| Path | What | Tracked |
| --- | --- | --- |
| `pico8/` | partial PICO-8 runtime in Python, hosting the cart's Lua under LuaJIT | yes |
| `dodge/` | Gymnasium environment, observations, PPO session, telemetry | yes |
| `tools/` | the cartridge extractor | yes |
| `play.py`, `training_gui.py`, `train_ppo.py`, `watch_ppo.py` | entry points | yes |
| `docs/superpowers/` | design specs and implementation plans | yes |
| `dodge.p8.png` | Artridge's public PICO-8 cartridge — supply this yourself | no, ignored |
| `dodge.lua`, `dodge.p8` | generated by the extractor | no, ignored |

## Licensing

The code in this repository is original work; see [LICENSE](LICENSE).

**The game is not.** `dodge` is a third-party PICO-8 cartridge, and this project
deliberately does not distribute it — no cartridge content is tracked in git or
present in its history. You bring your own `dodge.p8.png`, and everything derived
from it stays local and git-ignored.

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
