# Shared VelocityFlow v2 checkpoints

These selected saves are tracked in Git so they travel with the repository.
Training logs, runtime checkpoints, watch snapshots, and other model files stay
ignored. Each checkpoint is about 1.6 MB and is stored directly in Git.

Initial inventory (September 25, 2026):

| Checkpoint | Training steps | Completed updates |
| --- | ---: | ---: |
| `can-reach-100.zip` | 11,777,424 | 5,691 |
| `velocity-flow-v2.zip` | 18,257,744 | 8,855 |
| `velocity-flow-v2-latest.zip` | 43,235,992 | 21,051 |

All three use the `velocity-flow-v2` architecture. These counters describe
training progress, not a performance ranking. All three were checked by loading
on CPU and running ten game decisions; no comparative benchmark was performed.
The latest file is an autosave and can change when training resumes. Save a
separately named checkpoint in the dashboard to preserve a favorite, then add
its exact path to the allowlist in `.gitignore` and stage it with `git add`.

## Watch a shared model

From the repository root:

```sh
python -m pip install -r requirements-train.txt
```

Download the cartridge from the [original Dodge release](https://www.lexaloffle.com/bbs/?pid=66443#p)
and place it at the repository root as `dodge.p8.png`, as described in the
[project setup](../../README.md#setup). The game cartridge is supplied separately.

This example uses the latest checkpoint and the local game settings present
when this inventory was created: hard difficulty, patterns and powerups on,
both enemy types on, 100% spawn rate, and one decision per frame. These settings
are a snapshot of the current configuration, not verified training settings for
every checkpoint.

```sh
python watch_ppo.py artifacts/models/velocity-flow-v2-latest.zip --difficulty hard --patterns --powerups --normal-enemies --kamikaze-enemies --spawn-rate-percent 100 --action-repeat 1 --seed 42
```

Replace the checkpoint path to watch either of the other saves. Use `--no-field`
to hide the danger-field overlay. A CPU is sufficient for watching.

To resume training, run `python training_gui.py` and load the desired checkpoint
from Models. Game settings and reward weights are local preferences and are not
bundled in these model files.
