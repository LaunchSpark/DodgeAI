"""Write a freshly initialised policy checkpoint.

The dashboard's Models panel lists the checkpoints on disk, and loading one
switches the running session to that architecture. So an architecture with no
saved checkpoint cannot be reached from the dashboard at all. This writes an
untrained starting point, and is what both the Training panel's "Start a new
model" and tools/new_policy.py call.
"""

from pathlib import Path

from .critic import CRITIC_ARCHITECTURES
from .policies import ARCHITECTURES, V3_ACTOR_FLAGS, policy_class, policy_kwargs

# Offered first: the architecture a new run is most likely to want.
NEW_MODEL_ARCHITECTURES = ("velocity-flow-v3",) + tuple(
    name for name in ARCHITECTURES if name != "velocity-flow-v3")

# What the panel shows for each actor flag; all default on.
ACTOR_FLAG_TEXT = {
    "carry_velocity": ("Carry velocity",
                       "Read each action where the player's momentum takes it"),
    "graduated_holds": ("Graduated holds",
                        "Far slices probe 4 to 30px out instead of all at 10px"),
    "centred_logits": ("Centred logits",
                       "One danger scale for every state, so calm states stay uniform"),
}
assert set(ACTOR_FLAG_TEXT) == set(V3_ACTOR_FLAGS)


def actor_flag_choices():
    """Per architecture, the flags a new model can be started with."""
    return {"velocity-flow-v3": [
        {"name": name, "label": ACTOR_FLAG_TEXT[name][0],
         "hint": ACTOR_FLAG_TEXT[name][1], "default": True}
        for name in V3_ACTOR_FLAGS]}


def write_new_policy(architecture, models_dir, *, name=None, seed=0, **actor_flags):
    """Save an untrained `architecture` to models_dir/<name>.zip; its path.

    Refuses to overwrite: a name in use is a run someone may still want.
    """
    # Imported here: the environment and SB3 are heavy, and the dashboard
    # imports this module only to list the choices.
    from stable_baselines3 import PPO

    from .env import DodgeVecEnv
    from .training import hyperparameters, minibatch_size

    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    try:
        kwargs = policy_kwargs(architecture, **actor_flags)
    except TypeError as error:
        raise ValueError(str(error)) from None
    path = Path(models_dir) / f"{name or architecture}.zip"
    if path.exists():
        raise ValueError(f"{path.name} already exists; pick another name")

    # A throwaway environment: only its observation and action spaces are
    # recorded, and the session rebinds the real ones when it loads this.
    env = DodgeVecEnv(n_envs=1, max_episode_frames=20,
                      critic_features=architecture in CRITIC_ARCHITECTURES)
    # Seeded with the architecture's own schedule. These used to be bare
    # defaults, and because loading inherited whatever the checkpoint carried,
    # a seed's throwaway numbers became the schedule of every run started from
    # it. Loading now imposes the table too, but a checkpoint should not lie
    # about what it is.
    settings = hyperparameters(architecture)
    try:
        model = PPO(
            policy_class(architecture), env,
            n_steps=64, batch_size=minibatch_size(64, architecture),
            n_epochs=settings["n_epochs"], gamma=settings["gamma"],
            vf_coef=settings["vf_coef"], target_kl=settings["target_kl"],
            ent_coef=settings["ent_coef"], max_grad_norm=settings["max_grad_norm"],
            seed=seed, device="cpu",
            policy_kwargs=kwargs,
        )
    finally:
        env.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path.with_suffix(""))
    return path
