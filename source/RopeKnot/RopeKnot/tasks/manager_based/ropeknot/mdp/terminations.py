import torch


def done(env):
    done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if hasattr(env, "last_rewards"):
        done = env.last_rewards > 0.95
    return done
