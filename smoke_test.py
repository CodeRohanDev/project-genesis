"""Stage 0 smoke test: env + GPU work, random agent gives a near-zero baseline."""
import gymnasium as gym
import minigrid  # noqa: F401  registers MiniGrid envs
import torch

TASKS = ["MiniGrid-Empty-8x8-v0", "MiniGrid-DoorKey-8x8-v0", "MiniGrid-LavaCrossingS9N1-v0"]
EPISODES = 100


def random_success_rate(env_id, seed=0):
    env = gym.make(env_id)
    env.action_space.seed(seed)
    wins = 0
    for ep in range(EPISODES):
        env.reset(seed=seed + ep)
        done = False
        while not done:
            _, reward, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
        wins += reward > 0
    env.close()
    return wins / EPISODES


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"
    x = torch.randn(256, 256, device="cuda")
    assert torch.isfinite(x @ x).all()
    print(f"GPU ok: {torch.cuda.get_device_name(0)}")

    obs, _ = gym.make(TASKS[0]).reset(seed=0)
    assert obs["image"].shape == (7, 7, 3), obs["image"].shape
    print(f"obs ok: partial view {obs['image'].shape}")

    for t in TASKS:
        print(f"{t:32s} random success: {random_success_rate(t):.0%}")
