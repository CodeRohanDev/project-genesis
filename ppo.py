"""PPO on MiniGrid, CleanRL-style single file.

Trains on a sequence of tasks, one after another, with the same network.
One task = baseline B1 (scratch). Several tasks = baseline B2 (sequential, no protection).
After each training stage every eval task is tested -> success matrix for forgetting/transfer.

    python ppo.py --tasks MiniGrid-Empty-8x8-v0 --steps 200000 --seed 1
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  registers MiniGrid envs
import numpy as np
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from torch.distributions import Categorical

EVAL_SEED = 10**6  # eval layouts come from a seed range training never resets to


def make_env(spec):
    """spec: env id, optionally with a size override, e.g. 'MiniGrid-DoorKey-8x8-v0@size=7'."""
    env_id, _, size = spec.partition("@size=")
    kwargs = {"size": int(size)} if size else {}

    def thunk():
        env = ImgObsWrapper(gym.make(env_id, **kwargs))
        env.action_space = gym.spaces.Discrete(7)  # uniform; Dynamic-Obstacles (3 actions) maps extra ones to 'left'
        return env

    return thunk


def to_tensor(obs, device):
    # (N, 7, 7, 3) uint8 grid codes -> (N, 3, 7, 7) float
    return torch.as_tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 10.0


class Agent(nn.Module):
    """Small CNN from rl-starter-files. Same obs/action space for all MiniGrid tasks.
    memory=True adds an LSTM (working memory) between the CNN and the heads."""

    def __init__(self, n_actions=7, memory=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 2), nn.ReLU(),
            nn.Flatten(), nn.Linear(64, 64), nn.Tanh(),
        )
        self.lstm = nn.LSTM(64, 64) if memory else None
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def initial_state(self, n, device):
        return None if self.lstm is None else tuple(torch.zeros(1, n, 64, device=device) for _ in range(2))

    def forward(self, x, state=None, done=None):
        """x: (T*N, 3, 7, 7) time-major. state: LSTM state for N envs. done: (T*N,) 1 where a new episode starts.
        Without memory, state/done are ignored. Returns (dist, value, new_state)."""
        h = self.body(x)
        if self.lstm is not None:
            if state is None:
                state = self.initial_state(len(x), x.device)
            n = state[0].shape[1]
            h, done = h.view(-1, n, 64), (torch.zeros(len(x), device=x.device) if done is None else done).view(-1, n)
            out = []
            for ht, dt in zip(h, done):
                state = tuple((1.0 - dt).view(1, -1, 1) * s for s in state)  # wipe memory at episode start
                o, state = self.lstm(ht[None], state)
                out.append(o[0])
            h = torch.cat(out)
        return Categorical(logits=self.actor(h)), self.critic(h).squeeze(-1), state


class RND(nn.Module):
    """Random Network Distillation curiosity: bonus = how badly the predictor imitates a fixed random net.
    Familiar views are predicted well (low bonus); new views are not (high bonus)."""

    def __init__(self, dim=64):
        super().__init__()
        net = lambda: nn.Sequential(nn.Flatten(), nn.Linear(3 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, dim))
        self.target, self.predictor = net(), net()
        self.target.requires_grad_(False)
        self.var = 1.0  # running variance of the bonus, for scale normalisation

    def bonus(self, x):
        return (self.predictor(x) - self.target(x)).pow(2).mean(-1)


@torch.no_grad()
def evaluate(agent, env_id, episodes, device):
    env = make_env(env_id)()
    wins = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + ep)
        done, state = False, agent.initial_state(1, device)
        while not done:
            dist, _, state = agent(to_tensor(obs[None], device), state)
            obs, reward, term, trunc, _ = env.step(dist.sample().item())
            done = term or trunc
        wins += reward > 0
    env.close()
    return wins / episodes


def train_task(agent, opt, env_id, args, device, seed, rnd=None, rnd_opt=None, retain=None, life=None):
    """retain: optional fn() -> loss that protects old skills; added to every PPO minibatch loss.
    life: optional boundary-blind learner state (Life), updated after every rollout."""
    specs = env_id.split("|")  # 'A|B|C' = mixture: env i runs specs[i % len(specs)]
    envs = gym.vector.SyncVectorEnv(
        [make_env(specs[i % len(specs)]) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    T, N = args.num_steps, args.num_envs
    obs_buf = torch.zeros((T, N, 3, 7, 7), device=device)
    act_buf = torch.zeros((T, N), device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)

    obs, _ = envs.reset(seed=seed)
    next_obs = to_tensor(obs, device)
    next_done = torch.zeros(N, device=device)
    state = agent.initial_state(N, device)
    recent, curve, entropy, steps_to_80, step = [], [], [], None, 0

    while step < args.steps:
        ent_sum, outcomes = 0.0, []
        init_state = None if state is None else tuple(s.clone() for s in state)
        for t in range(T):
            obs_buf[t], done_buf[t] = next_obs, next_done
            with torch.no_grad():
                dist, value, state = agent(next_obs, state, next_done)
                action = dist.sample()
                ent_sum += dist.entropy().mean().item()
            act_buf[t], logp_buf[t], val_buf[t] = action, dist.log_prob(action), value
            obs, reward, term, trunc, _ = envs.step(action.cpu().numpy())
            done = term | trunc
            rew_buf[t] = torch.as_tensor(reward, device=device)
            next_obs = to_tensor(obs, device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)
            outcomes.extend((reward[done] > 0).tolist())
        recent.extend(outcomes)  # per-task, for measurement only; the agent never sees task identity
        step += T * N
        if life is not None:
            life.update(outcomes, obs_buf.flatten(0, 1))

        if rnd is not None:
            # ponytail: single critic on ext + int reward (episodic); separate int value head if this proves unstable
            with torch.no_grad():
                b = rnd.bonus(obs_buf.flatten(0, 1)).view(T, N)
            rnd.var = 0.99 * rnd.var + 0.01 * b.var().item()
            rew_buf += args.rnd_coef * b / (rnd.var**0.5 + 1e-8)
        recent = recent[-100:]
        rate = float(np.mean(recent)) if recent else 0.0
        curve.append((step, rate))
        entropy.append(ent_sum / T)  # policy entropy during rollout; max for 7 actions = ln 7 ~ 1.95
        if steps_to_80 is None and len(recent) >= 100 and rate >= 0.8:
            steps_to_80 = step

        # GAE. Timeouts treated as terminal (CleanRL convention).
        with torch.no_grad():
            _, next_value, _ = agent(next_obs, state, next_done)
            adv = torch.zeros_like(rew_buf)
            last = 0
            for t in reversed(range(T)):
                nonterm = 1.0 - (next_done if t == T - 1 else done_buf[t + 1])
                nv = next_value if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * nonterm - val_buf[t]
                adv[t] = last = delta + args.gamma * args.gae_lambda * nonterm * last
            ret = adv + val_buf

        b_obs, b_act, b_logp = obs_buf.flatten(0, 1), act_buf.flatten(), logp_buf.flatten()
        b_adv, b_ret, b_done = adv.flatten(), ret.flatten(), done_buf.flatten()
        mb = T * N // args.minibatches
        for _ in range(args.epochs):
            if init_state is None:  # feedforward: random samples
                batches = [(idx, None) for idx in torch.randperm(T * N, device=device).split(mb)]
            else:  # memory: whole env sequences, replayed from the rollout's starting LSTM state
                batches = [((torch.arange(T, device=device)[:, None] * N + e[None]).flatten(),
                            tuple(s[:, e] for s in init_state))
                           for e in torch.randperm(N, device=device).split(max(1, N // args.minibatches))]
            for idx, st in batches:
                dist, value, _ = agent(b_obs[idx], st, b_done[idx] if st is not None else None)
                ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                a = b_adv[idx]
                if not args.no_adv_norm:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                v_loss = 0.5 * (value - b_ret[idx]).pow(2).mean()
                loss = pg + args.vf_coef * v_loss - args.ent_coef * dist.entropy().mean()
                if retain is not None:
                    loss = loss + retain()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
                if rnd is not None:
                    rnd_loss = rnd.bonus(b_obs[idx]).mean()
                    rnd_opt.zero_grad()
                    rnd_loss.backward()
                    rnd_opt.step()

        if len(curve) % 20 == 0:
            print(f"  {env_id} step {step:>8d}  train success(last100) {rate:.0%}", flush=True)

    envs.close()
    # last rollout's observations: a sample of what this task looks like, for retention methods
    return {"curve": curve, "entropy": entropy, "steps_to_80": steps_to_80}, obs_buf.flatten(0, 1).clone()


class Life:
    """Boundary-blind continual learner. Uses only what the agent itself can know (its own success),
    persists across task switches, and is never told when the task changes.

    - competence-gated consolidation: while its own success rate >= 80%, copy observations + current action
      probabilities into a fixed-size reservoir (uniform sample over all mastered history; memory does not grow).
    - replay-distillation from the reservoir on every PPO minibatch (same KL loss as Retention 'distill').
    - surprise: success collapses (< 30%) after mastery -> "the world changed" -> optionally reset heads + optimizer.
    """

    def __init__(self, agent, opt, args, batch, device):
        self.agent, self.opt, self.args, self.batch = agent, opt, args, batch
        self.recent, self.mastered, self.step, self.events = [], False, 0, []
        self.cap, self.seen, self.size = args.reservoir, 0, 0
        self.obs = torch.zeros((self.cap, 3, 7, 7), device=device)
        self.logits = torch.zeros((self.cap, 7), device=device)

    def update(self, outcomes, obs):
        self.step += len(obs)
        self.recent = (self.recent + outcomes)[-100:]
        rate = float(np.mean(self.recent)) if self.recent else 0.0
        if len(self.recent) >= 100 and rate >= 0.8:
            self.mastered = True
            if self.cap:
                self.consolidate(obs)
        elif self.mastered and rate < 0.3:
            self.mastered = False
            self.events.append((self.step, "surprise"))
            if self.args.surprise_reset:
                self.agent.actor.reset_parameters()
                self.agent.critic.reset_parameters()
                self.opt.state.clear()

    @torch.no_grad()
    def consolidate(self, obs, k=256):
        x = obs[torch.randperm(len(obs), device=obs.device)[:k]]
        dist, _, _ = self.agent(x)
        for xi, li in zip(x, dist.logits):
            if self.size < self.cap:
                slot = self.size
                self.size += 1
            else:  # reservoir sampling: every item ever offered has equal chance to be kept
                slot = int(torch.randint(self.seen + 1, ()).item())
            self.seen += 1
            if slot < self.cap:
                self.obs[slot], self.logits[slot] = xi, li

    def __call__(self):
        if self.size == 0:
            return 0.0
        idx = torch.randint(self.size, (self.batch,), device=self.obs.device)
        dist, _, _ = self.agent(self.obs[idx])
        old = self.logits[idx]
        return self.args.retain_coef * (old.exp() * (old - dist.logits)).sum(-1).mean()


class Retention:
    """Protect old skills while learning new ones. Needs task boundaries (a known shortcut).

    distill: keep old-task observations + the old policy's action probabilities on them; while training a new
             task, penalise KL(old || current) on a random batch of them (rehearsal / "sleep replay").
    ewc:     Elastic Weight Consolidation (Kirkpatrick 2017), online version: weights important for old
             policies (diag Fisher) are pulled back to their values at the end of the previous task.
    """

    def __init__(self, agent, mode, coef, batch):
        self.agent, self.mode, self.coef, self.batch = agent, mode, coef, batch
        self.obs, self.logp = [], []
        self.fisher = self.anchor = None

    def remember(self, obs):
        """Call at the end of a task, before any task-switch reset."""
        with torch.no_grad():
            dist, _, _ = self.agent(obs)
        if self.mode == "distill":
            self.obs.append(obs)
            self.logp.append(dist.logits)  # Categorical stores normalised log-probs
        elif self.mode == "ewc":
            f = [torch.zeros_like(p) for p in self.agent.parameters()]
            n = min(256, len(obs))
            for x in obs[torch.randperm(len(obs))[:n]]:
                d, _, _ = self.agent(x[None])
                self.agent.zero_grad()
                d.log_prob(d.sample()).sum().backward()
                for fi, p in zip(f, self.agent.parameters()):
                    if p.grad is not None:
                        fi += p.grad.detach() ** 2 / n
            self.agent.zero_grad()
            self.fisher = f if self.fisher is None else [a + b for a, b in zip(self.fisher, f)]
            self.anchor = [p.detach().clone() for p in self.agent.parameters()]

    def __call__(self):
        if self.mode == "distill" and self.obs:
            obs, logp = torch.cat(self.obs), torch.cat(self.logp)
            idx = torch.randint(len(obs), (self.batch,))
            dist, _, _ = self.agent(obs[idx])
            old = logp[idx]
            return self.coef * (old.exp() * (old - dist.logits)).sum(-1).mean()
        if self.mode == "ewc" and self.fisher is not None:
            return self.coef / 2 * sum(
                (f * (p - a) ** 2).sum() for f, p, a in zip(self.fisher, self.agent.parameters(), self.anchor)
            )
        return 0.0


def on_task_switch(agent, opt, mode, lr):
    """Plasticity interventions applied before each new task (not before the first). Returns the optimizer."""
    if mode in ("adam", "heads"):
        if mode == "heads":
            agent.actor.reset_parameters()
            agent.critic.reset_parameters()
        return torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-8)
    if mode == "shrink":  # shrink & perturb (Ash & Adams 2020): blend weights with a fresh random init
        fresh = Agent(memory=agent.lstm is not None).to(next(agent.parameters()).device)
        with torch.no_grad():
            for p, q in zip(agent.parameters(), fresh.parameters()):
                p.mul_(0.8).add_(0.2 * q)
        return torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-8)
    return opt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True, help="comma-separated, trained in order")
    p.add_argument("--eval-tasks", default=None, help="default: same as --tasks")
    p.add_argument("--steps", type=int, default=500_000, help="env steps per task")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--name", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.05)  # 0.01 collapsed early, see results/stage1_b1.md
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--no-adv-norm", action="store_true", help="skip per-minibatch advantage normalisation")
    p.add_argument("--task-switch", default="none", choices=["none", "adam", "heads", "shrink"],
                   help="adam: fresh optimizer; heads: + reinit actor/critic heads; shrink: shrink & perturb + fresh optimizer")
    p.add_argument("--memory", action="store_true", help="LSTM working memory")
    p.add_argument("--retain", default="none", choices=["none", "distill", "ewc"])
    p.add_argument("--retain-coef", type=float, default=1.0)
    p.add_argument("--blind", action="store_true", help="no task-boundary signal; use Life (self-detected change)")
    p.add_argument("--reservoir", type=int, default=0, help="blind mode: replay reservoir size, 0 = no replay")
    p.add_argument("--surprise-reset", action="store_true", help="blind mode: reset heads when own success collapses")
    p.add_argument("--rnd-coef", type=float, default=0.0, help="curiosity bonus weight; 0 = off")
    args = p.parse_args()

    tasks = args.tasks.split(",")
    eval_tasks = args.eval_tasks.split(",") if args.eval_tasks else tasks
    name = args.name or "_".join(t.removeprefix("MiniGrid-") for t in tasks)
    out = Path("results/runs") / f"{name}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    agent = Agent(memory=args.memory).to(device)  # random init: the "newborn" starting point
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-8)
    rnd = RND().to(device) if args.rnd_coef > 0 else None
    rnd_opt = torch.optim.Adam(rnd.predictor.parameters(), lr=args.lr) if rnd else None
    mb = args.num_envs * args.num_steps // args.minibatches
    retain = Retention(agent, args.retain, args.retain_coef, mb) if args.retain != "none" else None
    life = None
    if args.blind:
        assert args.task_switch == "none" and args.retain == "none", "blind mode: no boundary-based interventions"
        life = retain = Life(agent, opt, args, mb, device)

    start = time.time()
    result = {"config": vars(args), "tasks": tasks, "eval_tasks": eval_tasks, "train": [], "matrix": []}
    for i, env_id in enumerate(tasks):
        print(f"[{name} seed {args.seed}] training on {env_id}", flush=True)
        if i > 0:
            if retain is not None and life is None:  # boundary-given retention only
                retain.remember(last_obs)
            opt = on_task_switch(agent, opt, args.task_switch, args.lr)
        res, last_obs = train_task(agent, opt, env_id, args, device, args.seed * 1000 + i, rnd, rnd_opt, retain, life)
        result["train"].append(res)
        row = [evaluate(agent, e, args.eval_episodes, device) for e in eval_tasks]
        result["matrix"].append(row)
        print("  eval after stage:", {e.removeprefix("MiniGrid-"): f"{s:.0%}" for e, s in zip(eval_tasks, row)})
    if life:
        result["life_events"], result["reservoir_size"] = life.events, life.size
    result["wall_seconds"] = time.time() - start
    out.write_text(json.dumps(result))
    torch.save(agent.state_dict(), out.with_suffix(".pt"))
    print(f"saved {out}  ({result['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
