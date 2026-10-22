# Project Genesis: Experiment 1 Plan

Status: DRAFT, awaiting Rohan's approval. No code written yet.

## AGI capability being tested
Continual learning + transfer (from the AGI list: "learn different tasks", "transfer to unfamiliar situations", "retain earlier abilities").

## Research question
Can a randomly initialised agent learn tasks one after another, keep old skills, and reuse them on new tasks faster than learning from scratch?

## Hypotheses
- H0 (null): our added mechanism does not beat plain baselines on forgetting or transfer.
- H1: mechanism X (added in Stage 2) reduces forgetting and speeds up learning on new tasks, beyond statistical noise.

Falsified if: H1 gain is within the confidence interval of the baselines over 5+ seeds.

## Environment
- MiniGrid (`minigrid` + `gymnasium` pip packages). Already built, no custom env needed.
- Small grid, partial view, discrete actions (turn, move, pickup, toggle).
- Task sequence (train in order):
  1. Empty / GoTo: reach the goal
  2. DoorKey: pick up key, open door, reach goal
  3. LavaCrossing: reach goal avoiding lava
  4. (optional) Unlock / PutNear
- Held-out test (never trained): new grid sizes, new random layouts, one unseen task variant.

## Agent
- Random init, no pretrained weights, no language.
- Small CNN/MLP policy, runs on CPU (Ryzen 9600X is enough) or GTX 1050 Ti.
- Framework: PyTorch. RL algo: PPO (CleanRL-style single file, reuse not rewrite).

## Stages
**Stage 0, setup (1 session):** check GPU/VRAM/CUDA, create venv, install deps, seed + logging + checkpoint utilities, one smoke test.

**Stage 1, baselines (measurement tool):**
- B1: PPO trained on each task from scratch (reference for sample efficiency)
- B2: PPO trained sequentially, no protection (expect forgetting)
- B3: PPO sequential + replay buffer
- B4: PPO sequential + EWC
Goal: prove the evaluation works and show how bad forgetting really is. If B2 does not forget, the experiment is too easy; make it harder.

**Stage 2, one mechanism at a time:**
- Candidate 1: curiosity (RND) for exploration
- Candidate 2: modular / expandable network (new module per task)
- Candidate 3: episodic memory feeding the policy
Add one, measure, then decide next. Never stack several untested.

## Metrics (all over 5+ seeds, mean ± 95% CI)
- Success rate per task after each training stage (matrix: task x time)
- Forgetting: drop in old-task success after learning new tasks
- Forward transfer: steps to reach 80% success on new task vs scratch
- Held-out generalisation: success on unseen layouts/sizes
- Memorisation check: same policy on fixed vs randomised layouts; big gap = memorising, not learning

## Rules
- Fixed seeds, config saved with every run, checkpoints recoverable
- Results logged as they are, including failures, in `results/`
- Each stage ends with a short write-up: what we expected, what happened, next step

## Revision 1 (2026-10-22, after Stage 1 B1, see results/stage1_b1.md)
8x8 DoorKey/Lava not learned reliably by plain PPO, so forgetting cannot be measured on them.
Two questions split:
- Retention (continual learning, main track): small versions of same skills,
  Empty-5x5 -> DoorKey-5x5 -> LavaGapS5. Run B1-B4 here.
- Exploration (side track, later): 8x8 DoorKey / LavaCrossing, retest curiosity there.
Baseline default: entropy coef 0.05.

## Decisions (made by Claude, Rohan delegated)
1. Environment: MiniGrid. Standard, cheap, comparable with published results.
2. Stage 2 order: decided by Stage 1 evidence, not guessed now.
   - Baselines fail to learn DoorKey/Lava at all: exploration problem, so curiosity (RND) first.
   - Baselines learn but forget: retention problem, so modular network first.
   - Memory (episodic) after one of the above works.
3. Task list: no extra tasks yet. Add only if Stage 1 shows tasks are too easy.

## Not in this experiment
Language, voice, companion, computer control. Those are later phases.
