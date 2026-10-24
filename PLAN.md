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

## Overnight plan (2026-10-23, ~6h, autonomous, Rohan asleep)
1. Blind continual learning: no task-boundary signal. Fixed-size reservoir replay (competence-gated),
   self-detected task change (own success collapses) -> heads reset. Compare to boundary-given distill and joint.
2. Longer sequence: 6+ tasks incl. new kinds and revisits, using best method from 1.
3. Write results/overnight.md, commit + push to private repo.

## Experiment 2 plan: goals + composition (DRAFT, 2026-10-23)
Capability: understand a new goal and combine known skills to reach it (where everything scored 0%: KeyCorridor).

Key observation: our agent only sees the image. KeyCorridor's goal ("pick up the purple ball") is in the mission
text, which the agent never received. So it could not know what to do. First fix: give the agent its goal.

Goal input = mission text split into words, each word a learned embedding from random init (no pretrained language).
Words get meaning only from reward. This is the first step of Phase 5 (grounded language).
Environments: BabyAI (ships with minigrid): GoToObj/GoToLocal, Pickup, Unlock/OpenDoor, UnlockPickup, KeyCorridor.

- 2a Word grounding: GoToLocal. Hold out some colour+object pairs (e.g. "red ball") from training while "red key"
  and "blue ball" are trained. Test the held-out pairs. Composition = held-out pairs ~ trained pairs.
  Falsified if held-out pairs stay near chance while trained pairs are high (memorised phrases).
- 2b Skill composition: train go-to, pick-up, unlock as separate tasks; test UnlockPickup / KeyCorridor never trained.
  Compare with an agent trained directly on them (reference) and the no-goal agent (0% now).
- 2c Put 2a/2b inside the blind continual learner (Life) so words and skills are learned over a sequence.
Baselines: no-goal CNN; goal agent trained on everything incl. held-out (upper reference). 10 seeds, tuning seeds separate.

## Experiment 3 plan: world model + planning (DRAFT, 2026-10-24)
Gap found in 2b: told "pick up the box", the agent never infers "first get the key, open the door". Learned skills
made the task easy to LEARN, but nothing lets the agent work out the chain by itself. That needs a model of how
the world responds to actions, and a way to search with it.

- 3a World model (no planning yet). Learn to predict next observation, reward and episode end from (observation,
  action), trained on experience the existing agents already generate. Test on held-out episodes:
  one-step accuracy, accuracy several steps ahead, and targeted causal probes:
  "toggle a locked door while carrying the right key -> opens; without key -> stays shut",
  "pick up when facing an object -> carrying it". Baseline: "nothing changes" predictor.
  Falsified if the model is no better than the baseline on the causal probes.
- 3b Planning with the model. Before acting, imagine a few action sequences (policy-guided search) and pick the one
  the model predicts leads to reward. Test zero-shot on UnlockPickup / KeyCorridor and on tasks where one wrong step
  is fatal (Lava). Compare: policy alone vs policy + planning, same agent.
- 3c Learning what skills achieve (effects of whole skills, e.g. "after unlock routine: door open"), then planning
  over skills instead of single steps. Only if 3b shows single-step planning is too short-sighted.

All learned from random init, CPU-sized (small MLP/CNN), data collected by our own agents.

### 3b revised design (2026-10-24, after 3a results; needs Rohan's go)
3a showed: the world model is accurate for many steps in familiar worlds, weaker in new layouts. Step-by-step search
over 20-40 actions with sparse success is unlikely to work. Plan over SKILLS instead (child-like: "get key, open
door, then take the box"):
- Skills = the goal-conditioned policy with a mission: "open the door", "pick up the key", "pick up the <c> box"...
  (already learned in 2b; no new training).
- Planner: for each short sequence of 1-3 candidate missions, imagine running each skill policy for up to N steps
  inside the world model, starting from the current view. Success test in imagination: the target object appears in
  the agent's own cell (holding it). Execute the first skill of the best sequence, then replan.
- Honest setup: retrain the world model on component-only data (Unlock + Pickup envs, B_comp agents), so neither the
  policy nor the model has ever seen UnlockPickup.
- Test UnlockPickup zero-shot (policy alone: 0%). Controls: random skill order; hand-written correct order
  (upper bound: shows whether failures come from planning or from skill execution).
- Falsified if the planner is no better than random skill order.

## Paper 1 plan (decided 2026-10-24): IEEE IJCNN 2027, deadline 31 Jan 2027 (backup: IEEE TCDS journal)
Claim: an agent never told when the task changes detects it itself, keeps a fixed-size competence-gated replay
memory, learns new tasks and keeps old ones, matches all-at-once training and beats CLEAR (8 stages, 6 task kinds).
Status:
1. Figures (paper/figs, figures.py) - DONE
2. Published baseline CLEAR - DONE (results/baselines.md: +5.0 on 3 tasks, +25.9 on 8 stages)
3. Second environment - DONE (results/minatar.md: blind > CLEAR > none 10/10 seeds; near joint, not equal; late-game plasticity loss)
4. Related work - draft 1 in paper/related_work.md (26 refs, each verified online)
5. Writing (IEEE format, 6 pages + up to 4 paid pages)
6. Review, fixes, submit
Venue facts (searched 2026-10-24): IJCNN 2027 Cape Town 14-18 Jun 2027, notification 15 Mar, final + fee 12 Apr,
extra pages $100 each; TCDS hybrid: no fee unless open access ($2,800), overlength $200/page beyond 10;
IEEE Access APC $2,160; ICDL 2027 Campinas 21-24 Sep 2027, call for papers not yet out.

## Next research track (after 3b): child-like language, faster
Rohan's goal (2026-10-24): the system should learn full English and Hindi the way a child does, but faster and more
easily. Not solved anywhere; must be designed and measured (words/examples needed vs a child). Candidate parts:
reading from scratch (small LM, child-level text), grounding words in the gridworld agent, a teacher curriculum.

## Decisions (made by Claude, Rohan delegated)
1. Environment: MiniGrid. Standard, cheap, comparable with published results.
2. Stage 2 order: decided by Stage 1 evidence, not guessed now.
   - Baselines fail to learn DoorKey/Lava at all: exploration problem, so curiosity (RND) first.
   - Baselines learn but forget: retention problem, so modular network first.
   - Memory (episodic) after one of the above works.
3. Task list: no extra tasks yet. Add only if Stage 1 shows tasks are too easy.

## Not in this experiment
Language, voice, companion, computer control. Those are later phases.
