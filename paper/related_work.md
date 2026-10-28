# Related work (paper step 4, draft 2, 2026-10-25)

Every reference below was looked up on 2026-10-24 (venue, year, pages from the official proceedings / journal page or
arXiv listing). Claims about each paper are taken only from its abstract / official summary, not from memory.
Unverified details are marked. Numbers [n] match the reference list at the end (IEEE style).

## Draft text (for the paper, ~0.6 column)

**Continual reinforcement learning.** Neural networks trained on a sequence of tasks forget earlier ones
(catastrophic forgetting). Khetarpal et al. [1] review continual RL and classify its non-stationarity; Wolczyk et
al. [2] show on a robotic benchmark that forward transfer, not only forgetting, is a key open problem. Three families
of remedies exist. *Regularisation* slows change of important weights, e.g. EWC [3]; *distillation* keeps the outputs
of an earlier model, e.g. Learning without Forgetting [4], policy distillation [5] and Progress & Compress [6], which
alternates learning a new task with consolidating it into a knowledge base. *Replay* re-trains on stored experience:
Isele and Cosgun [7] keep a long-term buffer chosen to match the global training distribution; CLEAR [8] combines
off-policy learning and behavioural cloning from replay with on-policy learning and does not need task identities.
Most regularisation and consolidation methods need to be told when a task ends ([3], [6]); we compare against CLEAR
because, like our agent, it needs no task labels.

**Task-free continual learning.** Aljundi et al. [9] move continual learning to a setting without task boundaries
and decide from the data stream itself when to consolidate. CURL [10] infers tasks inside the model in unsupervised
learning. In RL, OWL [11] learns one head per task and picks a head at test time with a bandit, and 3RL [12] adds a
recurrent (GRU) context to SAC with replay of past tasks so the agent can infer the current task from its history;
3RL can surpass its own multi-task version, which the authors relate to gradient interference between tasks. Dick et al. [26] detect
context changes with statistical tests on Wasserstein distances between past and current experience and train one
policy per detected context. Our agent is also task-free and keeps a single policy, but detects a task change from a collapse of its own
success and responds by re-initialising its network, and stores experience only while it is competent, so the
fixed-size memory holds mastered behaviour.
Reservoir sampling [13] keeps a uniform sample of a stream in fixed memory and is a strong replay baseline in
supervised continual learning [14]; we gate which experience enters the reservoir.

**Loss of plasticity.** Networks that keep training lose the ability to learn new things: Dohare et al. [15] show
this on ImageNet and RL and propose re-initialising rarely used units; Lyle et al. [16] link it to the curvature of
the loss landscape; Abbas et al. [17] show it for value-based agents on a repeating sequence of Atari games; Sokar et
al. [18] find growing numbers of dormant neurons. Nikishin et al. [19] counter the "primacy bias" in single-task, replay-based
agents by re-initialising the last layers (the whole network for SAC) on a fixed schedule while keeping the replay
buffer, and Ash and Adams [20] shrink and perturb weights before training on new data.
Reset & Distill [25] is the closest recipe to ours in continual RL: it resets the online actor and critic after each
task and distills old experts from a small stored buffer, but it is told when each task ends. On MiniGrid we
found that the main symptom in PPO is collapsed policy entropy, which resetting the output heads cures; on MinAtar
the loss persists with healthy entropy and under 2% dormant units, ReDo [18] does not help, and only a full
re-initialisation restores learning (to ~90% of learning from scratch). Unlike the fixed schedule in [19] and the
given task ends in [25], our reset is triggered by self-detected surprise, and old skills return only through
distillation from the competence-gated memory.

**Environments and algorithm.** We use PPO [21] in a CleanRL-style single-file implementation [22], MiniGrid [23]
and MinAtar [24], a 10x10 version of five Atari games designed for thorough, many-seed experiments.

## Where each claim comes from (for the author)
| Claim in our paper / text | Paper | Link |
|---|---|---|
| Continual RL survey, taxonomy of non-stationarity | [1] Khetarpal et al., JAIR 2022 | https://www.jair.org/index.php/jair/article/view/13673 |
| Forward transfer matters, not only forgetting | [2] Wolczyk et al., NeurIPS 2021 | https://arxiv.org/abs/2105.10919 |
| EWC: slow down weights important for old tasks (our EWC baseline) | [3] Kirkpatrick et al., PNAS 2017 | https://arxiv.org/abs/1612.00796 |
| Distillation keeps old outputs without old data | [4] Li & Hoiem, ECCV 2016 | https://arxiv.org/abs/1606.09282 |
| Distilling RL policies, multi-task consolidation (our KL distill retention) | [5] Rusu et al., ICLR 2016 | https://arxiv.org/abs/1511.06295 |
| Learn new task then compress into knowledge base, fixed size | [6] Schwarz et al., ICML 2018 | https://proceedings.mlr.press/v80/schwarz18a.html |
| Long-term replay buffer, distribution matching best | [7] Isele & Cosgun, AAAI 2018 | https://ojs.aaai.org/index.php/AAAI/article/view/11595 |
| CLEAR: replay + behavioural cloning, no task identities needed (our baseline) | [8] Rolnick et al., NeurIPS 2019 | https://arxiv.org/abs/1811.11682 |
| Task-free CL: no boundaries, detect when to consolidate from the stream | [9] Aljundi et al., CVPR 2019 | https://arxiv.org/abs/1812.03596 |
| Task inference inside the model, no labels/boundaries | [10] Rao et al., NeurIPS 2019 | https://arxiv.org/abs/1910.14481 |
| Same observation, different task: one head per task + bandit selection | [11] Kessler et al., AAAI 2022 | https://arxiv.org/abs/2106.02940 |
| Task-agnostic continual RL with replay + recurrence (3RL) | [12] Caccia et al., CoLLAs 2023 | https://proceedings.mlr.press/v232/caccia23a.html |
| Reservoir sampling (our fixed-size memory) | [13] Vitter, ACM TOMS 1985 | DOI 10.1145/3147.3165 |
| Reservoir replay works well across the board in CL | [14] Chaudhry et al., arXiv 2019 | https://arxiv.org/abs/1902.10486 |
| Loss of plasticity, continual backprop | [15] Dohare et al., Nature 2024 | https://doi.org/10.1038/s41586-024-07711-7 |
| Plasticity loss tied to loss-landscape curvature | [16] Lyle et al., ICML 2023 | https://proceedings.mlr.press/v202/lyle23b.html |
| Plasticity loss on repeating Atari sequence | [17] Abbas et al., CoLLAs 2023 | https://proceedings.mlr.press/v232/abbas23a.html |
| Dormant neurons, ReDo | [18] Sokar et al., ICML 2023 | https://proceedings.mlr.press/v202/sokar23a.html |
| Periodic reset of last layers (closest to our head reset) | [19] Nikishin et al., ICML 2022 | https://arxiv.org/abs/2205.07802 |
| Shrink & perturb (our shrink baseline in experiment 3) | [20] Ash & Adams, NeurIPS 2020 | https://arxiv.org/abs/1910.08475 |
| Reset actor+critic + distill from small buffer, task boundaries given (closest recipe) | [25] Ahn et al., arXiv 2024 | https://arxiv.org/abs/2403.05066 |
| Task change detection with Wasserstein + KS tests, one policy per context | [26] Dick et al., CoLLAs 2024 | https://arxiv.org/abs/2405.19047 |
| PPO | [21] Schulman et al., arXiv 2017 | https://arxiv.org/abs/1707.06347 |
| CleanRL single-file implementation | [22] Huang et al., JMLR 2022 | https://jmlr.org/papers/v23/21-1342.html |
| MiniGrid | [23] Chevalier-Boisvert et al., NeurIPS 2023 D&B | https://arxiv.org/abs/2306.13831 |
| MinAtar | [24] Young & Tian, arXiv 2019 | https://arxiv.org/abs/1903.03176 |

## Honest positioning (what is and is not new)
- Not new: replay, distillation, head/last-layer reset, reservoir sampling, task-free CL. Each exists ([5], [8], [9],
  [13], [19]).
- Our contribution is the combination and its trigger: competence-gated filling of a fixed reservoir +
  self-detected surprise (own success collapses) that re-initialises the network, in on-policy PPO, with no task
  signal; and the measurement: on MiniGrid (8 stages, 6 task kinds) matches told-boundary distillation and beats
  CLEAR; on MinAtar beats CLEAR, no protection and our joint baseline (also at 2x budget), in both game orders.
- Closest work (a reviewer will ask how we differ):
  - [25] Reset & Distill (arXiv 2024, SAC and PPO, Meta-World): reset actor+critic and distill from a small stored
    buffer. Checked full text: resets "after finishing learning each task", i.e. task boundaries given. Ours: no
    boundaries, reset triggered by own success collapse. Our final variant also resets the whole network, so the
    recipe is close: the difference is who decides when to reset (the agent itself vs. the experimenter) and
    gated, fixed-size memory vs. a per-task buffer. Must cite and compare in text.
  - [9] Aljundi et al.: checked full text. Detects plateaus in the loss with a sliding window (mean and variance below
    thresholds) and consolidates (MAS importance weights) there; keeps a 100-sample buffer of highest-loss samples.
    Our "store only while competent" is the same spirit as "consolidate at plateaus", in RL with success instead of
    loss. Say so openly.
  - [26] Dick et al.: statistical change detection, one policy per context (grows). Ours: one fixed-size network.
  - [12] 3RL (full text read 2026-10-25): SAC + GRU context + replay, task-agnostic, Meta-World; no resets, no
    change detection. Its finding that task-agnostic CL can beat multi-task training (gradient interference) matches
    our MinAtar joint result; cite it there.
  - [19] Primacy bias (full text read 2026-10-25): single-task sample efficiency, resets on a fixed schedule (SPR last
    layer every 2e4 steps; SAC whole network every 2e5; DrQ last 3 of 7 layers), replay buffer kept, optimizer state
    also reset ("almost no impact"). Not continual learning. Ours: triggered by surprise, PPO (no replay buffer of
    raw experience for RL updates), old skills via distillation from a gated reservoir.
- CLEAR is also task-free; our claim against it rests only on measured results, not on "they need task labels".

## Still to do
- Full text of [9], [12], [19], [25] read; positioning updated for the final method (full reset, exp 21).
- First search for 2024-2026 work done (found [25], [26]); one more pass before submission (new papers appear).
- Seen, not cited (different problem): Hafez & Erekmen, task-agnostic policy distillation, Sci. Rep. 14, 2024
  (intrinsically motivated exploration then distillation), https://doi.org/10.1038/s41598-024-80774-8
- Not verified: TPAMI 2018 journal version of [4] (cite ECCV version); MiniGrid paper page numbers.

## References (IEEE style)
[1] K. Khetarpal, M. Riemer, I. Rish, and D. Precup, "Towards continual reinforcement learning: A review and perspectives," J. Artif. Intell. Res., vol. 75, pp. 1401-1476, 2022.
[2] M. Wolczyk, M. Zajac, R. Pascanu, L. Kucinski, and P. Milos, "Continual World: A robotic benchmark for continual reinforcement learning," in Proc. NeurIPS, vol. 34, 2021.
[3] J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," Proc. Natl. Acad. Sci., vol. 114, no. 13, pp. 3521-3526, 2017.
[4] Z. Li and D. Hoiem, "Learning without forgetting," in Proc. ECCV, 2016, pp. 614-629.
[5] A. A. Rusu et al., "Policy distillation," in Proc. ICLR, 2016.
[6] J. Schwarz et al., "Progress & compress: A scalable framework for continual learning," in Proc. ICML, PMLR 80, 2018, pp. 4528-4537.
[7] D. Isele and A. Cosgun, "Selective experience replay for lifelong learning," in Proc. AAAI, vol. 32, no. 1, 2018.
[8] D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap, and G. Wayne, "Experience replay for continual learning," in Proc. NeurIPS, vol. 32, 2019, pp. 350-360.
[9] R. Aljundi, K. Kelchtermans, and T. Tuytelaars, "Task-free continual learning," in Proc. IEEE/CVF CVPR, 2019, pp. 11254-11263.
[10] D. Rao, F. Visin, A. Rusu, R. Pascanu, Y. W. Teh, and R. Hadsell, "Continual unsupervised representation learning," in Proc. NeurIPS, vol. 32, 2019.
[11] S. Kessler, J. Parker-Holder, P. Ball, S. Zohren, and S. J. Roberts, "Same state, different task: Continual reinforcement learning without interference," in Proc. AAAI, 2022.
[12] M. Caccia, J. Mueller, T. Kim, L. Charlin, and R. Fakoor, "Task-agnostic continual reinforcement learning: Gaining insights and overcoming challenges," in Proc. CoLLAs, PMLR 232, 2023, pp. 89-119.
[13] J. S. Vitter, "Random sampling with a reservoir," ACM Trans. Math. Softw., vol. 11, no. 1, pp. 37-57, 1985.
[14] A. Chaudhry et al., "On tiny episodic memories in continual learning," arXiv:1902.10486, 2019.
[15] S. Dohare, J. F. Hernandez-Garcia, Q. Lan, P. Rahman, A. R. Mahmood, and R. S. Sutton, "Loss of plasticity in deep continual learning," Nature, vol. 632, pp. 768-774, 2024.
[16] C. Lyle, Z. Zheng, E. Nikishin, B. Avila Pires, R. Pascanu, and W. Dabney, "Understanding plasticity in neural networks," in Proc. ICML, PMLR 202, 2023, pp. 23190-23211.
[17] Z. Abbas, R. Zhao, J. Modayil, A. White, and M. C. Machado, "Loss of plasticity in continual deep reinforcement learning," in Proc. CoLLAs, PMLR 232, 2023, pp. 620-636.
[18] G. Sokar, R. Agarwal, P. S. Castro, and U. Evci, "The dormant neuron phenomenon in deep reinforcement learning," in Proc. ICML, PMLR 202, 2023, pp. 32145-32168.
[19] E. Nikishin, M. Schwarzer, P. D'Oro, P.-L. Bacon, and A. Courville, "The primacy bias in deep reinforcement learning," in Proc. ICML, PMLR 162, 2022, pp. 16828-16847.
[20] J. Ash and R. P. Adams, "On warm-starting neural network training," in Proc. NeurIPS, vol. 33, 2020.
[21] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.
[22] S. Huang et al., "CleanRL: High-quality single-file implementations of deep reinforcement learning algorithms," J. Mach. Learn. Res., vol. 23, no. 274, pp. 1-18, 2022.
[23] M. Chevalier-Boisvert et al., "Minigrid & Miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks," in Proc. NeurIPS Datasets and Benchmarks Track, 2023.
[24] K. Young and T. Tian, "MinAtar: An Atari-inspired testbed for thorough and reproducible reinforcement learning experiments," arXiv:1903.03176, 2019.
[25] H. Ahn, J. Hyeon, Y. Oh, B. Hwang, and T. Moon, "Reset & distill: A recipe for overcoming negative transfer in continual reinforcement learning," arXiv:2403.05066, 2024.
[26] J. Dick, S. Nath, C. Peridis, E. Benjamin, S. Kolouri, and A. Soltoggio, "Statistical context detection for deep lifelong reinforcement learning," in Proc. CoLLAs, 2024.
