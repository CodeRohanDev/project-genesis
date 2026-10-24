# 3a: learned world model + skill-order feasibility (2026-10-24, overnight)

## Question
Can the agent learn how the world responds to its actions (a world model), including cause-effect rules, from its own
experience, and does that knowledge hold in layouts it never saw?

## Setup (worldmodel.py, wm_rollouts.py; code 381a092 -> d42714b)
- Data: our B_finetuneUP agents (Unlock + Pickup skills + UnlockPickup) act with 30% random actions, plus fully random
  play (supplies failed actions such as toggling a locked door without the key).
  Train: 849,968 transitions (agents 1-7, training env seeds). Test: 180,000 (agents 8-10, eval seed range).
  New worlds: 180,000 on DoorKey-8x8 + KeyCorridorS3R1, never seen by the model.
- Target: next 7x7x3 egocentric view (object type, colour, state per cell) from (view, action). 8 epochs, Adam 1e-3.
- Variants (5 model seeds each):
  plain   = MLP over the whole view;
  persist = MLP + "things stay as they are unless changed" (current view added to output logits; object permanence);
  conv    = 6-layer 3x3 conv net (same local rule everywhere = locality prior) + persistence.
- Baseline "copy": predicts that nothing changes.
- Metrics: changed-cell accuracy; exact next view on pickup/drop/toggle (view geometry unchanged, so fully
  predictable; turns and moves reveal unseen cells that no model can know); causal probes; multi-step imagination
  (start from a real view, feed real actions, never look at real views again).

## Results (mean ± 95% CI over 5 model seeds)
| | familiar test: changed cells | familiar: exact view, fixed geometry | new worlds: changed cells | new worlds: exact view, fixed geometry |
|---|---|---|---|---|
| copy (nothing changes) | 0 | 74 | 0 | 89 |
| plain | 96 ± 0 | 95 ± 1 | 72 ± 1 | 36 ± 5 |
| persist | 96 ± 0 | 98 ± 0 | 73 ± 1 | 56 ± 5 |
| conv | 96 ± 0 | 97 ± 1 | 71 ± 1 | 68 ± 6 |

Causal probes (front cell / own cell predicted exactly right):
| Probe (n familiar / n new) | plain fam | plain new | persist new | conv new | copy |
|---|---|---|---|---|---|
| toggle locked door WITH key -> opens (3366 / 495) | 100 | 98 ± 6 | 98 ± 5 | 80 ± 23 | 0 |
| toggle locked door WITHOUT key -> stays locked (75 / 872) | 100 | 97 ± 8 | 100 ± 0 | 94 ± 10 | 100 |
| pick up facing object -> object in hand, cell empty (9767 / 2671) | 100 | 95 ± 4 | 98 ± 2 | 97 ± 5 | 0 |
| walk into wall -> nothing changes (6401 / 18715) | 100 | 88 ± 9 | 92 ± 6 | 100 ± 0 | 100 |
(conv familiar "without key" 85 ± 42: one seed failed on the 75 familiar cases.)

Imagination, what the agent holds after k imagined steps (familiar | new worlds):
| | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| copy | 91 \| 97 | 76 \| 90 | 63 \| 85 | 65 \| 80 |
| plain | 100 \| 87 | 98 \| 78 | 91 \| 73 | 84 \| 66 |
| persist | 100 \| 90 | 98 \| 83 | 91 \| 79 | 85 \| 72 |
| conv | 100 \| 94 | 97 \| 84 | 89 \| 76 | 81 \| 68 |

## Findings
1. The agent learned real cause-effect rules from its own experience: key + toggle opens a locked door, no key does
   not, pick-up moves the object into the hand. 100% on familiar data, 94-100% in never-seen worlds (persist).
2. In familiar worlds imagination is reliable far ahead: after 20 imagined steps it still knows what it holds 85% of
   the time and 90% of all cells, much better than "nothing changes" (65%).
3. In new worlds the model is worse than "nothing changes" at keeping the unchanged scenery: it redraws walls and empty
   floor to look like the rooms it trained in (plain: 36% exact vs copy 89%). It memorised room layouts instead of
   learning that most things stay put.
4. Built-in priors help, step by step: object permanence 36 -> 56%, plus locality (convolution) -> 68%. Neither closes
   the gap to 89%. This mirrors what infants are thought to start with (object permanence, local physics) and suggests
   such priors matter for generalising world models.
5. Over many imagined steps in new worlds all models stay below "nothing changes" for what the agent holds.

## Feasibility for planning over skills (skill_sequence.py, B_comp agents, 100 episodes x 10 seeds, no new training)
The agent is told a fake mission while the environment keeps the real goal "pick up the <c> box" (UnlockPickup).
| Mission order | UnlockPickup success |
|---|---|
| real mission only (= zero-shot) | 0% (all seeds) |
| wrong order (box first, then "open the door") | 0% (max 1%) |
| correct order ("open the door" until open, then real mission) | **26%** (seeds: 23 31 16 16 52 15 19 16 33 37) |
So a planner that only chooses the ORDER of known skills could reach up to ~26% on a task that is 0% zero-shot.
Untested hypothesis for the gap to 100%: after opening the door the agent still holds the key and can carry only one
object; "drop the key" was never trained as a skill, so it must stumble on it.

## Limits
- The model predicts views, not rewards; success in imagination must be read from the predicted view (object in hand).
- UnlockPickup trajectories were in the world-model training data (from B_finetuneUP). For a fair zero-shot planning
  test the model must be retrained on component-only data (planned in 3b).
- 5 model seeds; probe counts for "without key" in familiar data are small (75).

---

# 3b: planning over skills with the world model (2026-10-24) -> NEGATIVE

Setup (planner.py): B_comp agents (seeds 1-10, never trained on UnlockPickup). World model retrained on component-only
data (`collect_comp`: B_comp agents in Unlock + Pickup worlds; UnlockPickup never seen), persist MLP, 3 model seeds,
trained on the GPU (3.5 min for 3 models vs ~25 min on CPU). On UnlockPickup transitions (unseen) this model is
accurate: changed cells 96%, fixed-geometry views 99%, all causal probes 98-100%.
Skills: "open the door", "pick up a key", "pick up a ball" (irrelevant), real mission. Completion read from the
agent's own view; the agent remembers which skills it completed in the episode.
Planner: per decision, each unfinished skill imagined K=16 times for H=40 steps in the world model; score uses the
imagined outcome and the agent's own critic (value) for the real mission. 100 episodes x 10 agents per method.

| Method | v1 (episodes 1e6+) | v2 (fresh episodes 1e6+1e4+) |
|---|---|---|
| real mission only (zero-shot) | 0% | 0% |
| random skill choice | 12% | 12% |
| hand-written correct order | 23% | 21% |
| **planner** | **9%** | **4%** |
v1 score: real mission = imagined success rate; other skills = completed x critic(real mission | imagined end state).
v2 score: real mission also scored with the critic when not completed in imagination (same yardstick).

Falsification criterion (planner must beat random) is met: hypothesis rejected for this planner.

Diagnosis:
- v1 got the first steps right: "pick up a key" first in ~75% of episodes, then "open the door". It then picked the
  irrelevant "pick up a ball" over the real mission. The box is in the unseen room, so the real mission never succeeds
  in imagination (score ~0), while any small critic value beats 0.
- v2 picks the real mission first in ~98% of episodes: the critic rates "pick up the box" from the start state higher
  than after fetching the key.
- Root cause: the critic only knows situations from training (box visible, single room). For a goal hidden behind
  unexplored space it gives meaningless values, and the world model cannot imagine what it has not seen.
  Planning needs a way to value progress toward a goal that is out of sight (subgoals / preconditions, or exploring
  to make the unknown visible). A good world model of local physics is not enough.
- Not iterated further on purpose: more scoring tweaks evaluated on these agents would fit the test.

## 3b v3: curiosity rule (2026-10-24)
Rule fixed before the run: if the target has been seen this episode -> real mission; otherwise run the unfinished skill
that is most doable in imagination (completion rate over K=16 imagined runs, threshold 0.2), counting only skills
whose object (door / key / ball) the agent has actually seen this episode. The seen-object condition was added after a
smoke test on non-evaluation episodes (seeds 5-12) showed the world model imagining a ball that did not exist.
Fresh episodes (eval seeds + 20,000), 100 per agent, 10 agents.

| Method | UnlockPickup |
|---|---|
| real mission only | 0% |
| random skill choice | 13% |
| **planner v3** | **16%** (per agent: 6 18 14 10 23 7 9 19 29 29) |
| hand-written correct order | 26% |

Paired over agents: planner - random = +3.7 ± 6.2 points (not significant); planner - correct = -9.8 ± 4.1 points.
First skill chosen (1000 episodes): pick up key 65%, real mission 29% (key/door not yet seen), open door 6%, ball 0%.

Reading: the planner now discovers the right first step on its own and never wastes time on the absent ball, but the
gain over random choice is small and not significant. Frequent pattern "key -> real mission" (door skipped): once the
door is out of the current view, imagining "open the door" from there rarely completes within 40 steps, so the
doable-score falls under the threshold. Imagination is local to what is in view; it does not plan routes to things
remembered but not visible.

## 3b overall conclusion
A learned world model that is accurate on local physics, plus learned skills, plus simple planning rules, is not
enough for reliable zero-shot composition here. Missing pieces identified by these runs:
1. memory of where things are (a map), so the agent can imagine going back to a remembered door;
2. a way to value progress toward goals that are out of sight (subgoals / preconditions);
3. a "drop" skill (only one object can be carried) - hypothesis for the 26% ceiling of the correct order.

## 3b v4: map memory (2026-10-24)
mapmem.py (hand-designed, declared): path integration from own actions + whether the cell ahead was free; top-down
map of everything seen; shortest-path navigation over remembered free cells; rendering the view the agent would see
from a remembered spot. Tested against the true game state: 0 pose errors and 0 wrong wall/door/empty cells over
30 random-play episodes (1107 cells); navigation to a remembered key 15/15; rendered view = real view 30/30.
Planner v4 (rule fixed before the run): if the target box is in the map -> imagine "pick up" from the rendered view
in front of it with the current hand and with the hand the world model predicts after "drop"; do whichever is more
doable. Otherwise, for every remembered key / closed door / ball, render the view in front of it, imagine that skill
there (K=16, 10 steps) and do the most doable one (>= 0.2), else explore with the real-mission policy for 40 steps.
Execution detail shared by all map methods: to drop, turn until the cell ahead is empty (found in smoke tests on
non-evaluation episodes 5-14: dropping against a wall silently fails).
Fresh episodes (eval seeds + 30,000), 100 per agent, 10 agents.

| Method | UnlockPickup |
|---|---|
| real mission only | 0% |
| random skill choice, no map | 13% |
| correct order, no map | 25% |
| **planner v4 (imagination + map)** | **83%** (97 89 93 90 76 98 62 57 87 82) |
| **map_random: same map/navigation/candidates, choices at random** | **93%** (100 95 87 93 83 97 92 92 93 100) |
| correct order + map (hand-written) | 98% |

Paired over agents: planner - random(no map) = +70.3 ± 10.9; planner - map_random = **-10.1 ± 9.5**;
planner - correct_map = -14.9 ± 9.1.

## What v4 actually shows (read this before citing the 83%)
- The large gain comes from MEMORY + NAVIGATION, not from imagination: with the map, even random choices reach 93%.
  Without the map, the correct order reaches only 25%. The bottleneck in 3b was finding and returning to things,
  not deciding what to do.
- Imagination-based choice is not better than random choice here (slightly worse). In this task there are only 2-3
  candidate skills, failed attempts are cheap, and the agent can retry many times within 288 steps, so random search
  with reliable execution works. A test where wrong choices are costly or irreversible is needed to measure the value
  of thinking ahead.
- The planner does make sensible choices (key first in ~50% of episodes, explore otherwise; door after key; drop
  before pick-up when needed), but sensible is not the same as better than random under cheap retries.
- Hand-designed components now carry much of the performance (map, path integration, navigation, drop turning).
  They are generic, not task-specific, but they are not learned.

## 3b v5: costly mistakes (tighter step budgets), fresh episodes (eval seeds + 40,000)
| Budget (max steps) | map_random | planner (imagination + map) | correct order + map |
|---|---|---|---|
| 60 | 47% | 44% | 74% |
| 90 | 76% | 75% | 97% |
| 120 | 86% | 80% | 98% |
Planner - map_random (paired over 10 agents): -3.0 ± 8.1 (60), -1.0 ± 11.3 (90), -6.9 ± 11.0 (120).
Right decisions matter (correct order is +27 points at budget 60), but the planner does not deliver them.

Diagnostic (non-evaluation episodes 200-219, agents 3/6 good, 7/8 poor): mean imagined doability per skill & hand
| | open, hand empty | open, holding key | pick up box, holding key | pick up box, hand empty | key, hand empty |
|---|---|---|---|---|---|
| agent 3 (20/20 wins) | 0.13 | 1.00 | 0.09 | 0.96 | 1.00 |
| agent 8 (12/20 wins) | 0.19 | 1.00 | 0.01 (117 evaluations) | 0.93 | 1.00 |
The world model's judgements are correct for every agent: a door does not open without the key, it does with the key;
a box cannot be picked up while holding the key. Poor agents evaluate "pick up box while holding key" over and over:
after dropping the key, their weak pick-up skill (component Pickup ~54%) picks the key up again -> loop.
The "open without key" score (0.13-0.19) also sits close to the 0.2 threshold, so noise sometimes sends the agent to
the door without the key.

## 3b final reading
- Learned causal knowledge is correct and usable (diagnostic above).
- Success is limited by memory/navigation (fixed by the map) and by skill execution (weak pick-up skill), not by
  the decision rule. With reliable execution, even random choices do well because retries are cheap.
- To show the value of imagination, the next test needs (a) reliable skills and (b) irreversible mistakes.
