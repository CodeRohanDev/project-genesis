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

Paired over agents: planner - random = +3.7 ± 6.1 points (not significant); planner - correct = -10 points.
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
