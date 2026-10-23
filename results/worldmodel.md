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
