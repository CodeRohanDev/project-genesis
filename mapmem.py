"""Map memory for the planner (3b v4): the agent builds a map of what it has seen and can go back to remembered things.

Hand-designed parts (declared): path integration (own position from own actions + whether the cell ahead was
free), writing the 7x7 view into a top-down map, shortest-path navigation on known free cells, and rendering the
view the agent WOULD see from a remembered spot. Learned parts stay learned: skills (policy) and world model.
The agent never reads the true position from the game; tests below check its own estimate against it.
"""
from collections import deque

import numpy as np

DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # minigrid: 0 right, 1 down, 2 left, 3 up
FREE = {1, 3, 8}  # empty, floor, goal
LEFT, RIGHT, FORWARD = 0, 1, 2


def walkable(cell):
    t, _, s = cell
    return t in FREE or (t == 4 and s == 0)  # or an open door


class Map:
    def __init__(self, size=64):
        self.cells = np.zeros((size, size, 3), np.int64)  # type 0 = never seen
        self.pos = np.array([size // 2, size // 2])
        self.dir = 0  # own frame: initial heading is "0"

    def offsets(self):
        fx, fy = DIRS[self.dir]
        rx, ry = -fy, fx  # right of heading
        return np.array([fx, fy]), np.array([rx, ry])

    def observe(self, view):
        """view: 147 ints (7x7x3, index (i*7+j)*3); agent at i=3, j=6 facing j-decreasing."""
        f, r = self.offsets()
        v = np.asarray(view[:147]).reshape(7, 7, 3)
        for i in range(7):
            for j in range(7):
                if v[i, j, 0] == 0 or (i == 3 and j == 6):
                    continue
                x, y = self.pos + (6 - j) * f + (i - 3) * r
                self.cells[x, y] = v[i, j]

    def front(self):
        f, _ = self.offsets()
        return tuple(self.pos + f)

    def step(self, action, view_before):
        """Path integration: update own pose after taking `action` from `view_before`."""
        if action == LEFT:
            self.dir = (self.dir - 1) % 4
        elif action == RIGHT:
            self.dir = (self.dir + 1) % 4
        elif action == FORWARD:
            ahead = np.asarray(view_before[:147]).reshape(7, 7, 3)[3, 5]
            if walkable(ahead):
                self.pos = self.pos + self.offsets()[0]

    def find(self, t, c=None):
        hits = np.argwhere((self.cells[..., 0] == t) & ((self.cells[..., 1] == c) if c is not None else True))
        return [tuple(h) for h in hits]

    def path_to_face(self, target):
        """Shortest action list that ends with the agent next to `target` and facing it, over known free cells."""
        start = (int(self.pos[0]), int(self.pos[1]), self.dir)
        goals = {(target[0] - dx, target[1] - dy, d) for d, (dx, dy) in enumerate(DIRS)}
        prev = {start: None}
        q = deque([start])
        while q:
            s = q.popleft()
            if s in goals:
                acts = []
                while prev[s] is not None:
                    s, a = prev[s]
                    acts.append(a)
                return acts[::-1]
            x, y, d = s
            nxt = [((x, y, (d - 1) % 4), LEFT), ((x, y, (d + 1) % 4), RIGHT)]
            fx, fy = DIRS[d]
            if walkable(self.cells[x + fx, y + fy]):
                nxt.append(((x + fx, y + fy, d), FORWARD))
            for n, a in nxt:
                if n not in prev:
                    prev[n] = (s, a)
                    q.append(n)
        return None

    def render(self, pos, d, hand):
        """The 7x7 view the agent would see at pos/heading d, from memory; hand = (t, c, s) in the agent cell."""
        fx, fy = DIRS[d]
        f, r = np.array([fx, fy]), np.array([-fy, fx])
        v = np.zeros((7, 7, 3), np.int64)
        for i in range(7):
            for j in range(7):
                x, y = np.asarray(pos) + (6 - j) * f + (i - 3) * r
                if 0 <= x < len(self.cells) and 0 <= y < len(self.cells):
                    v[i, j] = self.cells[x, y]
        v[3, 6] = hand
        return v.reshape(-1)


if __name__ == "__main__":
    # check own pose estimate and remembered cells against the true game state, under random play
    from ppo import make_env
    env = make_env("MiniGrid-UnlockPickup-v0")()
    rng = np.random.default_rng(0)
    start = np.array([32, 32])
    rot1 = lambda v: np.array([-v[1], v[0]])  # one quarter turn (minigrid: right -> down)

    def to_true(v, k):
        for _ in range(k):
            v = rot1(v)
        return v

    bad_pose = bad_cells = checked = 0
    for ep in range(30):
        obs, _ = env.reset(seed=ep)
        u = env.unwrapped
        m = Map()
        m.observe(obs)
        a0, d0 = np.array(u.agent_pos), u.agent_dir
        for t in range(150):
            a = int(rng.integers(0, 6))
            before = obs
            obs, r, term, trunc, _ = env.step(a)
            m.step(a, before)
            m.observe(obs)
            est = a0 + to_true(m.pos - start, d0)
            bad_pose += int(tuple(est) != tuple(u.agent_pos) or (m.dir + d0) % 4 != u.agent_dir)
            if term or trunc:
                break
        for x, y in np.argwhere(m.cells[..., 0] > 0):
            tx, ty = a0 + to_true(np.array([x, y]) - start, d0)
            cell = u.grid.get(int(tx), int(ty))
            mt = m.cells[x, y, 0]
            checked += 1
            if cell is None:
                bad_cells += int(mt not in (1, 5, 6, 7, 10))  # empty now (objects may have been picked up since)
            elif cell.type in ("wall", "door"):
                bad_cells += int(mt != {"wall": 2, "door": 4}[cell.type])
    print(f"pose errors: {bad_pose}, map cells checked: {checked}, wrong wall/door/empty cells: {bad_cells}")
    assert bad_pose == 0 and bad_cells == 0
    print("map memory ok")
