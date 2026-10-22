# Stage 0: Setup (2026-10-22)

- Hardware: GTX 1050 Ti 4GB (sm_61), driver 580, 12 CPU threads, ~14GB RAM, native Ubuntu.
- Python 3.14.4, torch 2.14.1+cu126, gymnasium 1.4.0, minigrid 3.2.0 (full list: requirements.txt).
- Issue: default torch (cu130) has no sm_61 kernels. Fix: install from cu126 index:
  `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`
- smoke_test.py: GPU ok, obs 7x7x3 partial view.

Random-agent success (100 episodes, seed 0), the floor every agent must beat:

| Task | Random success |
|---|---|
| Empty-8x8 | 19% |
| DoorKey-8x8 | 3% |
| LavaCrossingS9N1 | 0% |
