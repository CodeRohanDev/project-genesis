"""Self-check for ppo.redo: an exactly-dead unit is re-initialised, its Adam moments cleared, and (because its
outgoing weights are zeroed) the network's outputs do not change at the moment of the reset.  python test_redo.py"""
import torch

import ppo

for mode in ["minatar", "minigrid"]:
    ppo.set_mode(mode)
    torch.manual_seed(0)
    a = ppo.Agent()
    opt = torch.optim.Adam(a.parameters(), 1e-3)
    obs = torch.rand(256, ppo.OBS_DIM) if mode == "minatar" else torch.randint(0, 6, (256, ppo.OBS_DIM)).float()
    d, v, _ = a(obs)
    (d.logits.sum() + v.sum()).backward()
    opt.step()
    conv = a.body[0]
    with torch.no_grad():
        conv.weight[3], conv.bias[3] = 0, -1.0  # unit 3 of layer 1: exactly dead
    before = a(obs)
    f = ppo.redo(a, opt, obs, 1e-6)  # tau ~0: only exactly-dead units
    after = a(obs)
    assert f > 0 and conv.weight[3].abs().sum() > 0 and conv.bias[3] != -1.0
    assert torch.allclose(before[0].logits, after[0].logits, atol=1e-5) and torch.allclose(before[1], after[1], atol=1e-5)
    assert opt.state[conv.weight]["exp_avg"][3].abs().sum() == 0
    print(mode, "ok")
