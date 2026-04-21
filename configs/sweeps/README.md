# W&B Sweeps

One tiny example sweep so attendees can see the Sweeps UI during the
workshop. The grid is deliberately small — two seeds, 2 k steps each — so
10 agents finish in about an hour on a single H100.

```bash
wandb sweep configs/sweeps/diffusion_lr_search.yaml
# prints a sweep ID, then:
wandb agent <entity>/<project>/<sweep_id>
```

To extend: swap `method: grid` to `method: bayes`, add `parameters.lr` and
`parameters.batch-size` ranges, bump `steps`, and raise `--count` on the
agent.
