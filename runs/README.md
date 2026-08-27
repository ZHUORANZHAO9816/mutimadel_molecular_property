# Run directories

Training creates one immutable-style directory per invocation:

```text
runs/<experiment>/<UTC timestamp>_seed<seed>/
  config.yaml
  environment.json
  checkpoints/
```

`config.yaml` is the resolved effective configuration. `environment.json`
records command, Python/PyTorch/platform/device details, Git commit and dirty
state, seed, start/end times, duration, and final status. Generated run contents
are ignored by Git; only this policy README is tracked.
