# Configuration files

`pretrain_smoke.yaml` is a deliberately small CPU configuration for execution
checks. `pretrain.yaml` describes the production-size architecture and default
training parameters, but running it is not a claim that the paper has been
reproduced.

The three `finetune_*_smoke.yaml` files exercise binary classification,
multilabel classification with missing labels, and regression on 64 sampled
molecules. They use randomly initialized small encoders because
`pretrained_checkpoint` is null; their outputs are execution diagnostics, not
benchmark results. See `docs/finetuning.md` for checkpoint and encoder-mode
rules.

Paths in checked configuration files are resolved relative to the repository
root. Every training run writes its fully resolved effective configuration to a
unique ignored directory under the configured `output.root`.
