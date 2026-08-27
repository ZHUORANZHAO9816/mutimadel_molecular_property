# Public `GTproEncoder` API

`GTproEncoder` is exported directly by `gtpro`; applications do not import
training modules:

```python
from gtpro import GTproEncoder

encoder = GTproEncoder.from_pretrained("path/to/pretraining-best.pt", device="cpu")
embeddings = encoder.encode_smiles(["CCO", "CCN"], representation="joint")
```

`from_pretrained` accepts a strict D2 grouped pretraining checkpoint, rebuilds
the recorded text/GROVER architecture, and rejects missing or shape-incompatible
weights. `freeze=True` is the default. `device` accepts `auto`, `cpu`, `cuda`,
or `mps`; unavailable explicitly requested accelerators fail clearly.

Representations are float32 tensors on the configured device:

| Representation | Shape for a list | Compact checkpoint example |
|---|---|---|
| `graph` | `[B, D_graph]` | `[B, 64]` |
| `text` | `[B, D_text]` | `[B, 64]` |
| `joint` | `[B, D_text + D_graph]` | `[B, 128]` |

A string input returns a rank-1 `[D]` tensor; a list, including a one-element
list, returns rank-2 `[B, D]`. `batch_size` bounds graph/text forward batches.

The default invalid policy is `raise`, with the failing input index and reason.
`invalid_smiles="nan"` instead preserves input order and fills the entire row
with NaN. For text/joint encoding, a valid molecule over the checkpoint's
maximum SMILES token length follows the same policy. Graph-only encoding is not
subject to the text length limit.
