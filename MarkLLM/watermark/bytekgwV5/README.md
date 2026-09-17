# ByteKGWv5 (standalone)

This folder contains a *standalone* implementation of ByteKGWv5 with **no imports from MarkLLM**.
It relies only on `torch` and `transformers`.

## Features

- `sampling_scheme="token"`: HF-like token sampling (one multinomial per step).  
  Intended for **delta=0 alignment checks** of your logits pipeline (processors+warpers) without calling `model.generate` in the v5 implementation.

- `sampling_scheme="byte_tree"`: hierarchical **257-branch** sampling (256 bytes + END) using token visible UTF-8 bytes.  
  Watermark bias `delta` is applied at the **branch** level for green bytes at each `byte_pos`.

## Quick start

Install by adding this folder to your `PYTHONPATH`, e.g.:

```bash
export PYTHONPATH=$PWD/bytekgwv5:$PYTHONPATH
```

Generate:

```bash
python bytekgwv5/scripts/generate_v5.py --model ../Meta-Llama-3-8B --prompt "Hello, my name is" --do_sample --top_p 0.95 --top_k 50
```

Alignment check (v5 token-mode vs HF generate):

```bash
python bytekgwv5/scripts/test_alignment_token_mode.py --model ../Meta-Llama-3-8B --prompt "Hello, my name is" --do_sample --top_p 0.95 --top_k 50 --seed 1234
```

## Notes

- Byte-tree sampling uses float32 probabilities for stability; delta=0 is *distribution-equivalent* to token sampling, but not necessarily **seed-coupled** to HF.
- Token-mode aims to be seed-coupled to HF; if you see mismatches, try `--sampling_dtype float32`.
