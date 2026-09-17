# Data Provenance

This file answers the practical question:

> Are the current-method data files in this package, and can we tell whether they were produced by this code?

## Short Answer

Yes, the current-method data are included.

The current method is `ByteKGWv6`, and its generated samples are:

```text
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta1.0.csv
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta2.0.csv
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta3.0.csv
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta4.0.csv
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta5.0.csv
```

The corresponding clean and KGW baseline files in the same generation run are:

```text
outputs/v6_vs_kgw_gen_prfnew_mp/hf_generate.csv
outputs/v6_vs_kgw_gen_prfnew_mp/kgw_delta*.csv
```

The evaluated / bucketed current-method files are:

```text
outputs/wm_eval/bytekgw_v6/
```

## Evidence That The Files Match The Code

### 1. Method label in CSV

The `outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta*.csv` files contain:

```text
algo = bytekgwV6
```

and columns:

```text
algo,delta,device,seed,prompt_index,prompt_id,prompt_text,full_text,continuation_text,gen_params_json
```

This matches the output schema produced by:

```text
scripts/generate_clean_v6_kgw.py
```

### 2. Script output names match existing files

`scripts/generate_clean_v6_kgw.py` writes:

```text
hf_generate.csv
bytekgw_v6_delta{delta}.csv
kgw_delta{delta}.csv
```

The included directory contains exactly this naming pattern:

```text
outputs/v6_vs_kgw_gen_prfnew_mp/
```

### 3. Prompt metadata matches the generated CSVs

The generation metadata file:

```text
outputs/c4_samples_head_200/run_metadata.json
```

contains:

```text
model = ../Meta-Llama-3-8B-Instruct
n_prompts = 200
seed_base = 1234
deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
generation: temperature=1.0, top_p=0.95, top_k=50, max_new_tokens=128
```

The generated CSVs in `outputs/v6_vs_kgw_gen_prfnew_mp/` use the same prompts, seeds, deltas, and generation kwargs.

### 4. Config matches the thesis method

The current method config is:

```text
config/ByteKGWv6.json
```

Important parameters:

```text
delta = 3.0 default, swept as 1.0..5.0 in generation
n_bytes = 3
seed_window_chars = 18
m_bits = 256
target_anchors = 96
k_choices = [1, 2, 3, 4, 5, 6]
k_weight_mode = linear
decision_margin_bits = 12
normalize_whitespace = true
```

These correspond to the Chapter 4 method description: prefix classes plus fingerprint-driven PRF partitioning.

## What Cannot Be Proven From Files Alone

The CSV files do not contain the exact shell command, git commit, or a hash of the source file that produced them.

Therefore the strongest accurate statement is:

> The data are included and are strongly consistent with being produced by the included `scripts/generate_clean_v6_kgw.py` and `ByteKGWv6` implementation, using the included metadata/configs. The exact original command line is not embedded in the CSV files.

This is why `REPRODUCIBILITY.md` gives the command that should regenerate the same type of files, but bit-for-bit equality may depend on CUDA, Transformers, PyTorch, model revision, and sampling determinism.

## Included Data Categories

### Current Method

```text
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta*.csv
outputs/wm_eval/bytekgw_v6/
```

### Clean Cover Texts

```text
outputs/v6_vs_kgw_gen_prfnew_mp/hf_generate.csv
outputs/wm_eval/clean/hf_generate.csv
```

### Baselines

```text
outputs/v6_vs_kgw_gen_prfnew_mp/kgw_delta*.csv
outputs/wm_eval/kgw/
outputs/wm_eval/dip/
outputs/wm_eval/unbiased/
outputs/dip_gen_mp/
outputs/unbiased_gen_mp/
```

These are the comparison methods that are actually represented by included experiment data:

```text
KGW
DiP
Unbiased
```

Additional MarkLLM methods such as SynthID, Unigram, SIR, SWEET, TS, UPV, XSIR, EXP, and EXP-Gumbel are included as source/config modules where lightweight enough, but this clean package does not include full experiment result tables for all of them. They are preserved as runnable/inspectable baselines, not as completed result rows for the Chapter 4 tables.

### Mechanism Diagnostics

```text
outputs/trace_v6_attack_grid_bytekgw*.csv
sweep_v6_prf_r002/
traces_v6_process_r002/
outputs/entropy_prefix_sweep/
```

### Plotting Inputs

```text
plotting/tpr_attack_by_bucket_0.01.csv
plotting/tpr_attack_by_bucket_0.02.csv
plotting/tpr_attack_by_bucket_0.05.csv
plotting/tpr_attack_by_bucket_0.10.csv
```

## Excluded Data

Large or less relevant historical folders were not included, such as:

```text
outputs/wm_eval_attacked/
metrics/
textattack/
his/
large cache/log/archive files
```

These are not needed for the current ByteKGWv6 reproduction path.
