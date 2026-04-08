# Agent Guidelines: vLLM + Qwen3.5-27B on Mixed GPUs

## Critical Setup Requirements

### 1. Version Compatibility
- **vLLM 0.19.0 requires transformers 5.5** (not 4.49) for Qwen3.5-27B RoPE
- After installing vLLM, run:
  ```bash
  uv pip install -U transformers
  ```
- **Verification**: Run `transformers.__version__` - must be `>= 5.5`

### 2. GPU Parallelism (MUST FOLLOW)
- **NEVER use TP** (Tensor Parallel) on mixed GPU setups (e.g., RTX 4090 SM89 + RTX 3090 SM80)
  - Different FP8 W8A8 vs W8A16 implementations cause silent errors
  - Errors accumulate → garbage output within long contexts
- **Use PP** (Pipeline Parallel, `--pipeline-parallel-size 2`) instead
  - No cross-GPU matrix multiplication mixing
  - Only stable configuration for heterogeneous hardware

### 3. Model & Quantization Selection
**Recommended:** `QuantTrio/Qwopus3.5-27B-v3-AWQ`
- AWQ (INT4) + PP mode = full 219k context + tool calling stability
- Distilled from Claude 4.6 Opus (fixes "calling tools without `</thinking>`" bug)
- AWQ uniform quantization = consistent results on both SM80 and SM89

**Avoid:** `Qwen/Qwen3.5-27B-FP8`
- FP8 breaks on mixed GPUs (TP mode required for VRAM, but TP mode unstable)
- Distillation bug: unstable tool calling tools (premature stop tags)

Alternative: `mconcat/Qwopus3.5-27B-v3-FP8-Dynamic`
- Only use if memory-constrained; forces `--max-model-len 100000` (cut by ~55%)

### 4. Tool Calling Configuration
- **Qwopus series (distilled)**: Use `--tool-call-parser hermes`
  - These models were distilled from Claude (Hermes teacher format)
- **Official Qwen3.5-27B**: Use `--tool-call-parser qwen3_xml`
  - But **broken**: distillation artifacts cause premature `<stop>` tags
  - **Do not use official Qwen3.5-27B for tool calling**

**Rule:** Choose distilled model (Qwopus) → use Hermes parser. Avoid official model entirely unless for non-function-calling use cases.

---

## Summary Checklist
- [ ] `transformers>=5.5` installed after vLLM 0.19.0
- [ ] Using `QuantTrio/Qwopus3.5-27B-v3-AWQ`
- [ ] `--pipeline-parallel-size 2` (PP, NOT TP)
- [ ] `--tool-call-parser hermes`
- [ ] `--max-model-len 219520` (if AWQ) or `100000` (if FP8+PP)
- [ ] `NCCL_P2P_DISABLE=1`, `NCCL_ALGO=Ring` for WSL2 mixed GPU

See `README.md "Pain X"` sections for detailed explanations.
