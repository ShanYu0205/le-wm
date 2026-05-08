# LeWorldModel repository technical map (initial static pass)

Date: 2026-04-17
Repository: `/workspace/le-wm`

## 1) Structure map aligned to your research questions

### Training entry points
- `train.py`
  - Hydra entrypoint: `run(cfg)`
  - Training step logic: `lejepa_forward(self, batch, stage, cfg)`
  - Instantiates dataset, transforms, model components, optimizer, and trainer.

### Model definition files
- `jepa.py`
  - Core world model class: `JEPA`
  - Main methods: `encode`, `predict`, `rollout`, `criterion`, `get_cost`
- `module.py`
  - Building blocks: `ARPredictor`, `Transformer`, `ConditionalBlock`, `Embedder`, `MLP`, `SIGReg`

### Encoder module
- Encoder created in `train.py` via:
  - `spt.backbone.utils.vit_hf(...)`
- Used in `jepa.py::JEPA.encode`:
  - `output = self.encoder(pixels, interpolate_pos_encoding=True)`
  - CLS token extraction: `output.last_hidden_state[:, 0]`
  - Projection head: `self.projector(...)`

### Predictor / dynamics module
- `module.py::ARPredictor`
  - Causal transformer with AdaLN-zero conditioning (`ConditionalBlock`) on action embeddings.
- `jepa.py::JEPA.predict`
  - Calls `self.predictor(emb, act_emb)` then `self.pred_proj(...)`.

### Regularization / anti-collapse
- `module.py::SIGReg`
  - Sketch Isotropic Gaussian regularizer.
- Integrated in `train.py::lejepa_forward`:
  - `output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))`
  - total loss: `pred_loss + lambda * sigreg_loss`

### Evaluation / planning / rollout code
- `eval.py`
  - Hydra entrypoint: `run(cfg)`
  - Environment + planning/eval integration through `stable_worldmodel`.
  - Policy path:
    - If `policy != random`: loads model with `swm.policy.AutoCostModel(cfg.policy)`
    - Builds planner with `swm.PlanConfig(...)`, solver from `cfg.solver`, then `swm.policy.WorldModelPolicy(...)`
  - Executes rollout/eval: `world.evaluate_from_dataset(...)`
- `jepa.py::JEPA.rollout`
  - Internal latent rollout over action candidates for planning cost computation.
- `jepa.py::JEPA.get_cost`
  - Encodes goal, runs rollout, returns cost via `criterion`.

### Dataset loading pipeline
- Training data:
  - `train.py` uses `swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)`
  - Transform chain from `utils.py`:
    - image preprocessor: `get_img_preprocessor(...)`
    - per-column normalizer: `get_column_normalizer(...)`
  - Split and loaders:
    - `spt.data.random_split(...)`
    - `torch.utils.data.DataLoader(...)`
- Eval data:
  - `eval.py::get_dataset(cfg, dataset_name)`
  - `swm.data.HDF5Dataset(dataset_name, keys_to_cache=..., cache_dir=...)`

---

## 2) End-to-end information flow (with concrete paths/functions)

### How observations become latent states
1. `train.py::lejepa_forward` receives batch from HDF5 loader.
2. Calls `self.model.encode(batch)` where `self.model` is `JEPA`.
3. `jepa.py::JEPA.encode`:
   - reads `info['pixels']` shaped `(B, T, C, H, W)`
   - flattens time into batch via `einops.rearrange(..., "b t ... -> (b t) ...")`
   - runs ViT encoder (`self.encoder`)
   - selects CLS token
   - applies projector MLP
   - reshapes back to `(B, T, D)` and stores as `info['emb']`.

### How actions drive next-step latent prediction
1. `jepa.py::JEPA.encode` also computes action embeddings when `"action" in info`:
   - `info['act_emb'] = self.action_encoder(info['action'])`.
2. In training (`train.py::lejepa_forward`):
   - context latents: `ctx_emb = emb[:, :ctx_len]`
   - context actions: `ctx_act = act_emb[:, :ctx_len]`
   - target latents: `tgt_emb = emb[:, n_preds:]`
   - prediction: `pred_emb = self.model.predict(ctx_emb, ctx_act)`.
3. `jepa.py::JEPA.predict`:
   - `self.predictor(emb, act_emb)` where predictor is `module.py::ARPredictor`
   - ARPredictor uses `Transformer(..., block_class=ConditionalBlock)`
   - each block uses AdaLN-zero conditioning from action embedding (`c`).

### Where planning / rollout / evaluation happens
- Planning/evaluation orchestration: `eval.py::run`.
- Model-based action evaluation internals:
  - `jepa.py::JEPA.get_cost`
  - `jepa.py::JEPA.rollout`
  - `jepa.py::JEPA.criterion`
- Dataset-driven evaluation execution: `world.evaluate_from_dataset(...)` in `eval.py`.

### Minimal-change probing head insertion points
Recommended low-friction options:
1. **Probe on encoder latents (`emb`)**
   - Hook in `train.py::lejepa_forward` right after `output = self.model.encode(batch)`.
   - Add optional `probe_head(output['emb'])` and auxiliary logging/loss.
   - Why: cleanest access to temporally aligned latent trajectories.
2. **Probe on predictor outputs (`pred_emb`)**
   - Same function, after `pred_emb = self.model.predict(...)`.
   - Useful for “runtime correctability” signals on predicted futures.
3. **Attach in JEPA module for reuse in training+planning**
   - Add optional member in `jepa.py::JEPA.__init__` (e.g., `self.probe = ...`).
   - Expose helper methods to score `emb`/`pred_emb` from both train loop and `get_cost` rollouts.

For your thesis direction (feasibility + runtime correction), option (2) is likely most directly relevant, with optional consistency term between probe outputs on `tgt_emb` vs `pred_emb`.

---

## 3) Baseline reproduction attempt (single GPU / CPU fallback)

I attempted the smallest feasible setup first (dependency check), then tried to provision the environment according to README.

### Commands executed
```bash
pwd; rg --files -g 'AGENTS.md'
rg --files | head -n 200
sed -n '1,260p' README.md
sed -n '1,260p' train.py
sed -n '1,320p' module.py
sed -n '1,320p' jepa.py
sed -n '1,320p' eval.py
sed -n '1,320p' utils.py
sed -n '1,260p' config/train/lewm.yaml
sed -n '1,220p' config/train/data/pusht.yaml
sed -n '1,240p' config/eval/pusht.yaml
python - <<'PY'
import importlib
mods=['torch','einops','hydra','lightning','stable_pretraining','stable_worldmodel']
for m in mods:
    try:
        importlib.import_module(m)
        print(m+': OK')
    except Exception as e:
        print(m+': FAIL -> '+repr(e))
PY
uv --version || python -m pip --version
uv venv --python=3.10 .venv && source .venv/bin/activate && uv pip install 'stable-worldmodel[train,env]'
```

### What succeeded
- Static repository inspection (all requested mapping items).
- Python package manager/tool availability check (`uv` exists).

### What failed
- Runtime reproduction is currently blocked before training/eval start:
  - Missing dependencies in the environment (`torch`, `hydra`, `lightning`, `stable_pretraining`, `stable_worldmodel`, etc.).
  - Installing dependencies failed due network/tunnel connectivity to PyPI:
    - `Failed to fetch https://pypi.org/simple/stable-worldmodel/`
    - `tunnel error: unsuccessful`

### Practical implication
- No executable baseline (train/eval) could be run in this environment at this time.
- This is an environment/dependency access blocker, not a code-level blocker discovered from static inspection.

---

## 4) Recommended next-step modification points (minimally invasive)

1. **Add a latent probe head in training loop**
   - File: `train.py`
   - Function: `lejepa_forward`
   - Insert after `output = self.model.encode(batch)`.
   - Add optional probe loss terms (e.g., feasibility / correction diagnostics) with configurable weight.

2. **Expose probe-ready interfaces in JEPA**
   - File: `jepa.py`
   - Class: `JEPA`
   - Add optional methods returning intermediate latent sequences (`emb`, `pred_emb`) in rollout mode for runtime correction experiments.

3. **Keep anti-collapse untouched; add probe as additive head**
   - File: `module.py`
   - Preserve `SIGReg` path exactly; avoid destabilizing core JEPA training objective.

4. **Planning-time intervention hook**
   - Files: `jepa.py` (`get_cost` / `rollout`) and possibly `eval.py` (policy construction path)
   - Add optional correction/penalty term on action candidates using probe outputs during MPC scoring.

5. **Config-level toggles only (no architectural redesign)**
   - Add Hydra flags under `config/train/lewm.yaml` and `config/eval/*.yaml` for:
     - `probe.enabled`
     - `probe.weight`
     - `probe.target`
     - `probe.in_rollout`

These points keep LeWorldModel as the latent dynamics substrate while enabling your physical-feasibility and runtime-correctability research thread.
