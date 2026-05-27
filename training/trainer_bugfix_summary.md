# Trainer bugfix summary (Step A)

Modified file:
- `training/trainer.py`

## Fixed issues

### 1) Scheduler step bug
**Problem:** scheduler objects were created but never stepped. Because warmup initialization effectively set LR to `base_lr / 5`, LR stayed almost constant around `2e-5`.

**Fix:**
- add explicit fields:
  - `self.base_lr`
  - `self.warmup_epochs`
- initialize optimizer LR to `base_lr / warmup_epochs`
- add helper `self._step_learning_rate(epoch)`
- call it once per epoch in `fit()`

**Result:** LR now changes across epochs.

---

### 2) Duplicate validation bug
**Problem:** `fit()` called `validate()` twice per epoch:
- once in `validate_every_n_epochs` branch
- once again unconditionally

This caused repeated threshold calibration, repeated validation cost, and confusing curves/checkpoints.

**Fix:**
- rewrite `fit()` so each epoch performs **at most one validation**
- all checkpointing / logging / early stopping now reuse that single `val_metrics`

---

### 3) `loss_dict` key mismatch bug
**Problem:** `PIPHMLoss` returns keys like:
- `disp_loss`
- `event_loss`
- `risk_loss`
- `creep_constr`

But `trainer.py` was reading old keys like:
- `L_disp`
- `L_event`
- `L_risk`
- `L_creep`

So TensorBoard component losses and audit printouts were wrong / zero.

**Fix:**
- use current keys as primary
- keep old keys as backward-compatible fallback

Examples:
- `disp_loss` fallback to `L_disp`
- `event_loss` fallback to `L_event`
- `risk_loss` fallback to `L_risk`
- `creep_constr` fallback to `L_creep`

Also expanded tracked losses to include:
- `loss_aux`
- `loss_event`
- `loss_causal`

---

### 4) Phase score / early stopping logic bug
**Problem:** `val_combined_score` changes formula across curriculum phases, so it is **not phase-invariant** and should not be used for global early stopping.

**Fix:**
- keep `val_combined_score` for phase-aware analysis / plotting
- add `val_monitor_score = val_score_multi`
- use `val_monitor_score` for:
  - best score tracking
  - early stopping

This makes early stopping consistent across epochs.

---

## Additional cleanup
- phase transition logging is now aligned with `curriculum_stages`
- removed dependence on hardcoded `30/80/150` phase milestones in the main training loop
- add TensorBoard metric:
  - `val/monitor_score`
  - `val/event_prauc`
  - `val/strict_fpr`
  - `val/recall_at_calibrated`

## New helper methods added
- `_get_phase_boundary_epochs()`
- `_step_learning_rate(epoch)`

## Verification performed
I verified in Arena with a smoke test (`PatchTST-only`, 2 epochs):

1. `python -m py_compile training/trainer.py` passes.
2. Training runs successfully end-to-end.
3. Audit losses are no longer zero:
   - `Loss - Disp: 0.2917`
   - `Event: 13.4603`
   - `Risk: 10.0845`
4. Validation happens once per epoch.
5. LR is no longer flat:
   - epoch 0 log shows `LR: 4.00e-05`
   - previously it stayed near `2.00e-05`

## Note
Not fixed in this patch (non-blocking for Step A):
- `data/dataset.py` has a `FutureWarning` about `int(future_labels.max())`
- validation denormalization path is slow because it builds tensors from Python lists
