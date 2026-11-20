# GitHub Reproduction Audit: Pic2Word

**Audit date:** 2026-11-20 (Asia/Saigon)  
**Repository:** `E:\btl_xla\composed_image_retrieval`  
**Conclusion:** **NOT READY** as a self-contained GitHub proof of all benchmark runs. Fashion-IQ and COCO have detailed local logs and configs; ImageNet-R has a completed local run; CIRR validation has a successful run but an earlier failure, while CIRR Test1 artifacts lack the run log and params needed to verify provenance. Most evidence is untracked and will not be present in GitHub unless deliberately committed.

This was a read-only source/data/results/security audit. No benchmark was run, and no existing source, dataset, checkpoint, or result file was changed. This report is the only file created by this audit.

## I. Repository overview

The repository contains the core implementation in `src/` (data loaders, training/evaluation entry points, metrics, parameters, logging and utilities), model code in `model/`, vendored OpenCLIP components in `third_party/open_clip/`, dataset preparation docs/scripts in `data/`, and a Modal workflow in `modal_eval.py`. `results/` contains 21 files (~17.1 MB). The local `data/` tree contains multiple benchmark datasets, and `checkpoints/pic2word_model.pt` is ~1.72 GB.

The checked-out branch is `main` at `84d6f68`, tracking `origin/main`; working tree has five modified tracked files (`modal_eval.py`, `src/data.py`, `src/eval_retrieval.py`, `src/eval_utils.py`, `src/params.py`) and numerous untracked files/directories, including `results/`, helper scripts, local annotation files, backups, and `PIC2WORD_KE_REPOSITORY_AUDIT.md`. No branch, commit, or remote was changed.

## II. Source code assessment

### Present and usable

- Model and IM2TEXT implementation: `model/model.py`, `model/clip.py`.
- Training/evaluation: `src/main.py`, `src/trainer.py`, `src/eval_retrieval.py`.
- Dataset and metric code: `src/data.py`, `src/eval_utils.py`.
- Parameters, logging, and utility code: `src/params.py`, `src/logger.py`, `src/utils.py`.
- Dataset preparation references: `data/README.md`, `data/coco/prepare_data.py`.
- Modal actions include Fashion-IQ setup/evaluation, COCO, ImageNet-R, CIRR validation, and CIRR Test1 (`modal_eval.py`).

### Comparison to original

The configured upstream is `https://github.com/google-research/composed_image_retrieval`; `origin` is the user's fork (`quannca/composed_image_retrieval`). The upstream public tree contains the same main source layout and README guidance. The local checkout has no fetched `upstream/*` ref, so a complete Git object diff against current upstream could not be made. Compared with the available fork base `origin/main`, five tracked files differ. The additions include device handling and Fashion-IQ/CIRR data/evaluation adaptations, a CIRR Test1 prediction path, Modal orchestration, and a `--distributed` parser option. These can affect evaluation behavior (not only deployment): dataset loading, device selection, feature extraction, reference filtering, and prediction output have been changed. Keep these changes reviewable and document their rationale/results before presenting them as equivalent to upstream evaluation.

`modal_eval.py` is a large (~85 KB) orchestration file with data preparation, validation, and evaluation actions. Its Modal volume names and paths are hard-coded; reproduce instructions should specify the expected Modal volumes, mount layout, and the exact action/arguments. The local README is essentially the upstream README and does not explain this workflow.

### Reproduction risks / missing code documentation

- `requirements.txt` lists many old exact pins and does not pin the full runtime, Modal, or a CUDA/PyTorch build; there is no lockfile or tested environment manifest.
- README examples reference `env3`, while the local ignored environment is `.venv_coco`; no exact environment capture is committed.
- Evaluation parameters save a useful common subset (ViT-L/14, IM2TEXT 512/2 layers, batch 64, AMP, one worker, GPU field, resume path), but do not record GPU model/VRAM, CUDA/PyTorch versions, dataset manifest/hash, or an exact invocation for each experiment. Some resume paths are Modal container paths, not portable checkpoint identifiers.
- The original README's evaluation snippets have malformed shell continuation around `--gpu`, and omit exact dataset preparation and checkpoint checksum instructions.
- `src/eval_retrieval.py` and `src/eval_utils.py` have material local diffs. The metrics/results must be attributed to the modified implementation in this checkout, not described as an untouched upstream run.

## III. Benchmark evidence

| Benchmark | Dataset | Checkpoint/config | Logs | Params | Metrics / artifacts | State |
|---|---|---|---|---|---|---|
| Fashion-IQ | Local `data/fashion_iq/` images and annotations; logs report Dress 2,017 queries/3,817 gallery, Shirt 2,038/6,346, Toptee 1,961/5,373 | `pic2word_model.pt`; params say ViT-L/14, IM2TEXT 512 x 2, batch 64, AMP, one worker; GPU None | Present: `Dress_out.log`, `Shirt_out.log`, `Toptee_out.log`; Dress log contains earlier failures before successful metrics | Three category params files | Four feature modes (`composed`, `image`, `text`, `mixture`), R@1/5/10/50/100 for each category | **Completed locally with evidence**; results are untracked |
| COCO | Local `data/coco/`; log completes evaluation; no explicit sample count found in params/log summary | `pic2word_model.pt`; ViT-L/14, IM2TEXT 512 x 2, batch 64, AMP, GPU 0 | `coco_evaluation.log` | `coco_params.txt` | composed/image/text/mixture; both `image_to_ref` and `ref_to_image`; Mean Rank, Median Rank, R@1/5/10/50/100 | **Completed locally with evidence**; results are untracked |
| CIRR validation | Local `data/CIRR/`; successful metrics correspond to 4,181 queries and 2,297 gallery images (failure before fix records a reshape error) | `pic2word_model.pt`; ViT-L/14, IM2TEXT 512 x 2, batch 64, AMP, GPU 0 | `cirr_validation_out.log` includes an initial failure and a later successful run | `cirr_validation_params.txt` | composed/image/text/mixture, recall_R@1/5/10/50/100 | **Validation completed locally with evidence**, but must report the failed attempt as history; results are untracked |
| CIRR Test1 | Prediction JSONs exist for composed/image/text/mixture; summary CSV and four official-looking PNGs exist | No Test1-specific params or checkpoint identity record found | No Test1 evaluation log found | No Test1 params found | JSONs contain predictions; summary CSV is present, but its provenance and official evaluation command are not recorded here | **Artifacts found; execution/config not independently verifiable** |
| ImageNet-R | `data/imgnet/` present; `imgnet_real_10000.zip` is ~1.35 GB; local ImageNet data tree ~5.71 GB | `pic2word_model.pt`; ViT-L/14, IM2TEXT 512 x 2, batch 64, AMP, GPU 0 | `imgnet_official_out.log` | `imgnet_official_params.txt` | Log has Real2Sketch metrics in four feature modes and repeated evaluation sections | **Completed locally with evidence**; clarify repeated sections/dataset sample counts in documentation; results are untracked |

COCO metrics include both directions and the requested ranks 1, 5, 10, 50 and 100 for all four feature modes. Fashion-IQ logs contain all four modes and the same recall cutoffs, but not mean/median rank. The user-specified COCO requirements are met by the local log. ImageNet-R is a completed evaluation in the checked-out code; this updates the initial plan's “needs checking” status based on actual files. No benchmark was inferred complete from configuration alone.

## IV. Evidence for a reviewer

Currently available locally: parameter snapshots, timestamped logs, checkpoint load messages (epoch 30), benchmark metrics, CIRR Test1 prediction JSONs, a CIRR summary CSV and PNGs. This is useful evidence that runs occurred, but it is not yet reproducible from the remote repository because these artifacts are untracked and the heavyweight checkpoint/data are ignored.

Missing or weak: commit-visible results, per-run exact command and software/GPU versions, dataset counts/manifests, checkpoint filename/hash and acquisition note, CIRR Test1 execution log/config and evaluator provenance, a reference table against paper-reported numbers, and an explanation of local metric/data-loader changes. Logs alone cannot prove the Modal environment or bind a run cryptographically to the present source revision.

## V. README and documentation

Present: paper link/citation, general training/test data pointers, Google Drive checkpoint link, broad training/evaluation examples, demo instructions, Apache license context.

Missing or inadequate: project-specific reproduction scope; environment with exact versions; dataset acquisition/layout and sample validation; checkpoint identity/hash; commands matching actual Modal actions; results table and honest comparison to the paper; explanation of custom code changes; list of which benchmarks did/did not finish; CIRR Test1 provenance; artifact locations; repository structure for added scripts; and a limitations section. The current README is substantially the upstream README and does not document the local experiment. A replacement/extension should have the ten requested sections (overview, original sources, environment, datasets, checkpoint, reproduction, results, comparison, structure, references), while explicitly marking CIRR Test1 as unverified until its run record is recovered.

## VI. GitHub readiness and security

### Safe/appropriate to include after review

Core source, documentation, `requirements.txt`, licenses, small reproducibility scripts, and compact text/CSV result summaries/params can be committed after checking paths, provenance, and any personal/container path details. CIRR prediction JSONs are ~4 MB each and may be publishable if dataset/license rules permit, but are not required by default; keep or publish them deliberately.

### Keep out of normal Git history

Datasets and archives (`data/`, `imgnet_real_10000.zip`, Fashion-IQ zip), the ~1.72 GB `.pt` checkpoint, virtual environment `.venv_coco/`, caches, generated images, and bulky intermediate artifacts. `.gitignore` already excludes `data/`, `*.zip`, common images, `checkpoints/`, model weights, Python caches and `.venv*`; check ignore behavior before staging. Do not assume ignored/untracked means safe if someone force-adds these assets.

Current evidence files under `results/` are **untracked**, not ignored, so they will not be included by a normal commit unless specifically added. The local-only files include personal/helper scripts, backups, annotation/URL lists, notebook, and `data_structure.txt`; review contents and necessity before adding. The `results/*_params.txt` files disclose container paths (`/root/project`, `/results`) and GPU IDs, not credentials, but should be accompanied by a note that these are run-environment paths.

A basic pattern scan returned only generic token-related code references in the CLIP tokenizer; no credential value was identified. This is not an external secret scanner or a scan of binary checkpoint/data archives. There is no basis in this audit to claim that all history or every ignored binary is secret-free. No API key/token was printed in this report. `origin` points to the user's GitHub fork and `upstream` is configured for Google Research; both were left untouched.

## VII. Work remaining

### P0 — before publishing

- **Repository root / `results/`** — Confirm which untracked evidence files should be versioned; stage only reviewed source, report, params/logs or compact summaries. Ensure datasets, ZIPs and checkpoint remain untracked/ignored. **Priority: P0.**
- **`modal_eval.py`, `src/eval_retrieval.py`, `src/eval_utils.py`, `src/data.py`** — Review and describe local changes that alter dataset/evaluation logic; cite the exact evaluation path used for reported results. **Priority: P0.**
- **`README.md`** — Correct incomplete shell commands and add exact runnable commands/Modal setup, environment, data/checkpoint preparation, and limitations. **Priority: P0.**
- **GitHub publication review** — Inspect all files intended for staging, especially helper scripts, `asin2url.*.txt`, notebook and any local paths. This audit found no credential-shaped source value, but did not prove the entire Git history is clear. **Priority: P0.**

### P1 — needed for strong reproduction evidence

- **`results/cirr/cirr_test1/` and `modal_eval.py`** — Preserve the Test1 run log, exact params/invocation, checkpoint identity/hash, and official evaluator output/provenance; until then describe predictions as artifacts found, not verified benchmark results. **Priority: P1.**
- **`results/*_params.txt`, logs, README** — Record GPU model/VRAM, CUDA/PyTorch/torchvision/Modal versions, dataset sample counts/manifest, code commit, exact command, and checkpoint hash for each run. Recover from original Modal run records where possible; do not reconstruct as fact if unavailable. **Priority: P1.**
- **`README.md` or `reports/`** — Add paper-versus-local results comparison with metric definitions, benchmark splits, feature modes, and explain differences. **Priority: P1.**
- **`results/`** — Ensure only intended, provenance-checked artifacts are tracked so a GitHub reviewer can actually access them. **Priority: P1.**

### P2 — polish

- **`requirements.txt` / environment docs** — Provide a tested, platform-appropriate environment specification or lockfile and explain GPU/PyTorch compatibility. **Priority: P2.**
- **`reports/` and project docs** — Add a short artifact index and file/benchmark map; clarify repeated ImageNet-R sections and the earlier failed Fashion-IQ/CIRR attempts. **Priority: P2.**
- **Repository root** — Remove nothing as part of this audit; before publication, decide whether old backups, local notebooks and one-off scripts belong in the public project. **Priority: P2.**

## VIII. Conclusion

**NOT READY** as currently represented on GitHub to prove the entire planned reproduction. Fashion-IQ, COCO, ImageNet-R, and CIRR validation each have local run evidence, but results are untracked and reproducibility metadata is incomplete. CIRR Test1 has predictions/reports but no local log or params proving how they were produced. The evaluation code has substantial local modifications that need documentation and review. The local checkout can support a transparent project report, but the GitHub repository must include a deliberate evidence set and clear limitations before it can substantiate all claimed experiments.

