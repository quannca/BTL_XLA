# Pic2Word Reproduction: Composed Image Retrieval

This repository prepares the source code and saved experiment artifacts for a local reproduction of **Pic2Word: Mapping Images to
Prompts for Zero-Shot Composed Image Retrieval** (S. Saito, A. Kanezaki, S. Murase, ECCV 2022). Pic2Word maps a reference image into a token embedding that can be inserted into a text prompt, enabling composed image retrieval without task-specific training data.

Paper: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). The implementation is based on the [Google Research Pic2Word repository](https://github.com/google-research/composed_image_retrieval); see the included Apache-2.0 `LICENSE`. The local evaluation source contains documented changes from its tracked fork base; see `reports/SOURCE_CODE_DIFF_REVIEW.md` before interpreting metrics.

## Contents

- `src/`, `model/`: training, model, data-loading, and retrieval evaluation code.
- `third_party/open_clip/`: vendored OpenCLIP code, configurations, and its license.
- `data/`: dataset preparation instructions, COCO preparation script, and ImageNet-R query/gallery lists. Dataset images and annotations are not included.
- `results/`: unchanged logs, parameter snapshots, and CIRR Test1 prediction/report artifacts from the local experiments.
- `reports/`: source diff review, repository audit, and the preparation inventory.
- `modal_eval.py`: Modal workflow for the evaluation actions listed below.

## Environment

The dependency list is in `requirements.txt`. The provided shell setup scripts are `env.sh` and `setenv.sh`; `valprep.sh` is the ImageNet validation preparation script. The scripts target a Unix-like environment; `modal_eval.py` describes the Modal image/runtime used by the project. Exact local package, CUDA, and GPU versions were not recorded in the saved parameter files, so this release does not claim a fully pinned runtime.

The Modal commands below require Modal CLI authentication and the Modal volumes/data/checkpoint paths configured in `modal_eval.py`. From this repository root:

```bash
modal run modal_eval.py --action run-fashion --category dress
modal run modal_eval.py --action run-fashion --category shirt
modal run modal_eval.py --action run-fashion --category toptee
modal run modal_eval.py --action run-coco
modal run modal_eval.py --action run-cirr-validation
modal run modal_eval.py --action run-imgnet
modal run modal_eval.py --action run-cirr-test1
```

These are the actions exposed by the checked-in Modal wrapper. They are provided as workflow entry points; this release preparation did not run them.

## Checkpoint

The original project README links the Pic2Word model checkpoint on [Google Drive](https://drive.google.com/file/d/1IxRi2Cj81RxMu0ViT4q4nkfyjbSHm1dF/view?usp=sharing). Download it separately and place it at `checkpoints/pic2word_model.pt` for the Modal workflow (or pass its path using `--resume` for direct evaluation). The checkpoint is not included in this repository. Verify the source and terms before use; no checkpoint checksum was recorded with the saved experiments.

## Datasets

Download and prepare each dataset separately according to its official source and the instructions in [`data/README.md`](data/README.md):

- Fashion-IQ: annotations and images using the sources listed in `data/README.md`; the Modal workflow expects `data/fashion-iq/`.
- COCO: validation images and `instances_val2017.json`, then run `python data/coco/prepare_data.py` from the repository root to prepare evaluation inputs as described by the script and data guide.
- CIRR: images plus validation/test1 annotations and splits from the CIRR project; the Modal workflow expects `data/CIRR/`.
- ImageNet-R / ImageNet Real: follow the ImageNet-R source and `data/README.md`; place the ImageNet-R and real validation image trees under `data/imgnet/`. The query and target lists are included in `data/imgnet/`.

Raw images, downloaded annotations, and dataset archives are excluded. Benchmark terms and access conditions remain those of their respective dataset providers.

## Evaluation commands

For the prepared Modal workflow, use the seven commands in **Environment**. The category actions correspond to Fashion-IQ Dress, Shirt, and Toptee; the other actions run COCO, CIRR validation, ImageNet-R, and CIRR Test1. The wrapper passes the dataset-specific options and paths to `src/eval_retrieval.py`.

The underlying evaluator supports `--eval-mode` values `fashion`, `coco`, `cirr`, `cirr_test`, and `imgnet`; evaluation requires a valid checkpoint (`--resume`) and the dataset layout expected by `src/data.py`. The source code and Modal wrapper are the authoritative locations for those options and paths.

## Saved results

- Fashion-IQ: three category logs and three parameter snapshots.
- COCO: evaluation log and parameter snapshot.
- CIRR validation: evaluation log and parameter snapshot.
- ImageNet-R: evaluation log and parameter snapshot.
- CIRR Test1: four method prediction JSON files (`composed`, `image`, `text`, `mixture`), one summary CSV, and four report images. No Test1 log or parameter snapshot is included or claimed.

See [`results/README.md`](results/README.md) for the file map. CIRR Test1 artifacts are presented as saved predictions and reports; the repository audit records the limits of their run/evaluator provenance. The logs and parameter files are preserved byte-for-byte from the local repository; they can contain environment-specific paths.

## License and attribution

The project source is distributed under the included Apache License 2.0. Vendored OpenCLIP has a separate included license. Dataset and checkpoint access is governed by their own sources and terms.
