# GitHub release checklist

Prepared 2026-11-20 (Asia/Saigon). This inventory describes the staging tree only. Source and result artifacts were copied from the local repository without changing their contents; newly written README, ignore rules, and data guidance are release documentation.

## Readiness summary

- [x] Core source: `src/`, `model/`, Modal wrapper, environment scripts, and COCO preparation script.
- [x] OpenCLIP runtime files, tokenizer vocabulary, model configs, and third-party license included.
- [x] ImageNet-R query and target lists included; dataset images/archives and model weights excluded.
- [x] Fashion-IQ: 3 logs + 3 params; COCO: log + params; CIRR validation: log + params; ImageNet-R: log + params.
- [x] CIRR Test1: 4 method prediction JSONs, summary CSV, and 4 report images.
- [x] README links resolve to staged paths; results guide documents Test1 scope.
- [x] Dataset/checkpoint/archive patterns excluded; `results/` is not ignored.
- [x] License files included: project Apache-2.0 and `third_party/open_clip/LICENSE`.
- [ ] Final public-file review by repository owner is pending. CIRR Test1 provenance limits and absolute paths retained in historical logs/audit reports should be reviewed before publication.

## Copied and created files

| File | Bytes | Status |
|---|---:|---|
| `.gitignore` | 580 | Release documentation/configuration |
| `check_fashioniq_images.py` | 530 | Copied unchanged; source/dependency |
| `data/coco/prepare_data.py` | 4392 | Copied unchanged; source/dependency |
| `data/imgnet/imgnet_real_query.txt` | 568850 | Copied unchanged; source/dependency |
| `data/imgnet/imgnet_targets.txt` | 917826 | Copied unchanged; source/dependency |
| `data/README.md` | 3771 | Release documentation/configuration |
| `env.sh` | 127 | Copied unchanged; source/dependency |
| `LICENSE` | 11560 | Ready for owner review |
| `modal_eval.py` | 85641 | Copied unchanged; source/dependency |
| `model/clip.py` | 11105 | Copied unchanged; source/dependency |
| `model/model.py` | 25555 | Copied unchanged; source/dependency |
| `README.md` | 5708 | Release documentation/configuration |
| `reports/GITHUB_PREPARATION_PLAN.md` | 24651 | Copied audit report; review paths/provenance |
| `reports/GITHUB_REPRODUCTION_AUDIT.md` | 14635 | Copied audit report; review paths/provenance |
| `reports/SOURCE_CODE_DIFF_REVIEW.md` | 22915 | Copied audit report; review paths/provenance |
| `requirements.txt` | 821 | Copied unchanged; source/dependency |
| `results/cirr/cirr_test1/predictions/composed.json` | 4194030 | Copied unchanged; evidence artifact |
| `results/cirr/cirr_test1/predictions/image.json` | 4193730 | Copied unchanged; evidence artifact |
| `results/cirr/cirr_test1/predictions/mixture.json` | 4193033 | Copied unchanged; evidence artifact |
| `results/cirr/cirr_test1/predictions/text.json` | 4192534 | Copied unchanged; evidence artifact |
| `results/cirr/cirr_test1/report/cirr_test1_composed_official.png` | 59067 | Copied unchanged; evidence artifact |
| `results/cirr/cirr_test1/report/cirr_test1_image_official.png` | 55014 | Copied unchanged; evidence artifact |
| `results/cirr/cirr_test1/report/cirr_test1_mixture_official.png` | 54496 | Copied unchanged; evidence artifact |
| `results/cirr/cirr_test1/report/cirr_test1_predictions_summary.csv` | 320 | Copied unchanged; evidence artifact |
| `results/cirr/cirr_test1/report/cirr_test1_text_official.png` | 54244 | Copied unchanged; evidence artifact |
| `results/cirr/validation/logs/cirr_validation_out.log` | 9615 | Copied unchanged; evidence artifact |
| `results/cirr/validation/params/cirr_validation_params.txt` | 1250 | Copied unchanged; evidence artifact |
| `results/coco/logs/coco_evaluation.log` | 5756 | Copied unchanged; evidence artifact |
| `results/coco/params/coco_params.txt` | 1226 | Copied unchanged; evidence artifact |
| `results/fashion_iq/logs/Dress_out.log` | 21887 | Copied unchanged; evidence artifact |
| `results/fashion_iq/logs/Shirt_out.log` | 4770 | Copied unchanged; evidence artifact |
| `results/fashion_iq/logs/Toptee_out.log` | 4775 | Copied unchanged; evidence artifact |
| `results/fashion_iq/params/Dress_params.txt` | 1215 | Copied unchanged; evidence artifact |
| `results/fashion_iq/params/Shirt_params.txt` | 1215 | Copied unchanged; evidence artifact |
| `results/fashion_iq/params/Toptee_params.txt` | 1219 | Copied unchanged; evidence artifact |
| `results/imagenet_r/logs/imgnet_official_out.log` | 9684 | Copied unchanged; evidence artifact |
| `results/imagenet_r/params/imgnet_official_params.txt` | 1240 | Copied unchanged; evidence artifact |
| `results/README.md` | 1069 | Copied unchanged; evidence artifact |
| `setenv.sh` | 189 | Copied unchanged; source/dependency |
| `src/data.py` | 26409 | Copied unchanged; source/dependency |
| `src/demo.py` | 11889 | Copied unchanged; source/dependency |
| `src/eval_retrieval.py` | 15194 | Copied unchanged; source/dependency |
| `src/eval_utils.py` | 36149 | Copied unchanged; source/dependency |
| `src/logger.py` | 3181 | Copied unchanged; source/dependency |
| `src/main.py` | 14294 | Copied unchanged; source/dependency |
| `src/params.py` | 10943 | Copied unchanged; source/dependency |
| `src/trainer.py` | 9518 | Copied unchanged; source/dependency |
| `src/utils.py` | 1826 | Copied unchanged; source/dependency |
| `third_party/open_clip/bpe_simple_vocab_16e6.txt.gz` | 1356917 | Copied unchanged; source/dependency |
| `third_party/open_clip/clip.py` | 9611 | Copied unchanged; source/dependency |
| `third_party/open_clip/environment.yml` | 4708 | Copied unchanged; source/dependency |
| `third_party/open_clip/LICENSE` | 1252 | Copied unchanged; source/dependency |
| `third_party/open_clip/model.py` | 17808 | Copied unchanged; source/dependency |
| `third_party/open_clip/model_configs/RN101.json` | 337 | Copied unchanged; source/dependency |
| `third_party/open_clip/model_configs/RN50.json` | 337 | Copied unchanged; source/dependency |
| `third_party/open_clip/model_configs/RN50_a2.json` | 411 | Copied unchanged; source/dependency |
| `third_party/open_clip/model_configs/RN50_a2s.json` | 410 | Copied unchanged; source/dependency |
| `third_party/open_clip/model_configs/RN50x16.json` | 338 | Copied unchanged; source/dependency |
| `third_party/open_clip/model_configs/RN50x4.json` | 338 | Copied unchanged; source/dependency |
| `third_party/open_clip/model_configs/ViT-B-16.json` | 282 | Copied unchanged; source/dependency |
| `third_party/open_clip/model_configs/ViT-B-32.json` | 282 | Copied unchanged; source/dependency |
| `third_party/open_clip/scheduler.py` | 675 | Copied unchanged; source/dependency |
| `third_party/open_clip/simple_tokenizer.py` | 4762 | Copied unchanged; source/dependency |
| `valprep.sh` | 2271000 | Copied unchanged; source/dependency |
| `GITHUB_RELEASE_CHECKLIST.md` | generated below | Release inventory |

## Exclusion and secret review

The staged tree was assembled from an explicit allowlist. No dataset image tree, dataset archive, checkpoint, virtual environment, or cache was copied. The two ImageNet-R list files are text benchmark inputs, not image data. A credential-pattern scan is recorded after generation; source references to generic tokenizer/API terms are not credentials. This check does not claim review of the original repository Git history.

The logs, parameter files, prediction JSON, CSV, and PNG files were copied without modification. No CIRR Test1 log or params were present in the source set, and none were fabricated.

## Verification results

- SHA-256 comparison: all copied source and result artifacts match their source files. The one expected difference is `data/README.md`, which has release-specific preparation guidance added above the original text.
- Text credential-pattern scan: 0 matches for common API key, GitHub token, AWS access key, or private-key formats.
- Local README links: 0 broken links.
- Size/exclusion scan: 0 files over 100 MiB; 0 checkpoint or dataset archive files. The only `.gz` file is OpenCLIP's required tokenizer vocabulary (small third-party source data, not a dataset archive).
- `.gitignore` rules were revised so `data/README.md`, `data/coco/prepare_data.py`, and `data/imgnet/*.txt` are allowed while other dataset contents stay excluded. Result images are not globally ignored.
