# GitHub Preparation Plan

**Ngày kiểm kê:** 2026-11-20 (Asia/Saigon)  
**Repository:** `E:\btl_xla\composed_image_retrieval`  
**Phạm vi:** Kiểm kê trạng thái repo, mã nguồn, file evidence, dataset/checkpoint và tài liệu để lập kế hoạch đưa lên GitHub. Đây là báo cáo kế hoạch, không phải thao tác publish.

> Theo yêu cầu, không có code, dataset, checkpoint, log, params hay kết quả nào bị sửa, xóa, di chuyển, tải lên hoặc push. Báo cáo này là file mới duy nhất được tạo theo yêu cầu đầu ra. Không staging/commit.

## 1. Tổng quan repository

### Cấu trúc và Git

- Branch: `main`, tracking `origin/main` (`84d6f68`); remote `origin` là fork `quannca/composed_image_retrieval`; `upstream` được cấu hình tới Google Research. Không branch/remote/history nào bị thay đổi.
- Tracked hiện có: code gốc chính trong `src/`, `model/`, `third_party/open_clip/`, `data/README.md`, `data/coco/prepare_data.py`, hai danh sách ImageNet, Modal scripts/docs, README/license/requirements.
- Modified tracked: `modal_eval.py`, `src/data.py`, `src/eval_retrieval.py`, `src/eval_utils.py`, `src/params.py`.
- Untracked (không phải toàn bộ đều cần public): hai báo cáo audit trước đây; `results/`; scripts hỗ trợ Fashion-IQ; annotation/URL lists; notebook; bản backup source/Modal; thư mục cache `src/.cph/`.
- Ignored theo `.gitignore`: `data/`, `checkpoints/`, `.venv_coco/`, `*.zip`, ảnh phổ biến, `__pycache__`, `*.pt`, `*.pth`. Vì bị ignore, các file này không xuất hiện trong `git status` mặc định. Kích thước checkpoint và dataset bên dưới được lấy từ filesystem, không hàm ý file đã track.
- `results/` không bị ignore nên đang là thư mục untracked. Chúng không tự có trong GitHub nếu không được chủ ý thêm sau review.

### Nguồn và thành phần

`src/`: `data.py`, `eval_retrieval.py`, `eval_utils.py`, `params.py`, `logger.py`, `utils.py`, `main.py`, `trainer.py`, `demo.py`; có các backup untracked.  
`model/`: `model.py`, `clip.py` và một Modal backup root-level.  
`third_party/open_clip/`: CLIP/model/tokenizer/scheduler, configs, vocab và license.  
`data/`: README, COCO prep script, ImageNet query/target lists, cùng các dataset thực tế bị ignore.  
Root: `modal_eval.py`, `requirements.txt`, `env.sh`, `setenv.sh`, `valprep.sh`, README, license và các helper phát sinh trong quá trình audit/download.

## 2. Mã nguồn cần có trên GitHub

| File/thư mục hiện tại | Chức năng | Đưa lên GitHub? | Lý do / điều kiện |
|---|---|---:|---|
| `src/data.py` | Dataset/DataLoader, Fashion-IQ/CIRR và các input format | Có, sau review diff | Cần cho chạy benchmark; document layout Fashion-IQ category-first và các sửa đổi so với `origin/main`. |
| `src/eval_retrieval.py` | CLI entrypoint, load checkpoint, chọn dataset/evaluator | Có, sau review diff | Entry point chính để reproduce. |
| `src/eval_utils.py` | Trích feature và metrics COCO/Fashion/CIRR/ImageNet | Có, sau review diff | Logic đánh giá phải được public cùng kết quả; ghi rõ các thay đổi CIRR. |
| `src/params.py` | CLI/default parameters | Có | Tái lập invocation; giải thích `--distributed`. |
| `src/logger.py`, `src/utils.py` | Logging, distributed/device/helpers/transforms | Có | Dependencies nội bộ của source. |
| `src/main.py`, `src/trainer.py` | Training pipeline | Có | Giữ nếu repository tuyên bố gồm training; nếu chỉ inference thì vẫn là upstream source, không xóa nhưng có thể phân biệt training khỏi eval docs. |
| `src/demo.py` | Demo retrieval | Có | Thành phần project upstream; không bắt buộc cho benchmark nhưng hữu ích cho đầy đủ source. |
| `model/model.py`, `model/clip.py` | Model Pic2Word/CLIP, IM2TEXT, image/text encode | Có | Cần khởi tạo kiến trúc và load checkpoint. |
| `third_party/open_clip/**` | Vendor OpenCLIP, config/tokenizer/vocabulary | Có | Mã chạy phụ thuộc; giữ license và attribution. |
| `modal_eval.py` | Modal app, volume setup, audit/run functions | Có nếu public workflow Modal | Cần khi muốn người khác tái tạo chính workflow trên Modal. Mô tả mounts, volume names, GPU, command, output paths và hành vi dọn output tạm. |
| `data/README.md` | Dataset preparation docs upstream | Có | Giữ hướng dẫn benchmark/dữ liệu. Cần bổ sung cách cài file theo đúng local code nếu hiện thiếu. |
| `data/coco/prepare_data.py` | Chuẩn bị/biến đổi COCO eval inputs | Có | Script nhỏ, cần để tái lập data prep. |
| `data/imgnet/imgnet_real_query.txt`, `data/imgnet/imgnet_targets.txt` | Danh sách query/gallery ImageNet-R evaluation | Có sau license/privacy/path review | Danh sách input benchmark là cấu hình tái lập thiết yếu, file nhỏ và tracked sẵn. Kiểm tra không chứa absolute/local paths. |
| `valprep.sh` | ImageNet validation preparation | Có sau review | Script benchmark/data prep; giải thích mục đích và nguồn dữ liệu. |
| `requirements.txt`, `env.sh`, `setenv.sh` | Dependencies/environment scripts | Có sau cập nhật tài liệu | Hiện requirements có nhiều pin cũ và chưa khóa đầy đủ Modal/CUDA/PyTorch runtime. Không gọi là environment đã được đảm bảo tái lập nếu chưa ghi versions thực tế. |
| `README.md`, `LICENSE`, `CONTRIBUTING.md` | Docs, license, contribution | Có | README hiện chủ yếu là upstream README; cần mô tả local workflow và evidence. |
| `audit_fashion_annotations.py` | Audit annotation local | Có thể, nếu dùng như bước tái lập | Inspect source/dependencies/output path trước khi public; khác helper one-off thì đưa vào `scripts/` chỉ trong kế hoạch tương lai. |
| `check_fashioniq_images.py` | Check ảnh Fashion-IQ | Có thể | Giữ nếu được dùng/được README tham chiếu; nếu chỉ kiểm tra một lần, đưa vào supplemental audit chứ không phải required runtime. |
| `compare_fashioniq_sources.py` | So sánh nguồn ảnh Fashion-IQ | Có thể | Có ích provenance nếu docs mô tả source recovery; cần nói rõ inputs/outputs. |
| `download_fashioniq_val.py` | Download ảnh validation | Có thể, sau kiểm tra nguồn/điều khoản | Chỉ public nếu link/source và cách tải hợp lệ, không chứa credentials; README phải nêu đây là script download. |
| `export_fashioniq_modal_inventory.py` | Xuất inventory cho Modal | Có thể | Public nếu cần tạo manifest reproducibility; kiểm tra không ghi đường dẫn riêng tư. |
| `fashion_iq_audit.py` | Audit dataset Fashion-IQ | Có thể | Giữ nếu là quy trình kiểm tra chuẩn; cần xác nhận dependencies và phạm vi. |
| `prepare_fashion_eval_fix.py` | Helper sửa/chuẩn bị Fashion evaluation | Chưa quyết định | Tên gợi ý có thể mutate input; phải đọc toàn bộ và mô tả tác động trước khi public/chạy. Không tự chạy hoặc sửa file trong audit này. |
| `verify_fashioniq_recovery.py` | Xác nhận phục hồi Fashion-IQ | Có thể, sau review | Giữ nếu cần tái lập nguồn ảnh; ghi input/output và tiêu chí. |
| `modal_eval_backup.py`, `modal_eval.py.backup_before_modal_setup`, `modal_eval_before_image_test.py` | Bản cũ của script Modal | Không đưa theo mặc định | Dễ gây nhầm script chính; giữ archive cục bộ cho lịch sử làm việc, không xóa. Chỉ public nếu cần audit trail có giải thích. |
| `src/data.py.backup`, `src/data.py.backup_before_fix`, `src/eval_utils.py.before_fashion_fix` | Bản source cũ | Không đưa theo mặc định | Dễ bị import/nhầm với source hiện hành; report diff là tài liệu dễ hiểu hơn. Lưu cục bộ nếu cần recovery; không xóa. |

### Báo cáo đã có

| File | Trạng thái | Đề xuất |
|---|---|---|
| `PIC2WORD_KE_REPOSITORY_AUDIT.md` | Untracked, 25,904 bytes | Đọc/reconcile nội dung trước khi public; đây là audit cũ, không thay cho evidence index. |
| `reports/GITHUB_REPRODUCTION_AUDIT.md` | Untracked, 14,635 bytes | Có thể public như audit report nếu đồng nhất kết luận và artifacts hiện tại. |
| `reports/SOURCE_CODE_DIFF_REVIEW.md` | Untracked, 22,915 bytes | Có thể public; ghi rõ base `origin/main` và review date. |
| `GITHUB_PREPARATION_PLAN.md` | Được tạo bởi nhiệm vụ này | Có thể public làm inventory/plan sau khi rà soát lại link/path. |

## 3. Bằng chứng thực nghiệm hiện có

`results/` có 21 file (~17.06 MB), tất cả untracked. Bảng dưới liệt kê đầy đủ từng file đã tìm thấy; đề xuất “có” nghĩa là artifact cục bộ phù hợp để cân nhắc thêm, sau khi kiểm tra privacy/provenance và nội dung, không tự động staging.

### Fashion-IQ

| File | Nội dung | Có thể đưa? | Vai trò |
|---|---|---:|---|
| `results/fashion_iq/Dress_out.log` | Log; gồm các lần lỗi trước và lần thành công với 4 feature modes | Có, sau path/privacy review | Bằng chứng quá trình chạy và metrics; giữ cả lỗi lịch sử để minh bạch. |
| `results/fashion_iq/Dress_params.txt` | Config Dress | Có, sau bỏ/giải thích container paths | Cấu hình run (ViT-L/14, IM2TEXT 512/2, batch 64, AMP, gpu None). |
| `results/fashion_iq/Shirt_out.log` | Log metrics Shirt | Có | Log/metrics. |
| `results/fashion_iq/Shirt_params.txt` | Config Shirt | Có, sau path review | Cấu hình run. |
| `results/fashion_iq/Toptee_out.log` | Log metrics Toptee | Có | Log/metrics. |
| `results/fashion_iq/Toptee_params.txt` | Config Toptee | Có, sau path review | Cấu hình run. |
| `results/fashion_iq/*` category tables | Metrics R@1/5/10/50/100 cho composed/image/text/mixture | Đã có trong log; có thể thêm summary CSV nếu tính trực tiếp từ log, nhưng không tạo trong nhiệm vụ này | Người đọc có thể đối chiếu logs; không cần nhân bản số liệu nếu chưa có source artifact. |

### COCO

| File | Nội dung | Có thể đưa? | Vai trò |
|---|---|---:|---|
| `results/coco/coco_evaluation.log` | Log metric composed/image/text/mixture, image_to_ref và ref_to_image | Có | Bằng chứng đánh giá và metrics, gồm mean/median rank cùng R@1/5/10/50/100. |
| `results/coco/coco_params.txt` | Config COCO | Có, sau path review | Config (ViT-L/14, 512/2, batch 64, AMP, GPU 0). |
| `results/coco/` summary | Không thấy file summary riêng ngoài log/params | Cần bổ sung sau này nếu muốn bảng dễ đọc | README có thể trỏ trực tiếp log; không được suy diễn metric khác log. |

### CIRR validation và Test1

| File | Nội dung | Có thể đưa? | Vai trò |
|---|---|---:|---|
| `results/cirr/cirr_validation_out.log` | Log validation; có reshape failure trước metrics thành công cho 4 modes | Có | Bằng chứng lịch sử fail và successful validation metrics. |
| `results/cirr/cirr_validation_params.txt` | Config validation | Có, sau path review | Config (ViT-L/14, 512/2, batch 64, AMP, GPU 0). |
| `results/cirr/cirr_test1/predictions/composed.json` | CIRR Test1 submission predictions, ~4.19 MB | Có điều kiện | Prediction output; cần xác nhận license/schema và nguồn sinh. Không phải score/evaluation log. |
| `results/cirr/cirr_test1/predictions/image.json` | Predictions image baseline, ~4.19 MB | Có điều kiện | Prediction output baseline. |
| `results/cirr/cirr_test1/predictions/text.json` | Predictions text-only, ~4.19 MB | Có điều kiện | Prediction output baseline. |
| `results/cirr/cirr_test1/predictions/mixture.json` | Predictions mixture, ~4.19 MB | Có điều kiện | Prediction output baseline. |
| `results/cirr/cirr_test1/report/cirr_test1_predictions_summary.csv` | Summary: R1/R2/R5/R10/R50 cho 4 modes | Có điều kiện | Metrics summary nhưng thiếu local run log/params/evaluator provenance; phải ghi limitation và nguồn chấm. |
| `results/cirr/cirr_test1/report/cirr_test1_composed_official.png` | Hình kết quả composed | Có điều kiện | Hình minh họa; cần mô tả cách tạo và nguồn metric. |
| `results/cirr/cirr_test1/report/cirr_test1_image_official.png` | Hình kết quả image | Có điều kiện | Hình minh họa. |
| `results/cirr/cirr_test1/report/cirr_test1_text_official.png` | Hình kết quả text | Có điều kiện | Hình minh họa. |
| `results/cirr/cirr_test1/report/cirr_test1_mixture_official.png` | Hình kết quả mixture | Có điều kiện | Hình minh họa. |
| `results/cirr/cirr_test1/*params*`, `*log*` | Không tìm thấy trong thư mục hiện tại | Chưa có | Cần khôi phục từ run records nếu tồn tại; không dựng lại như thể là log đã lưu. |

### ImageNet-R

| File | Nội dung | Có thể đưa? | Vai trò |
|---|---|---:|---|
| `results/imgnet/imgnet_official_out.log` | Log nhiều khối metrics lặp cho 4 modes | Có, giữ nguyên | Run evidence; README cần giải thích đoạn nào tương ứng run/metric intended, không tự chọn một khối. |
| `results/imgnet/imgnet_official_params.txt` | Config ImageNet mode | Có, sau path review | Config (ViT-L/14, 512/2, batch 64, AMP, GPU 0). |
| Hình/CSV ImageNet-R riêng | Không tìm thấy trong `results/imgnet/` | Không áp dụng | Metrics hiện có trong log; chưa có summary riêng. |

**Đối chiếu trạng thái:** Fashion-IQ có ba category logs và params; COCO có log và params; CIRR validation có log/params, Test1 có prediction/report artifacts nhưng thiếu Test1 run log/params; ImageNet-R có log/params. Không kết luận Test1 predictions được tạo bởi chính revision/wrapper hiện tại nếu không có provenance.

## 4. File không nên đưa lên GitHub

| File/thư mục | Lý do | Cách thay thế |
|---|---|---|
| `data/` toàn bộ datasets/ảnh/archives (CIRR 4,623 files ~510 MB; COCO 9,775 ~2.76 GB; Fashion-IQ `fashion_iq` 15,866 ~225 MB và `fashion-iq` ~1.53 MB; ImageNet ~40,004 ~5.71 GB) | Dung lượng lớn; benchmark data có điều khoản phân phối riêng; không cần commit raw copies | Giữ `.gitignore`; `data/README.md` nêu nguồn tải, cấu trúc, split, checksum/count validation. Thư mục `data/imgnet/` có query/target txt đang tracked; kiểm tra chúng chỉ là file danh sách, không phải ảnh. |
| `checkpoints/pic2word_model.pt` (~1.72 GB) | Quá lớn cho source repo và provenance/license cần kiểm tra | README dẫn nguồn checkpoint upstream/Drive, filename, SHA-256; tải thủ công theo terms. Không upload. |
| `imgnet_real_10000.zip` (~1.35 GB) | Archive dataset cục bộ, GitHub unsuitable | Hướng dẫn tải/tạo từ nguồn hợp lệ; ghi checksum. |
| `fashion_iq_validation_check.zip` (~285 KB) | Archive local helper/check artifact, chưa thấy là prerequisite của official pipeline | Ghi nguồn/giải thích nếu cần; nếu là helper thì thay bằng script công khai và hướng dẫn, không commit archive mặc định. |
| `*.zip`, `*.tar`, các COCO annotation archives, `data/imgnet/imagenet-r.tar` (~2.19 GB) | Dữ liệu raw/intermediate | Hướng dẫn chuẩn bị và kiểm tra cấu trúc; giữ ignore. |
| `*.pt`, `*.pth`, checkpoint folder | Model weights nặng | Link/checksum/version; không upload bình thường. |
| `.venv_coco/`, `__pycache__/`, `src/.cph/`, `.vscode/` | Environment/cache/IDE state, không portable | Giữ ignore rules; ghi môi trường đã dùng. `.vscode` hiện không cần public trừ config dự án có chủ ý. |
| `data_structure.txt` (~3.75 MB) | Inventory/crawl dump rất lớn, không thay thế hướng dẫn dữ liệu | Tạo cấu trúc mẫu nhỏ trong README/data README; giữ cục bộ trừ khi chứng minh cần làm artifact. |
| `asin2url.dress.txt`, `asin2url.shirt.txt`, `asin2url.toptee.txt` (~1.47/2.38/2.02 MB) | Mapping URL/IDs từ quá trình thu thập; quyền riêng tư/nguồn và vai trò chưa xác nhận | Nếu cần tái lập download, công khai script + nguồn/cách lấy hợp lệ; kiểm tra không có dữ liệu cá nhân/token. Không đưa theo mặc định. |
| `cap.dress.val.json`, `split.dress.val.json` | Bản annotation cục bộ untracked, có thể chứa dữ liệu benchmark hoặc bản sửa | Dùng annotation chính thức theo hướng dẫn; nếu bản này là sửa local, lưu checksum/patch/provenance và mô tả trước khi public. |
| `dress_download_log.json` | Log helper download, untracked; có thể chứa URLs/IDs hoặc chi tiết cục bộ | Public chỉ sau kiểm tra nội dung/PII và khi cần provenance; nếu không, giữ local. |
| `test_fashion_image.jpg` | Ảnh test local; dữ liệu/asset không cần để chạy official pipeline | Dùng sample có quyền phân phối hoặc không commit. `.gitignore` đã bỏ ảnh. |
| `notebook9f3a5c75d1.ipynb` | Notebook local untracked có thể chứa paths/output/data; không phải entrypoint đã tài liệu hóa | Review/clear outputs và chuyển quy trình ổn định thành script nếu cần; không public mặc định. |
| `modal_eval_backup.py`, `modal_eval.py.backup_before_modal_setup`, `modal_eval_before_image_test.py`, `src/*.backup*`, `src/eval_utils.py.before_fashion_fix` | Backup/phiên bản cũ gây nhầm và nhân bản source | Lưu archive local hoặc tạo Git commits có mô tả nếu lịch sử cần; không đưa nhiều file backup vào repo chính. Không xóa theo nhiệm vụ này. |
| File tạm trong `res_cirr/` nếu xuất hiện khi chạy | Intermediate output, có thể stale | Ghi output chính thức vào result volume; wrapper hiện xóa thư mục tạm trước run. Thêm ignore nếu cần trong bước được ủy quyền sau này. |

Hiện `.gitignore` đã loại `data/`, checkpoint folder, `*.zip`, `*.tar` (cần xác nhận quy tắc tar; file `imagenet-r.tar` hiện ở data nên bị ignore bởi `data/`), images, caches và env. Không sửa `.gitignore` trong nhiệm vụ hiện tại.

## 5. Cấu trúc repository đề xuất

Đề xuất đích sau khi có nhiệm vụ triển khai riêng; không di chuyển file trong audit này:

```text
composed_image_retrieval/
├── src/                         # entrypoint, loaders, metrics, model utilities
├── model/                       # CLIP + Pic2Word / IM2TEXT
├── third_party/open_clip/       # vendored dependency + license
├── data/
│   ├── README.md                # nguồn, layout, split, lệnh chuẩn bị
│   ├── coco/prepare_data.py
│   └── imgnet/                   # chỉ danh sách nhỏ đã review; không chứa images
├── scripts/                     # helpers được chọn lọc sau khi review
├── reports/
│   ├── GITHUB_REPRODUCTION_AUDIT.md
│   ├── SOURCE_CODE_DIFF_REVIEW.md
│   └── GITHUB_PREPARATION_PLAN.md
├── results/                     # log, params, compact metrics, có provenance
│   ├── fashion_iq/
│   ├── coco/
│   ├── cirr/
│   └── imgnet/
├── modal_eval.py                 # optional Modal workflow
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

Không cần giữ dataset/checkpoint trong cây này. `results/` nên gồm artifacts được chọn có chủ đích; file Test1 predictions ~16.8 MB tổng riêng có thể giữ nếu terms cho phép và provenance được ghi, nếu không có thể chỉ giữ compact summary/metrics cộng link tới artifact kho khác. Không tạo cấu trúc mới trong nhiệm vụ này.

## 6. Nội dung cần bổ sung

1. README: project scope, link paper/upstream, lưu ý local changes; environment/modal image; lệnh setup; nguồn và layout từng dataset; checkpoint source/hash; exact command/function mỗi benchmark; đường dẫn logs/params/metrics; kết quả và giới hạn chưa xác minh.
2. Data docs: counts, split names, expected directories, list/annotation source, checksum/manifest, cách chạy data-prep scripts. Không đưa raw data lên GitHub.
3. Result index: từng benchmark ánh xạ tới log, params, metrics, prediction/report artifacts; chỉ rõ logs có failed attempts và các artifacts nào chưa truy nguyên. Hiện không có file results index riêng.
4. Environment: Python, PyTorch/torchvision/CUDA, Modal, GPU model/VRAM và các package thực tế; `requirements.txt` hiện không khóa được toàn bộ runtime.
5. Checkpoint: tên, nguồn, epoch 30 theo logs, checksum và hướng dẫn placement; path trong params hiện là `/root/project/checkpoints/pic2word_model.pt`, không portable.
6. CIRR Test1: exact invocation, config, checkpoint identity, log, evaluator official/version/source và provenance của summary CSV/PNGs. Hiện thiếu run log/params.
7. ImageNet-R: giải thích repeated metric blocks, xác định intended run từ record gốc, query/gallery count và protocol; không tự chọn metrics thay người dùng.
8. Script catalog: chỉ rõ helpers nào dùng để chuẩn bị/kiểm tra và script nào là one-off; trước đó cần review `prepare_fashion_eval_fix.py`, notebook và download helpers để hiểu side effects.
9. Result numbers: không nhập lại tay thành bảng mới nếu chưa trích chính xác từ log và xác định đúng run. Có thể dùng log/params đã lưu trực tiếp.

## 7. Kế hoạch triển khai ưu tiên

1. **Chốt nguồn sẽ public:** review năm modified tracked files và các helper untracked; phân biệt source cần thiết, script audit và bản backup. Không xóa file; chỉ lựa chọn tập file sau khi người sở hữu duyệt nội dung nếu cần.
2. **Chốt artifacts:** giữ đủ 21 files đang có theo benchmark; kiểm tra nội dung, privacy/path, provenance và license. Đưa log + params + compact metrics/report vào GitHub có chủ ý; không bỏ sót Test1 JSON/CSV/images nếu quyết định rằng chúng là evidence cần public.
3. **Xử lý gaps:** khôi phục CIRR Test1 run records/evaluator provenance nếu có; giữ trạng thái “unverified” nếu không có. Xác định ImageNet log groups từ run records.
4. **Viết docs:** README/data README và result index nêu command/Modal function, environment, data/checkpoint preparation, evidence map và giới hạn.
5. **Ignore review:** `.gitignore` đã ignore data/weights/archives/images/env; trước khi thay đổi phải đối chiếu `git check-ignore` cho từng large file và đảm bảo không có file dữ liệu đã bị track trước đó. Nhiệm vụ hiện tại không cập nhật file này.
6. **Kiểm tra trước commit:** xem `git status`, danh sách file sẽ stage, `git diff --cached --stat`, file lớn, paths, secret scan và licenses. Tránh force-add dataset/checkpoint/archives. Chỉ stage artifacts được chọn rõ ràng.
7. **Commit/push:** ngoài phạm vi hiện tại; chỉ thực hiện khi có yêu cầu riêng. Không commit/push ở đây.

## Tóm tắt

Repository hiện có đầy đủ các module source chính, Modal orchestration đã chỉnh sửa, dữ liệu benchmark cục bộ và checkpoint, cùng evidence logs/params/results cho ba Fashion-IQ categories, COCO, CIRR validation, ImageNet-R và CIRR Test1 prediction/report artifacts. Nhưng các source changes, helper scripts, reports và `results/` phần lớn modified/untracked; hiện GitHub branch không tự chứa evidence mới. Datasets, archives và checkpoint bị ignore và không nên commit.

Đủ điều kiện **để chuẩn bị một public source/evidence set sau review**, chưa đủ căn cứ để nói người khác có thể xác minh mọi run chỉ từ remote: CIRR Test1 thiếu log/params, ImageNet log cần disambiguate nhiều blocks, và configs thiếu environment/data/checkpoint hashes. Bước tiếp theo là review chọn lọc helper/annotation files, xác lập artifact/provenance policy, rồi cập nhật README và result index trước khi stage/commit. Không push GitHub trong nhiệm vụ này.
