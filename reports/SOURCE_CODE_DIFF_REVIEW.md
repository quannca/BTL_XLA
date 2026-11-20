# Review chi tiết diff 5 file source so với `origin/main`

**Ngày:** 2026-11-20 (Asia/Saigon)  
**Base:** `origin/main` tại thời điểm audit; working branch `main` là `84d6f68`.  
**Phạm vi:** `modal_eval.py`, `src/data.py`, `src/eval_retrieval.py`, `src/eval_utils.py`, `src/params.py`. Đọc diff và đối chiếu artifacts cục bộ. Không sửa source, Git refs, log, params hay dữ liệu; không chạy benchmark. Tệp review này là artifact mới duy nhất.

## Tóm tắt

| File | Nội dung thay đổi | Nhóm chính | Rủi ro với khả năng đối chiếu paper |
|---|---|---|---|
| `modal_eval.py` | Modal volumes, image dependencies, data audits, wrapper gọi eval entrypoint, lưu logs/results, CIRR Test1 orchestration | Triển khai Modal; chuẩn bị/kiểm tra dữ liệu; logging | Thấp đối với metric nếu wrapper chỉ truyền đúng tham số; trung bình vì kết quả còn phụ thuộc source hiện tại, mount/data và cấu hình runtime. Có hành động xóa thư mục `res_cirr` trong một Modal function. |
| `src/data.py` | Fashion-IQ resolve ảnh category-first, rồi flat root | Xử lý dữ liệu | Trung bình/cao: có thể chọn ảnh khác khi trùng basename giữa hai layout; cách chọn file có thể làm đổi gallery. |
| `src/eval_retrieval.py` | `torch.load` explicit `weights_only=False`; CIRR Test1 output directory creation; sửa experiment name interpolation; traceback được log và exception propagate | Checkpoint loading, output/logging | Thấp cho feature/metric nếu cùng checkpoint; rõ ràng hơn về lỗi. `weights_only=False` có rủi ro unpickle chỉ khi checkpoint không tin cậy. |
| `src/eval_utils.py` | Viết lại đáng kể `evaluate_cirr_test`, `evaluate_fashion`; chuyển device theo model; ép composition Pic2Word; thay CIRR validation reference filtering và Test1 output generation | Xử lý dữ liệu, trích đặc trưng, logic metric | Cao nhất: logic xếp hạng/loại reference và so sánh tên đổi; Test1 chỉ tạo prediction, không tính metric trong code này; có guard kiểm tra vô hiệu. |
| `src/params.py` | Thêm `--distributed` boolean | Tham số | Thấp cho các log đã lưu (đều `distributed: False`); ảnh hưởng nhánh DDP, không phải các lượt được ghi nhận. |

`origin/main` cung cấp baseline Git khả dụng. Trước đó đã xác nhận không có local `upstream/main` ref, nên review này không phải diff trực tiếp với revision mới nhất của Google Research. Các nhận định “thay đổi so với gốc” dưới đây nghĩa là thay đổi so với fork base `origin/main`, không khẳng định upstream hiện tại giống hệt base đó.

## Bảng thay đổi theo file

| File/vùng | Thay đổi so với `origin/main` | Phân loại | Lý do biểu hiện trong code | Tác động/rủi ro |
|---|---|---|---|---|
| `modal_eval.py` đầu file | Thêm `results_volume`, `cirr_volume`, `cirr_results_volume`; thêm `wandb`, `tensorboard` vào image | Chỉ triển khai Modal | Mount persistent data/results, thỏa import dependency không điều kiện trong entrypoint | Không đổi metric trực tiếp. Kết quả phụ thuộc volume chính xác và phiên bản image; cần ghi rõ Modal image/GPU/dependencies. |
| `modal_eval.py` `check_coco_on_modal`, audit Fashion-IQ/CIRR | Thêm chức năng kiểm tra file, schema, số lượng, annotation/path và ảnh đọc được | Xử lý/kiểm tra dữ liệu | Fail-fast trước khi tốn GPU; một số audit ghi report JSON vào volume | Audit chủ yếu read-only với input nhưng có ghi report vào volume. Không phải phần metric. Cần mô tả nếu dùng làm bước chuẩn bị; không coi audit là xác minh toàn bộ provenance dữ liệu. |
| `modal_eval.py` `run_*_official` | Thêm wrappers chạy `src/eval_retrieval.py` bằng subprocess và commit volume; cấu hình T4/CPU/RAM/timeout; params trong command như model/checkpoint/batch/precision | Triển khai Modal + logging | Chuẩn hóa invocation, persist artifacts và tách result dirs | Nếu command tương ứng chính là command đã chạy thì wrapper không tự viết lại metric. Tuy nhiên logs cục bộ không chứng minh wrapper cụ thể sinh ra chúng. Fashion-IQ params ghi `gpu: None`, COCO/CIRR/ImageNet ghi `gpu: 0`; đều không lưu tên GPU. Không thể gắn riêng mỗi log với wrapper chỉ qua params. |
| `modal_eval.py` ImageNet-R preflight | Đọc ImageList entries, kiểm tra label trong `[0,6999]`, file tồn tại; in counts | Kiểm tra dữ liệu | Bắt lỗi trước khi cấp GPU | Chỉ kiểm tra cấu trúc/giới hạn label được hard-code; không chứng minh danh sách đúng benchmark split chính thức. Không tác động metric sau khi dữ liệu hợp lệ. |
| `modal_eval.py` CIRR Test1 | Kiểm tra annotations/pairid, gọi `cirr_test`, copy predictions sang persistent dir và tạo summary artifacts | Triển khai Modal + xử lý dữ liệu/output | Tạo submission outputs có thể persist | Có `shutil.rmtree(project/res_cirr)` trong wrapper trước khi tạo output để tránh stale files; đó là xóa output tạm trong container. Phần này quan trọng khi người khác chạy. Artifact hiện có không kèm Test1 log/params nên chưa xác nhận wrapper này tạo ra artifacts đang lưu. |
| `src/data.py` `FashionIQ._resolve_image` | Trước: resolve tên ảnh tại `root_img` với thứ tự extension `IMAGE_EXTENSIONS`; sau: thử `root_img/<cloth>/<filename>` trước rồi mới `root_img/<filename>` | Xử lý dữ liệu | Tương thích cả layout theo category và layout flat | Có thể đổi ảnh được chọn nếu hai vị trí cùng tên tồn tại; extension precedence vẫn theo `IMAGE_EXTENSIONS` trong mỗi layout. Ảnh hưởng gallery/source và metric. Nếu chỉ một vị trí tồn tại thì chỉ đổi cách resolve, không đổi nội dung. Cần ghi layout, số ảnh và tính duy nhất basename; logs chỉ báo số lượng, không chứng minh không có collision. |
| `src/eval_retrieval.py` `load_model` | `torch.load(..., weights_only=False)`, map về CPU nếu `gpu=None`, map tới CUDA device nếu chỉ định | Checkpoint/model loading | Checkpoint cũ có metadata pickle (log Fashion-IQ ghi lỗi weights-only trước khi đổi) | Giúp đọc checkpoint có object metadata trên PyTorch mới. Không đổi kiến trúc/weights khi checkpoint tin cậy, nhưng explicit unpickle cho phép code execution từ checkpoint không tin cậy. Logs thành công ghi epoch 30, không ghi hash checkpoint. |
| `src/eval_retrieval.py` CIRR Test1 output | Tạo `res_cirr/` nếu cần, ghi JSON UTF-8, bỏ qua nếu results `None` | Logging/output | Tránh lỗi thư mục chưa tồn tại, output có encoding rõ | Không đổi thứ tự prediction. Đường dẫn vẫn tương đối với cwd; Modal wrapper cần copy đúng directory. |
| `src/eval_retrieval.py` experiment name | Sửa f-string để các tham số thực sự được nội suy vào tên mặc định | Logging | Tên run trước đây có thể chứa literal `{args.wd}`/`{args.aggregate}` | Chỉ ảnh hưởng tên log, không metric. Các params hiện dùng tên tường minh, nên sửa này không được thể hiện trong `name` tương ứng. |
| `src/eval_retrieval.py` exception handler | Thay `except: print('evaluation done')` bằng log traceback + raise | Logging/error handling | Làm lỗi hiện rõ và trả mã lỗi thất bại cho wrapper | Không đổi kết quả hợp lệ; giúp phân biệt lần fail/success. Dress log có nhiều lần fail trước run thành công; CIRR validation cũng có reshape fail trước lần thành công. |
| `src/eval_utils.py` `evaluate_cirr_test` | Viết lại loader/device handling, parse pairids robust hơn, bắt query/gallery rỗng, sinh bốn output mode | Dữ liệu/trích đặc trưng/output | Hỗ trợ DataLoader batch/collation và CIRR Test1 | Cách normalize embedding và composition nhìn chung giữ Pic2Word composition; xem chi tiết phía dưới. Một guard so sánh `len(all_ref_paths)` với chính biểu thức chứa `len(all_ref_paths)` nên vô hiệu; chỉ guard kế tiếp ref count vs pairid thực chất hữu dụng. |
| `src/eval_utils.py` composition CIRR Test1 | Xóa nhánh `args.eval_combiner`; luôn `img2text(image)` rồi chèn image token vào text qua `encode_text_img_retrieval` | Mô hình/trích đặc trưng | Ép dùng đường Pic2Word token mapping thay vì đường combiner khác | Nếu benchmark được chạy với Pic2Word `eval_combiner=False`, đường này giống ý tưởng baseline; nếu flag true thì thay đổi semantics và kết quả. Params CIRR Test1 hiện không tồn tại, nên không xác minh được flag/config đã dùng. Không tuyên bố tương đương chỉ từ comment. |
| `src/eval_utils.py` `evaluate_fashion` | Đổi `.cuda(args.gpu)` thành `.to(device model)`; viết lại định dạng nhưng vẫn encode target gallery, reference image, caption, image token + text, mixture | Xử lý device + trích đặc trưng | Fashion log có `gpu: None` trong khi Modal model vẫn có thể ở CUDA; log cũ ghi mismatch CPU/CUDA | Chuyển device đúng về nguyên tắc; không đổi tensor values/metric nếu cùng model/device/precision. Trình tự/normalization tương đương nhìn bề mặt code nhưng chưa có numeric parity test (không chạy theo yêu cầu). `image_features` tính ở từng source batch nhưng không dùng kết quả; target gallery được encode ở vòng riêng. |
| `src/eval_utils.py` `get_metrics_cirr` | So sánh basename thay vì full path; tạo mask theo từng query và yêu cầu còn `gallery_size-1`; kiểm tra mỗi query có đúng một target | Metric/data naming | Sửa lỗi reshape khi path/filename format khác nhau; loại reference | Có thể thay kết quả nếu tên có cùng basename ở nhiều thư mục, hoặc basename normalization làm path khác nhau được xem là cùng ảnh. Nếu mỗi query có đúng một reference trong gallery và tên duy nhất thì ranking sau lọc tương ứng mục tiêu loại reference một lần. Log ghi lần đầu reshape fail rồi run thành công; không có bằng chứng trực tiếp bằng log rằng dữ liệu thỏa uniqueness guard hoặc score mới bằng cách tính paper/upstream. Đây là thay đổi metric có rủi ro cần mô tả. |
| `src/eval_utils.py` `get_cirr_testoutput` | Tách basename bỏ extension; lọc reference từng query, giữ top 50, pairid scalar/list robust, kiểm tra số lượng | Metric/submission | Sinh rc2 JSON predictions theo expected id/image naming | Bỏ extension là cần cho định dạng tên prediction khi annotation dùng stem. Tuy nhiên chỉ basename/stem matching có thể collision; cần kiểm tra unique IDs. Đây là prediction generation chứ không phải metric scoring. Summary CSV và PNG hiện có nhưng không có official evaluator log/input provenance. |
| `src/params.py` parse args | Thêm `--distributed` (mặc định False) | Tham số/runtime | `eval_utils`/model code đọc `args.distributed` trong nhánh DDP | Các params có sẵn đều `distributed: False`; do đó nhánh mới không áp dụng cho các run được ghi nhận. DDP true có thể đổi shard/aggregation nếu không cấu hình đồng bộ đúng; chưa có run được lưu để đánh giá nhánh đó. |

## Phân tích kỹ thuật và rủi ro đánh giá

### Fashion-IQ: chọn ảnh và metric

`src/data.py` đổi precedence: category subfolder giờ được ưu tiên trước flat image root. Đây là thay đổi đầu vào, không phải chỉ “tìm file cho tiện”. Cần kiểm tra bằng manifest rằng mỗi ID resolve đúng một ảnh và không có basename collision. Logs chỉ cho biết counts: Dress 2,017 captions / 3,817 images; Shirt 2,038 / 6,346; Toptee 1,961 / 5,373.

`evaluate_fashion` chuyển input tới device thực của model; các params ghi batch 64, AMP, `gpu: None`, ViT-L/14, IM2TEXT middle_dim 512 / 2 layers. Logs cho thấy Dress có lỗi CPU/CUDA trước khi thành công; Shirt/Toptee có metrics. Mỗi log báo composed/image/text/mixture R@1/5/10/50/100. Đây là đối chiếu hỗ trợ giả thuyết sửa device để chạy được, nhưng không có source revision/hash theo lần chạy hoặc parity test để chứng minh chỉ device đổi.

Metric Fashion-IQ (`get_metrics_fashion`) không có diff trong phần metric: vẫn xếp hạng cosine distance trên normalized features, so khớp answer filename và recall@k. Tuy vậy data resolution có thể làm gallery khác. Logs không ghi đầy đủ dataset manifest/hash.

### COCO

`evaluate_coco`/`get_metrics_coco` không đổi trong diff hiển thị. Params: ViT-L/14; IM2TEXT 512/2; batch 64; AMP; GPU 0; checkpoint path kết thúc `pic2word_model.pt`; `eval_mode=coco`. Log ghi đủ `composed/image/text/mixture`, cả `image_to_ref` và `ref_to_image`, Mean Rank, Median Rank, R@1/5/10/50/100. Vì metric core không sửa ở phần diff này, rủi ro trực tiếp từ 5 file thấp hơn CIRR; vẫn không có hash checkpoint/data, versions hoặc số query chính xác trong params/log. Không thể kết luận COCO output byte/numeric-identical với upstream.

### CIRR validation và Test1

Validation params: cùng ViT-L/14, 512/2, batch 64, AMP, GPU 0. Log có lỗi đầu tiên `cannot reshape array ... into shape (4181,2296)` rồi lần sau có metrics cho bốn feature mode. Thay đổi `get_metrics_cirr` và filename normalization trực tiếp liên quan tới đường lỗi này; do đó phải ghi nhận metrics là do phiên bản local sau thay đổi tạo ra.

Test1: `results/cirr/cirr_test1/` có bốn JSON, summary CSV và bốn PNG; CSV nêu các R@k nhưng artifacts không đi cùng `cirr_test1` log hoặc params. Không có chứng cứ trong thư mục rằng đúng revision/function/config nào sinh chúng hoặc evaluator official nào chấm. Không gọi đó là verified benchmark score. `evaluate_cirr_test` chỉ sinh top-50 predictions; một script/evaluator ngoài repository cần đánh giá submission.

### ImageNet-R

Params: ViT-L/14, 512/2, batch 64, AMP, GPU 0. Log có nhiều đoạn `Eval ...` cho bốn feature modes, gồm nhiều nhóm metrics khác nhau/thời điểm; cần xác định group nào là run kết thúc intended (và vì sao lặp lại) trước khi đưa một con số duy nhất vào paper. Các thay đổi review chính không sửa `evaluate_imgnet_retrieval` hoặc `get_metrics_imgnet`; Modal wrapper chỉ gọi entrypoint. Vẫn cần pin dataset list/counts, checkpoint hash và runtime versions. Params `eval_mode=imgnet` xác nhận chế độ code được gọi nhưng không đủ để suy ra đây là benchmark ImageNet-R paper protocol nếu không nêu rõ query/target lists và metric implementation.

## Thay đổi có thể làm số liệu khác phương pháp paper

1. **CIRR `get_metrics_cirr`: cao** — name matching/path stripping và masking được đổi; mục đích là loại reference đúng và xử lý path, nhưng basename collisions hoặc dữ liệu không duy nhất có thể đổi ranking. Ghi rõ local fix và split; không so trực tiếp như cùng implementation nếu chưa xác minh.
2. **Fashion-IQ image resolution: trung bình/cao** — category-first selection có thể đổi ảnh/gallery khi cả category folder và root chứa cùng tên. Cần manifest/hash hoặc kiểm tra collision read-only để khẳng định data giống nhau.
3. **CIRR Test1 output normalization: trung bình** — basename + stem normalization và lọc reference thay output IDs/order; kiểm tra against official submission schema/evaluator.
4. **CIRR Test1 composition hard-coded: trung bình** — bỏ nhánh combiner, luôn Pic2Word path. Đây là thay đổi semantics nếu `eval_combiner` được bật; params Test1 thiếu nên trạng thái không rõ.
5. **`torch.load(weights_only=False)`: rủi ro bảo mật, không phải metric** — checkpoint tin cậy hiện báo epoch 30; không có checksum để khóa identity.
6. **Device transfer in Fashion-IQ/CIRR: thấp có điều kiện** — nhằm đặt inputs trên device model. Không nên khẳng định numeric parity tuyệt đối khi chưa so sánh output hoặc cùng precision/runtime.
7. **Modal orchestration/CLI/logging/`--distributed`: thấp với evidence runs hiện có** — không đổi metric khi giữ đúng invocation và distributed false; wrappers và params không tự chứng minh artifacts được sinh bởi chính wrappers.

## File/log đối chiếu và mức độ chứng minh

| Benchmark | Evidence local | Params thực tế | Quan hệ với diff | Giới hạn |
|---|---|---|---|---|
| Fashion-IQ | `results/fashion_iq/{Dress,Shirt,Toptee}_out.log` | `results/fashion_iq/{Dress,Shirt,Toptee}_params.txt` | Log Dress có lỗi args, unpickle, device trước thành công; thành công dùng device-aware `evaluate_fashion`. `src/data.py` resolver mới có khả năng áp dụng. | Không có code commit hash, exact command, GPU model/hash dataset. Resolver behavior không được params mô tả. |
| COCO | `results/coco/coco_evaluation.log` | `results/coco/coco_params.txt` | COCO evaluator/metric không thuộc vùng diff metric; Modal wrapper là điều phối. | Không có sample count/config hash/checkpoint hash/runtime versions; chưa xác nhận wrapper cụ thể tạo log. |
| CIRR validation | `results/cirr/cirr_validation_out.log` | `results/cirr/cirr_validation_params.txt` | Log reshape fail rồi metrics thành công sau logic CIRR được sửa; basename/filter mới có thể quyết định metrics. | Không ghi exact source hash, nên không biết precisely code revision của từng attempt. |
| CIRR Test1 | `results/cirr/cirr_test1/predictions/*.json`, `report/*.csv`, `report/*.png` | Không tìm thấy Test1 params | Khớp với chức năng dự kiến của `evaluate_cirr_test`/Modal wrapper, nhưng không chứng minh là artifacts cùng lần chạy. | Không có run log, command/config/checkpoint ID/evaluator provenance. Không xác nhận Test1 hoàn thành có thể tái lập. |
| ImageNet-R | `results/imgnet/imgnet_official_out.log` | `results/imgnet/imgnet_official_params.txt` | Các metric ImageNet trong `eval_utils.py` không có diff; Modal wrapper gọi entrypoint. | Log lặp nhiều metric block; cần xác định kết quả nào dùng. Không có source hash/data manifest/checkpoint hash. |

## README cần mô tả file nào

README cần nêu đích danh các file sau để phân biệt baseline, custom changes, execution và evidence:

- `modal_eval.py`: Modal volumes, image/dependency setup, mount paths, T4 resources, action/function names, chính xác cách gọi mỗi benchmark, nơi lưu output, behavior xóa thư mục output tạm `res_cirr`, preflight validation và giới hạn khi artifacts được persist.
- `src/data.py`: Fashion-IQ supported layouts và thứ tự tìm file category-first; cách xác nhận split/gallery và annotation counts.
- `src/eval_retrieval.py`: CLI flags được dùng, checkpoint loading behavior, output `res_cirr`, log/exception behavior; cung cấp invocation tái lập cho Fashion-IQ/COCO/CIRR/ImageNet-R.
- `src/eval_utils.py`: Pic2Word composition path, auxiliary image/text/mixture modes, normalize/ranking details, CIRR filename/reference filtering, Test1 top-50 output và việc scoring cần evaluator ngoài. Đánh dấu rõ các sai khác so với `origin/main`.
- `src/params.py`: ý nghĩa `--distributed`; nêu các runs đã lưu đều `distributed=False`.
- `requirements.txt` và environment/Modal image specification: dependencies/versions thực tế. File này không nằm trong năm file diff, nhưng README cần tham chiếu vì hiện chưa khóa đầy đủ môi trường.
- `results/` artifact index hoặc report liên quan: ánh xạ từng benchmark tới log/params/metrics, chỉ rõ CIRR Test1 chưa có run record và ImageNet log có nhiều blocks. Hiện results untracked, README cần nói người đọc xem đúng artifacts nào sau khi chủ đích đưa chúng vào repo.
- `model/model.py`, `model/clip.py`: README nên trỏ tới implementation model/IM2TEXT nền tảng, nhưng diff hiện tại không sửa hai file này.

## Cách mô tả trung thực trong báo cáo khoa học

Đề xuất ngôn ngữ:

> “Chúng tôi sử dụng implementation Pic2Word từ repository Google Research làm baseline và thực hiện các thay đổi cục bộ để chạy trên Modal, hỗ trợ layout Fashion-IQ tại máy, xử lý device placement/checkpoint loading, và sửa CIRR filename/reference filtering cùng Test1 prediction export. Các kết quả được báo cáo được tạo từ source đã sửa trong repository này; chúng tôi không khẳng định evaluator là byte-for-byte/từng điểm số tương đương upstream. Chúng tôi lưu cấu hình và log cục bộ; CIRR Test1 prediction artifacts hiện chưa có run log/params để truy nguyên độc lập. Mọi so sánh với kết quả paper cần nêu rõ split, metric, preprocessing và khác biệt implementation.”

Không gọi các Modal wrappers là “official evaluation” theo nghĩa được Google Research chứng nhận; chúng chỉ gọi entrypoint hiện có trong fork. Tránh claim “reproduce exact paper results” trước khi khóa revision/data/checkpoint/runtime, giải thích ImageNet-R protocol, xác minh resolver/collision và CIRR scoring/output với evaluator chuẩn.

## Bằng chứng cần bổ sung để hạ rủi ro

- Per-run source revision/hash gắn với log; checkpoint SHA-256 và provenance.
- CUDA/PyTorch/torchvision/Modal/GPU model và exact command cho từng result.
- Dataset manifests/counts/split hashes; Fashion-IQ duplicate basename/resolution audit.
- CIRR validation before/after source revision liên kết được với lần chạy; giải thích path normalization và uniqueness.
- CIRR Test1 params/log/checkpoint/source ID và official evaluator command/version/score output.
- ImageNet-R log interpretation, exact query/target lists and intended metric protocol.
- Paper-vs-local table với metric units/definitions/splits, và đánh dấu các metric chưa xác minh.

