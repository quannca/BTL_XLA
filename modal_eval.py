import modal
import os

# --------------------------------------------------
# 0. KHAI BÁO APP VÀ MÔI TRƯỜNG
# --------------------------------------------------

app = modal.App("pic2word-evaluation")

fashion_volume = modal.Volume.from_name("pic2word-data")
results_volume = modal.Volume.from_name("pic2word-results", create_if_missing=True)
cirr_volume = modal.Volume.from_name("pic2word-cirr-data", create_if_missing=True)
cirr_results_volume = modal.Volume.from_name("pic2word-cirr-results", create_if_missing=True)

# Môi trường tải ảnh
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests",
        "Pillow",
    )
)

# Môi trường Pic2Word cho các tác vụ sau này
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        # Dependencies imported by src/data.py and eval_utils.py
        "braceexpand",
        "pandas",
        "pyarrow",
        "tqdm",
        # Tokenizer dependencies used by third_party/open_clip
        "ftfy",
        "regex",
        # Imported unconditionally by src/eval_retrieval.py
        "wandb",
        "tensorboard",
    )
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            "data/**",
            ".git/**",
            ".venv/**",
            "env3/**",
            "__pycache__/**",
        ],
    )
)


# --------------------------------------------------
# 1. KIỂM TRA CHECKPOINT
# --------------------------------------------------

@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=600,
)
def check_checkpoint():
    import os
    import torch

    checkpoint_path = (
        "/root/project/checkpoints/pic2word_model.pt"
    )

    print("PyTorch version:", torch.__version__)
    print("Checkpoint exists:", os.path.exists(checkpoint_path))

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    print(
        "Checkpoint size:",
        os.path.getsize(checkpoint_path) / (1024**3),
        "GB"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    print("\nCheckpoint keys:")
    print(list(checkpoint.keys()))

    required_keys = [
        "state_dict",
        "state_dict_img2text",
        "epoch",
    ]

    for key in required_keys:
        print(key, ":", key in checkpoint)

    print("\nCheckpoint inspection completed.")


# --------------------------------------------------
# 2. CHUẨN BỊ FASHION-IQ ANNOTATION
# --------------------------------------------------

@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/data": fashion_volume},
    cpu=1,
    memory=1024,
    timeout=300,
)
def prepare_fashion_iq():
    import zipfile
    from pathlib import Path

    zip_path = Path("/data/fashion_iq_validation.zip")
    target_dir = Path("/data/fashion-iq")

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Cannot find ZIP file: {zip_path}"
        )

    required_files = [
        "captions/cap.dress.val.json",
        "captions/cap.shirt.val.json",
        "captions/cap.toptee.val.json",
        "image_splits/split.dress.val.json",
        "image_splits/split.shirt.val.json",
        "image_splits/split.toptee.val.json",
    ]

    print("ZIP file:", zip_path)
    print("ZIP size:", zip_path.stat().st_size, "bytes")

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:

        normalized_members = {}

        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/")
            member_path = Path(normalized_name)

            if (
                member_path.is_absolute()
                or ".." in member_path.parts
            ):
                raise ValueError(
                    f"Unsafe ZIP path: {member.filename}"
                )

            if normalized_name in normalized_members:
                raise ValueError(
                    f"Duplicate ZIP path: {normalized_name}"
                )

            normalized_members[normalized_name] = member

        missing = [
            name for name in required_files
            if name not in normalized_members
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing files in ZIP: {missing}"
            )

        print("\nExtracting Fashion-IQ annotations:")

        for name, member in normalized_members.items():
            destination = (target_dir / name).resolve()

            if not destination.is_relative_to(target_dir.resolve()):
                raise ValueError(
                    f"Unsafe destination path: {name}"
                )

            if member.is_dir() or name.endswith("/"):
                destination.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                continue

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with archive.open(member) as source:
                with open(destination, "wb") as output:
                    output.write(source.read())

            print("Extracted:", name)

    print("\nChecking extracted files:")

    for name in required_files:
        file_path = target_dir / name

        if not file_path.is_file():
            raise FileNotFoundError(file_path)

        print(
            name,
            "-",
            round(file_path.stat().st_size / 1024, 2),
            "KB"
        )

    fashion_volume.commit()

    print("\nFashion-IQ annotation preparation completed.")
    print("Data directory:", target_dir)


# --------------------------------------------------
# 3. TẠO THƯ MỤC LƯU ẢNH FASHION-IQ
# --------------------------------------------------

@app.function(
    image=download_image,
    volumes={"/data": fashion_volume},
    cpu=1,
    memory=1024,
    timeout=300,
)
def create_fashion_image_dirs():
    from pathlib import Path

    base_dir = Path("/data/fashion-iq/images")

    for category in ["dress", "shirt", "toptee"]:
        folder = base_dir / category
        folder.mkdir(parents=True, exist_ok=True)
        print("Created:", folder)

    fashion_volume.commit()

    print("Fashion-IQ image directories created.")


# --------------------------------------------------
# 4. TẢI ẢNH VALIDATION THEO TỪNG DANH MỤC
# --------------------------------------------------

BATCH_SIZE = 100

# Chỉ tải Shirt và Toptee.
# Dress đã được tải trước đó nên không chạy lại.
CATEGORIES_TO_DOWNLOAD = ["shirt", "toptee"]


@app.function(
    image=download_image,
    volumes={"/data": fashion_volume},
    cpu=2,
    memory=2048,
    timeout=1800,
)
def download_fashion_batch(
    category: str,
    start_index: int,
    end_index: int,
):
    """
    Tải một batch ảnh validation cho một danh mục Fashion-IQ.

    - Hỗ trợ dress, shirt, toptee.
    - Bỏ qua ảnh đã tồn tại và hợp lệ.
    - Thử tải tối đa 3 lần.
    - Tối đa 5 luồng tải đồng thời.
    - Ghi log riêng cho từng danh mục.
    - Commit Volume sau mỗi batch.
    """

    import json
    import time
    import requests

    from io import BytesIO
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image

    allowed_categories = {
        "dress",
        "shirt",
        "toptee",
    }

    if category not in allowed_categories:
        raise ValueError(
            f"Unsupported category: {category}. "
            f"Expected one of {sorted(allowed_categories)}"
        )

    if start_index < 0 or end_index < start_index:
        raise ValueError(
            f"Invalid batch range: {start_index}:{end_index}"
        )

    base_dir = Path("/data/fashion-iq")

    split_candidates = [
        base_dir / f"split.{category}.val.json",
        base_dir / "image_splits" / f"split.{category}.val.json",
    ]

    split_path = next(
        (
            path for path in split_candidates
            if path.is_file()
        ),
        None,
    )

    if split_path is None:
        raise FileNotFoundError(
            "Cannot find validation split. Checked: "
            + ", ".join(str(path) for path in split_candidates)
        )

    metadata_path = (
        base_dir / f"asin2url.{category}.txt"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Cannot find metadata file: {metadata_path}"
        )

    output_dir = (
        base_dir / "images" / category
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        base_dir / f"{category}_download_log.json"
    )

    with open(split_path, encoding="utf-8") as f:
        image_ids = json.load(f)

    if not isinstance(image_ids, list):
        raise ValueError(
            f"Validation split is not a JSON list: {split_path}"
        )

    batch_ids = image_ids[start_index:end_index]

    print("Category:", category)
    print("Validation split:", split_path)
    print("Total validation IDs:", len(image_ids))
    print("Batch range:", f"{start_index}:{end_index}")
    print("Images in this batch:", len(batch_ids))

    if not batch_ids:
        print("No image IDs in this batch.")

        return {
            "category": category,
            "start": start_index,
            "end": end_index,
            "processed": 0,
            "success": 0,
            "failed": 0,
        }

    url_map = {}

    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) >= 2:
                url_map[parts[0]] = parts[1]

    print("Total metadata entries:", len(url_map))

    def is_valid_image(image_path):

        if not image_path.is_file():
            return False

        try:
            with Image.open(image_path) as img:
                img.verify()

            return True

        except Exception:
            image_path.unlink(missing_ok=True)
            return False

    def download_one(image_id):

        image_path = (
            output_dir / f"{image_id}.jpg"
        )

        if is_valid_image(image_path):
            return image_id, "already_exists"

        url = url_map.get(image_id)

        if not url:
            return image_id, "missing_url"

        for attempt in range(3):

            temp_path = image_path.with_suffix(".jpg.part")

            try:
                response = requests.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    },
                    timeout=30,
                )

                response.raise_for_status()

                with Image.open(
                    BytesIO(response.content)
                ) as img:
                    img.verify()

                temp_path.write_bytes(
                    response.content
                )

                temp_path.replace(image_path)

                return image_id, "success"

            except Exception as e:

                temp_path.unlink(missing_ok=True)

                if attempt == 2:
                    return (
                        image_id,
                        f"failed: {str(e)[:150]}"
                    )

                time.sleep(2 * (attempt + 1))

        return image_id, "failed"

    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:

        futures = [
            executor.submit(
                download_one,
                image_id,
            )
            for image_id in batch_ids
        ]

        for future in as_completed(futures):

            result = future.result()

            results.append(result)

            print(
                result[0],
                "->",
                result[1],
            )

    success = sum(
        status in (
            "success",
            "already_exists",
        )
        for _, status in results
    )

    failed = len(results) - success

    previous_log = {}

    if log_path.exists():

        try:
            with open(log_path, encoding="utf-8") as f:
                old_data = json.load(f)

            if isinstance(old_data, list):

                previous_log = {
                    item["image_id"]: item["status"]
                    for item in old_data
                    if (
                        "image_id" in item
                        and "status" in item
                    )
                }

        except Exception:
            print(
                "Warning: existing log unreadable; "
                "creating a new log."
            )

    for image_id, status in results:
        previous_log[image_id] = status

    with open(log_path, "w", encoding="utf-8") as f:

        json.dump(
            [
                {
                    "image_id": image_id,
                    "status": status,
                }
                for image_id, status in previous_log.items()
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    fashion_volume.commit()

    print("\nBatch summary:")
    print("Category:", category)
    print("Processed:", len(results))
    print("Successful or already existing:", success)
    print("Failed:", failed)
    print("Log saved:", log_path)
    print("Volume committed.")

    return {
        "category": category,
        "start": start_index,
        "end": end_index,
        "processed": len(results),
        "success": success,
        "failed": failed,
    }


# --------------------------------------------------
# 5. ĐỌC SỐ LƯỢNG ẢNH VALIDATION
# --------------------------------------------------

@app.function(
    image=download_image,
    volumes={"/data": fashion_volume},
    cpu=1,
    memory=1024,
    timeout=300,
)
def get_fashion_validation_count(category: str):

    import json
    from pathlib import Path

    if category not in {
        "dress",
        "shirt",
        "toptee",
    }:
        raise ValueError(
            f"Unsupported category: {category}"
        )

    base_dir = Path("/data/fashion-iq")

    split_candidates = [
        base_dir / f"split.{category}.val.json",
        base_dir / "image_splits" / f"split.{category}.val.json",
    ]

    split_path = next(
        (
            path for path in split_candidates
            if path.is_file()
        ),
        None,
    )

    if split_path is None:
        raise FileNotFoundError(
            "Cannot find validation split. Checked: "
            + ", ".join(str(path) for path in split_candidates)
        )

    with open(split_path, encoding="utf-8") as f:
        image_ids = json.load(f)

    return len(image_ids)


# --------------------------------------------------
# 6. KIỂM TRA NHANH SỐ LƯỢNG ẢNH FASHION-IQ
# --------------------------------------------------

@app.function(
    image=download_image,
    volumes={"/data": fashion_volume},
    cpu=2,
    memory=4096,
    timeout=1800,
)
def check_fashion_images():
    """
    Đối chiếu ID trong validation split với tên file .jpg hiện có.
    Hàm này chỉ kiểm tra sự tồn tại của file, không mở từng ảnh bằng Pillow.
    """

    import json
    from pathlib import Path

    categories = ["dress", "shirt", "toptee"]
    base_dir = Path("/data/fashion-iq")
    overall = {}

    for category in categories:
        print(
            f"\n========== BẮT ĐẦU KIỂM TRA {category.upper()} ==========",
            flush=True,
        )

        split_candidates = [
            base_dir / f"split.{category}.val.json",
            base_dir / "image_splits" / f"split.{category}.val.json",
        ]
        split_path = next(
            (p for p in split_candidates if p.is_file()),
            None,
        )

        if split_path is None:
            print(f"{category.upper()}: KHÔNG TÌM THẤY FILE SPLIT", flush=True)
            overall[category] = {"error": "missing_split"}
            continue

        with open(split_path, encoding="utf-8") as f:
            image_ids = json.load(f)

        if not isinstance(image_ids, list):
            raise ValueError(f"Validation split không phải JSON list: {split_path}")

        image_dir = base_dir / "images" / category

        if image_dir.is_dir():
            existing_ids = {
                path.stem
                for path in image_dir.glob("*.jpg")
                if path.is_file()
            }
        else:
            existing_ids = set()

        expected_ids = {str(image_id) for image_id in image_ids}
        present_ids = expected_ids & existing_ids
        missing_ids = expected_ids - existing_ids
        unexpected_ids = existing_ids - expected_ids

        result = {
            "validation_ids": len(image_ids),
            "present_images": len(present_ids),
            "missing_images": len(missing_ids),
            "unexpected_jpg_files": len(unexpected_ids),
        }
        overall[category] = result

        print(f"Validation IDs: {result['validation_ids']}", flush=True)
        print(f"File ảnh hiện có: {result['present_images']}", flush=True)
        print(f"Thiếu ảnh: {result['missing_images']}", flush=True)
        print(f"File .jpg ngoài validation split: {result['unexpected_jpg_files']}", flush=True)

        if missing_ids:
            print("Một số ID bị thiếu:", sorted(missing_ids)[:20], flush=True)

        print(f"{category.upper()} SUMMARY: {result}", flush=True)

    print("\n========== TỔNG KẾT TOÀN BỘ FASHION-IQ ==========", flush=True)
    for category, result in overall.items():
        print(category.upper(), ":", result, flush=True)

    return overall


# --------------------------------------------------
# 7. KIỂM TRA DỮ LIỆU COCO TRÊN MODAL (CHỈ ĐỌC)
# --------------------------------------------------

@app.function(
    image=download_image,
    volumes={"/data": fashion_volume},
    cpu=2,
    memory=2048,
    timeout=1800,
)
def check_coco_on_modal():
    """
    Read-only audit of COCO data on pic2word-data.
    Expected: /data/coco/val2017, val2017_masked, coco_eval.csv.
    This function does not create, modify, delete, or commit files.
    """
    import csv
    from pathlib import Path

    root = Path("/data/coco")
    print("=" * 68, flush=True)
    print("COCO DATA CHECK ON MODAL (READ ONLY)", flush=True)
    print("COCO root:", root, flush=True)
    print("Fashion-IQ directory exists:", Path("/data/fashion-iq").is_dir(), flush=True)
    print("=" * 68, flush=True)

    if not root.is_dir():
        print("STATUS: COCO DIRECTORY NOT FOUND", flush=True)
        print("No files were changed.", flush=True)
        return {"status": "not_found", "root": str(root)}

    original_dir = root / "val2017"
    masked_dir = root / "val2017_masked"
    csv_path = root / "coco_eval.csv"

    original_count = sum(1 for p in original_dir.iterdir() if p.is_file()) if original_dir.is_dir() else 0
    masked_count = sum(1 for p in masked_dir.iterdir() if p.is_file()) if masked_dir.is_dir() else 0

    report = {
        "status": "checked",
        "root": str(root),
        "original_dir_exists": original_dir.is_dir(),
        "original_file_count": original_count,
        "masked_dir_exists": masked_dir.is_dir(),
        "masked_file_count": masked_count,
        "csv_exists": csv_path.is_file(),
    }

    print("\n[Directories]", flush=True)
    print("val2017 exists:", report["original_dir_exists"], flush=True)
    print("val2017 file count:", original_count, flush=True)
    print("val2017_masked exists:", report["masked_dir_exists"], flush=True)
    print("val2017_masked file count:", masked_count, flush=True)
    print("coco_eval.csv exists:", report["csv_exists"], flush=True)

    if not csv_path.is_file():
        print("STATUS: CSV NOT FOUND; no files were changed.", flush=True)
        return report

    report["csv_size_bytes"] = csv_path.stat().st_size
    missing_original = []
    missing_masked = []
    row_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        required = {"id", "query_regions", "query_class", "classes"}
        missing_columns = sorted(required - set(columns))
        report["csv_columns"] = columns

        if missing_columns:
            report["missing_columns"] = missing_columns
            print("CSV columns:", columns, flush=True)
            print("Missing required columns:", missing_columns, flush=True)
            print("STATUS: CSV SCHEMA MISMATCH", flush=True)
            return report

        for row in reader:
            row_count += 1
            image_id = (row.get("id") or "").strip()
            if not image_id:
                continue
            if not (original_dir / image_id).is_file() and len(missing_original) < 30:
                missing_original.append(image_id)
            if not (masked_dir / image_id).is_file() and len(missing_masked) < 30:
                missing_masked.append(image_id)

    report.update({
        "csv_rows": row_count,
        "missing_original_sample": missing_original,
        "missing_masked_sample": missing_masked,
    })

    print("\n[CSV]", flush=True)
    print("CSV size (bytes):", report["csv_size_bytes"], flush=True)
    print("CSV columns:", columns, flush=True)
    print("CSV rows:", row_count, flush=True)
    print("Missing original references (up to 30):", missing_original, flush=True)
    print("Missing masked references (up to 30):", missing_masked, flush=True)
    print("\nRead-only check finished. No files were changed or committed.", flush=True)
    return report




# --------------------------------------------------
# 7. CHẠY ĐÁNH GIÁ COCO CHÍNH THỨC
# --------------------------------------------------

@app.function(
    image=image,
    volumes={
        "/root/project/data": fashion_volume,
        "/results": results_volume,
    },
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=21600,
)
def run_coco_official(
    batch_size: int = 64,
    workers: int = 1,
):
    """
    Chạy pipeline COCO chính thức có sẵn trong src/eval_retrieval.py.

    Không tự viết lại CsvCOCO, DataLoader hoặc evaluate_coco.
    Không thay đổi số lượng mẫu hay logic metric.
    Kết quả được lưu riêng trong /results/coco.
    """
    import csv
    import os
    import subprocess
    import sys
    from pathlib import Path

    project_dir = Path("/root/project")
    data_root = project_dir / "data" / "coco"
    original_dir = data_root / "val2017"
    masked_dir = data_root / "val2017_masked"
    csv_path = data_root / "coco_eval.csv"
    checkpoint = project_dir / "checkpoints" / "pic2word_model.pt"
    eval_script = project_dir / "src" / "eval_retrieval.py"

    results_root = Path("/results/coco")
    log_root = results_root / "logs"

    # Kiểm tra đầu vào trước khi khởi chạy benchmark.
    required_paths = [
        original_dir,
        masked_dir,
        csv_path,
        checkpoint,
        eval_script,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required COCO evaluation path missing: {path}")

    if not original_dir.is_dir() or not masked_dir.is_dir():
        raise NotADirectoryError(
            f"Expected COCO image directories: {original_dir}, {masked_dir}"
        )

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if workers < 0:
        raise ValueError("workers cannot be negative")

    # Xác nhận schema và số dòng CSV; không lọc hoặc sửa dữ liệu.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        required_columns = {"id", "query_regions", "query_class", "classes"}
        missing_columns = sorted(required_columns - set(columns))
        if missing_columns:
            raise ValueError(
                f"COCO CSV is missing required columns: {missing_columns}"
            )
        csv_rows = sum(1 for _ in reader)

    original_count = sum(1 for p in original_dir.iterdir() if p.is_file())
    masked_count = sum(1 for p in masked_dir.iterdir() if p.is_file())

    print("=" * 72, flush=True)
    print("OFFICIAL COCO EVALUATION", flush=True)
    print("Project:", project_dir, flush=True)
    print("COCO data:", data_root, flush=True)
    print("CSV rows:", csv_rows, flush=True)
    print("Original images:", original_count, flush=True)
    print("Masked images:", masked_count, flush=True)
    print("Checkpoint:", checkpoint, flush=True)
    print("GPU requested: Modal T4, CUDA device 0", flush=True)
    print("Batch size:", batch_size, flush=True)
    print("Workers:", workers, flush=True)
    print("Results:", results_root, flush=True)
    print("=" * 72, flush=True)

    results_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(eval_script),
        "--eval-mode", "coco",
        "--model", "ViT-L/14",
        "--resume", str(checkpoint),
        "--middle_dim", "512",
        "--n-layer", "2",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--workers", str(workers),
        "--precision", "amp",
        "--logs", str(log_root),
        "--name", "coco_official",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(project_dir) + os.pathsep
        + str(project_dir / "src") + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    env["WANDB_MODE"] = "disabled"

    print("Working directory:", project_dir, flush=True)
    print("Command:", " ".join(command), flush=True)

    try:
        completed = subprocess.run(
            command,
            cwd=str(project_dir),
            env=env,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Official COCO evaluation failed with exit code "
                f"{completed.returncode}. Inspect logs under {log_root}."
            )
    finally:
        # Persist logs and partial outputs even if evaluation fails.
        results_volume.commit()

    print("\nOFFICIAL COCO EVALUATION COMPLETED", flush=True)
    print("Persistent results directory:", results_root, flush=True)

    return {
        "status": "completed",
        "benchmark": "COCO",
        "csv_rows": csv_rows,
        "original_image_count": original_count,
        "masked_image_count": masked_count,
        "gpu": "T4",
        "cuda_device": 0,
        "results_dir": str(results_root),
        "logs_dir": str(log_root),
    }


# --------------------------------------------------
# 8. CHẠY ĐÁNH GIÁ IMAGENET-R / IMAGENET REAL CHÍNH THỨC
# --------------------------------------------------

@app.function(
    image=image,
    volumes={
        "/root/project/data": fashion_volume,
        "/results": results_volume,
    },
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=21600,
)
def run_imgnet_official(
    batch_size: int = 64,
    workers: int = 1,
):
    """
    Run the repository's official ImageNet-R / ImageNet Real evaluation.

    Reads:
      data/imgnet/imgnet_real_query.txt
      data/imgnet/imgnet_targets.txt

    Writes logs only under /results/imgnet/logs.
    Does not modify source datasets or existing benchmark result folders.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    project_dir = Path("/root/project")
    data_root = project_dir / "data" / "imgnet"
    query_list = data_root / "imgnet_real_query.txt"
    target_list = data_root / "imgnet_targets.txt"
    checkpoint = project_dir / "checkpoints" / "pic2word_model.pt"
    eval_script = project_dir / "src" / "eval_retrieval.py"

    results_root = Path("/results/imgnet")
    log_root = results_root / "logs"

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if workers < 0:
        raise ValueError("workers cannot be negative")

    required_files = [
        query_list,
        target_list,
        checkpoint,
        eval_script,
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required ImageNet evaluation file is missing: {path}"
            )

    # Validate every list entry against the mounted data tree before
    # allocating the GPU. This is read-only and follows ImageList's
    # whitespace-separated "relative_image_path label" format.
    def audit_image_list(list_path, list_name):
        total = 0
        missing_images = []
        malformed_lines = []
        invalid_labels = []

        with list_path.open("r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    malformed_lines.append(line_number)
                    continue

                relative_path = parts[0]
                try:
                    label = int(parts[1])
                except ValueError:
                    malformed_lines.append(line_number)
                    continue

                total += 1
                if label < 0 or label >= 7000:
                    invalid_labels.append((line_number, label))

                image_path = project_dir / "data" / relative_path
                if not image_path.is_file() and len(missing_images) < 30:
                    missing_images.append(
                        (line_number, relative_path)
                    )

        if total == 0:
            raise ValueError(f"{list_name} contains no usable entries: {list_path}")
        if malformed_lines:
            raise ValueError(
                f"{list_name} contains malformed lines; "
                f"sample line numbers: {malformed_lines[:10]}"
            )
        if invalid_labels:
            raise ValueError(
                f"{list_name} contains labels outside [0, 6999]; "
                f"sample: {invalid_labels[:10]}"
            )
        if missing_images:
            raise FileNotFoundError(
                f"{list_name} references missing images; "
                f"sample: {missing_images[:10]}"
            )

        print(f"{list_name} entries:", total, flush=True)
        print(f"{list_name} image paths: all found", flush=True)
        return total

    query_count = audit_image_list(query_list, "ImageNet Real query list")
    target_count = audit_image_list(target_list, "ImageNet-R target list")

    checkpoint_size_gb = checkpoint.stat().st_size / (1024 ** 3)

    print("=" * 72, flush=True)
    print("OFFICIAL IMAGENET-R / IMAGENET REAL EVALUATION", flush=True)
    print("Project:", project_dir, flush=True)
    print("Data root:", data_root, flush=True)
    print("Query images:", query_count, flush=True)
    print("Target images:", target_count, flush=True)
    print("Checkpoint:", checkpoint, flush=True)
    print("Checkpoint size (GB):", round(checkpoint_size_gb, 3), flush=True)
    print("Model: ViT-L/14", flush=True)
    print("IM2TEXT: middle_dim=512, n_layer=2", flush=True)
    print("GPU: Modal T4, CUDA device 0", flush=True)
    print("Batch size:", batch_size, flush=True)
    print("Workers:", workers, flush=True)
    print("Persistent results:", results_root, flush=True)
    print("=" * 72, flush=True)

    results_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(eval_script),
        "--eval-mode", "imgnet",
        "--model", "ViT-L/14",
        "--resume", str(checkpoint),
        "--middle_dim", "512",
        "--n-layer", "2",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--workers", str(workers),
        "--precision", "amp",
        "--logs", str(log_root),
        "--name", "imgnet_official",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(project_dir) + os.pathsep
        + str(project_dir / "src") + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    env["WANDB_MODE"] = "disabled"

    print("Working directory:", project_dir, flush=True)
    print("Command:", " ".join(command), flush=True)

    try:
        completed = subprocess.run(
            command,
            cwd=str(project_dir),
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Official ImageNet evaluation failed with exit code "
                f"{completed.returncode}. Inspect logs under {log_root}."
            )
    finally:
        # Persist logs and partial outputs even if evaluation fails.
        results_volume.commit()

    output_dir = log_root / "imgnet_official"
    print("\\nOFFICIAL IMAGENET EVALUATION COMPLETED", flush=True)
    print("Persistent output directory:", output_dir, flush=True)

    if output_dir.is_dir():
        for output_file in sorted(output_dir.iterdir()):
            if output_file.is_file():
                print(
                    output_file.name,
                    output_file.stat().st_size,
                    "bytes",
                    flush=True,
                )

    return {
        "status": "completed",
        "benchmark": "ImageNet-R / ImageNet Real",
        "query_count": query_count,
        "target_count": target_count,
        "gpu": "T4",
        "cuda_device": 0,
        "results_dir": str(output_dir),
        "logs_dir": str(log_root),
    }



# --------------------------------------------------
# 7. ENTRY POINT
# --------------------------------------------------

@app.function(
    image=image,
    volumes={
        "/root/project/data": fashion_volume,
        "/results": results_volume,
    },
    cpu=4,
    memory=8192,
    timeout=3600,
)
def audit_fashioniq_data():
    """
    Audit Fashion-IQ on Modal at the exact path expected by
    src/eval_retrieval.py. Checks split coverage, image readability,
    annotation structure, and annotation image references.
    """
    import json
    from pathlib import Path
    from PIL import Image

    root = Path("/root/project/data/fashion-iq")
    categories = ("dress", "shirt", "toptee")
    report = {}
    output_dir = Path("/results/fashion_iq")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "modal_data_audit.json"

    def persist_progress():
        # Persist completed categories immediately so a later interruption
        # does not discard earlier audit results.
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        results_volume.commit()
        print(f"Progress report committed: {output_path}", flush=True)

    def resolve_image(category, image_id):
        image_id = str(image_id)
        suffix = Path(image_id).suffix.lower()
        extensions = [suffix] if suffix in {".jpg", ".jpeg", ".png"} else [
            ".jpg", ".jpeg", ".png"
        ]
        for ext in extensions:
            filename = image_id if suffix else image_id + ext
            candidates = (
                root / "images" / category / filename,
                root / "images" / filename,
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
        return None

    for category in categories:
        split_path = root / "image_splits" / f"split.{category}.val.json"
        # Match src/data.py exactly: prefer json/, fall back to captions/.
        json_caption_path = root / "json" / f"cap.{category}.val.json"
        fallback_caption_path = root / "captions" / f"cap.{category}.val.json"
        caption_path = (
            json_caption_path if json_caption_path.is_file()
            else fallback_caption_path
        )

        for required in (split_path, caption_path):
            if not required.is_file():
                raise FileNotFoundError(f"Required Fashion-IQ file missing: {required}")

        with split_path.open(encoding="utf-8") as f:
            image_ids = json.load(f)
        with caption_path.open(encoding="utf-8") as f:
            annotations = json.load(f)

        if not isinstance(image_ids, list) or not isinstance(annotations, list):
            raise TypeError(f"Unexpected JSON structure for {category}")

        missing = []
        unreadable = []
        readable = 0
        for image_id in image_ids:
            image_path = resolve_image(category, image_id)
            if image_path is None:
                missing.append(str(image_id))
                continue
            try:
                with Image.open(image_path) as im:
                    im.verify()
                readable += 1
            except Exception:
                unreadable.append(str(image_id))

        missing_annotation_refs = []
        malformed_annotations = []
        for index, item in enumerate(annotations):
            if not isinstance(item, dict) or not all(
                key in item for key in ("candidate", "target", "captions")
            ):
                malformed_annotations.append(index)
                continue
            # data.py later unpacks captions as cap1, cap2 = ref_caps[idx].
            # Flag malformed captions here rather than letting evaluation fail.
            captions = item["captions"]
            if (
                not isinstance(captions, (list, tuple))
                or len(captions) < 2
                or not all(isinstance(c, str) for c in captions[:2])
            ):
                malformed_annotations.append(index)
                continue

            for field in ("candidate", "target"):
                if not isinstance(item[field], str) or resolve_image(category, item[field]) is None:
                    missing_annotation_refs.append({
                        "annotation_index": index,
                        "field": field,
                        "image_id": str(item[field]),
                    })

        report[category] = {
            "annotation_file_used": str(caption_path.relative_to(root)),
            "gallery_split_count": len(image_ids),
            "gallery_readable_count": readable,
            "gallery_missing_count": len(missing),
            "gallery_unreadable_count": len(unreadable),
            "annotation_count": len(annotations),
            "malformed_annotation_count": len(malformed_annotations),
            "missing_annotation_image_reference_count": len(missing_annotation_refs),
            "sample_missing_gallery_ids": missing[:10],
            "sample_unreadable_gallery_ids": unreadable[:10],
            "sample_missing_annotation_refs": missing_annotation_refs[:10],
            "sample_malformed_annotation_indices": malformed_annotations[:10],
        }

        print(f"\\n[{category.upper()}]")
        for key, value in report[category].items():
            print(f"{key}: {value}")

        # Commit after each category. If the next category fails or the
        # function is interrupted, completed category results remain saved.
        persist_progress()

    print(f"\\nAudit report saved to {output_path}")
    return report


@app.function(
    image=image,
    volumes={
        "/root/project/data": fashion_volume,
        "/results": results_volume,
    },
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=21600,
)
def run_fashioniq_official(
    category: str = "dress",
    batch_size: int = 64,
    workers: int = 1,
):
    """
    Run the repository's official Fashion-IQ evaluation entrypoint.
    This does not replace or reimplement evaluate_fashion().
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    if category not in {"dress", "shirt", "toptee"}:
        raise ValueError("category must be dress, shirt, or toptee")

    project_dir = Path("/root/project")
    data_root = project_dir / "data" / "fashion-iq"
    checkpoint = project_dir / "checkpoints" / "pic2word_model.pt"
    eval_script = project_dir / "src" / "eval_retrieval.py"
    log_root = Path("/results/fashion_iq")

    for required in (data_root, checkpoint, eval_script):
        if not required.exists():
            raise FileNotFoundError(f"Required path missing: {required}")

    log_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(eval_script),
        "--eval-mode", "fashion",
        "--source-data", category,
        "--model", "ViT-L/14",
        "--resume", str(checkpoint),
        "--middle_dim", "512",
        "--n-layer", "2",
        "--batch-size", str(batch_size),
        "--workers", str(workers),
        "--precision", "amp",
        "--logs", str(log_root),
        "--name", category,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(project_dir) + os.pathsep
        + str(project_dir / "src") + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    env["WANDB_MODE"] = "disabled"

    print("Working directory:", project_dir)
    print("Data root:", data_root)
    print("Checkpoint:", checkpoint)
    print("Results root:", log_root)
    print("Command:", " ".join(command), flush=True)

    try:
        completed = subprocess.run(
            command,
            cwd=str(project_dir),
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Official Fashion-IQ evaluation failed with exit code "
                f"{completed.returncode}. Check {log_root / category / 'out.log'}"
            )
    finally:
        # Persist logs and partial outputs even when evaluation fails.
        results_volume.commit()

    print("Official Fashion-IQ evaluation completed:", category)
    print("Persistent output directory:", log_root / category)
    return {
        "status": "completed",
        "category": category,
        "results_dir": str(log_root / category),
    }


@app.local_entrypoint()
def main(
    action: str = "audit-fashion",
    category: str = "dress",
    batch_size: int = 64,
    workers: int = 1,
):
    """
    Examples:
      modal run modal_eval.py
      modal run modal_eval.py --action audit-fashion
      modal run modal_eval.py --action run-fashion --category dress
      modal run modal_eval.py --action run-coco
      modal run modal_eval.py --action run-imgnet
      modal run modal_eval.py --action check-checkpoint
      modal run modal_eval.py --action check-coco
      modal run modal_eval.py --action audit-cirr
      modal run modal_eval.py --action run-cirr-validation
      modal run modal_eval.py --action run-cirr-test1
    """
    if action == "check-coco":
        check_coco_on_modal.remote()
    elif action == "audit-fashion":
        audit_fashioniq_data.remote()
    elif action == "run-fashion":
        run_fashioniq_official.remote(
            category=category,
            batch_size=batch_size,
            workers=workers,
        )
    elif action == "run-coco":
        run_coco_official.remote(
            batch_size=batch_size,
            workers=workers,
        )
    elif action == "run-imgnet":
        run_imgnet_official.remote(
            batch_size=batch_size,
            workers=workers,
        )
    elif action == "audit-cirr":
        audit_cirr_data.remote()
    elif action == "run-cirr-validation":
        run_cirr_validation.remote(
            batch_size=batch_size,
            workers=workers,
        )
    elif action == "run-cirr-test1":
        run_cirr_test1.remote(
            batch_size=batch_size,
            workers=workers,
        )
    elif action == "check-checkpoint":
        # CPU-only file/key inspection; does not allocate a GPU.
        check_checkpoint.remote()
    elif action == "check-compatibility":
        # CPU-only architecture/tensor-shape validation.
        check_checkpoint_compatibility.remote()
    elif action == "test-checkpoint-loading":
        # Explicitly GPU-backed test of loading the real model.
        test_checkpoint_loading.remote()
    else:
        raise ValueError(
            "Unsupported action. Choose audit-fashion, run-fashion, run-coco, "
            "run-imgnet, audit-cirr, run-cirr-validation, run-cirr-test1, "
            "check-coco, check-checkpoint, "
            "check-compatibility, or test-checkpoint-loading."
        )

# --------------------------------------------------
# 8. KIỂM TRA TƯƠNG THÍCH CHECKPOINT VỚI KIẾN TRÚC
# --------------------------------------------------

@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1200,
)
def check_checkpoint_compatibility():

    import os
    import torch

    checkpoint_path = (
        "/root/project/checkpoints/pic2word_model.pt"
    )

    print("=" * 60)
    print("CHECKPOINT COMPATIBILITY CHECK")
    print("=" * 60)

    # 1. Kiểm tra checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    print("\n[1] Checkpoint")
    print("Path:", checkpoint_path)
    print(
        "Size:",
        round(os.path.getsize(checkpoint_path) / (1024 ** 3), 3),
        "GB"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required = [
        "state_dict",
        "state_dict_img2text",
        "epoch",
    ]

    for key in required:
        if key not in checkpoint:
            raise KeyError(f"Missing checkpoint key: {key}")

    print("Epoch:", checkpoint["epoch"])
    print("Required checkpoint keys: OK")

    # 2. Kiểm tra state dictionaries
    sd = checkpoint["state_dict"]
    sd_img2text = checkpoint["state_dict_img2text"]

    if not isinstance(sd, dict) or not isinstance(sd_img2text, dict):
        raise TypeError("Invalid state_dict format")

    print("\n[2] State dictionaries")
    print("CLIP tensors:", len(sd))
    print("IM2TEXT tensors:", len(sd_img2text))

    # 3. Kiểm tra tiền tố module.
    clip_keys = list(sd.keys())
    im2text_keys = list(sd_img2text.keys())

    clip_module_prefix = all(
        key.startswith("module.") for key in clip_keys
    )

    im2text_module_prefix = all(
        key.startswith("module.") for key in im2text_keys
    )

    print("\n[3] Module prefixes")
    print("CLIP module prefix:", clip_module_prefix)
    print("IM2TEXT module prefix:", im2text_module_prefix)

    # Chuẩn hóa key để kiểm tra tên và kích thước.
    normalized_sd = {
        key.removeprefix("module."): value
        for key, value in sd.items()
    }

    normalized_im2text = {
        key.removeprefix("module."): value
        for key, value in sd_img2text.items()
    }

    # 4. Kiểm tra kiến trúc CLIP ViT-L/14
    print("\n[4] CLIP architecture")

    expected_clip_shapes = {
        "visual.conv1.weight": (1024, 3, 14, 14),
        "visual.positional_embedding": (257, 1024),
        "visual.proj": (1024, 768),
        "token_embedding.weight": (49408, 768),
        "positional_embedding": (77, 768),
        "text_projection": (768, 768),
    }

    for key, expected_shape in expected_clip_shapes.items():

        if key not in normalized_sd:
            raise KeyError(f"Missing CLIP tensor: {key}")

        actual_shape = tuple(normalized_sd[key].shape)

        print(key, actual_shape)

        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"expected {expected_shape}, got {actual_shape}"
            )

    # Kiểm tra số block Transformer.
    visual_blocks = [
        key for key in normalized_sd
        if key.startswith("visual.transformer.resblocks.")
        and key.endswith(".attn.in_proj_weight")
    ]

    text_blocks = [
        key for key in normalized_sd
        if key.startswith("transformer.resblocks.")
        and key.endswith(".attn.in_proj_weight")
    ]

    print("Visual transformer blocks:", len(visual_blocks))
    print("Text transformer blocks:", len(text_blocks))

    if len(visual_blocks) != 24:
        raise ValueError("Expected 24 visual transformer blocks")

    if len(text_blocks) != 12:
        raise ValueError("Expected 12 text transformer blocks")

    # 5. Kiểm tra IM2TEXT
    print("\n[5] IM2TEXT architecture")

    expected_im2text_shapes = {
        "layers.0.0.weight": (512, 768),
        "layers.0.0.bias": (512,),
        "layers.1.0.weight": (512, 512),
        "layers.1.0.bias": (512,),
        "fc_out.weight": (768, 512),
        "fc_out.bias": (768,),
    }

    for key, expected_shape in expected_im2text_shapes.items():

        if key not in normalized_im2text:
            raise KeyError(f"Missing IM2TEXT tensor: {key}")

        actual_shape = tuple(normalized_im2text[key].shape)

        print(key, actual_shape)

        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"expected {expected_shape}, got {actual_shape}"
            )

    # 6. Kết luận
    print("\n" + "=" * 60)
    print("CHECKPOINT COMPATIBILITY CHECK PASSED")
    print("CLIP architecture: ViT-L/14")
    print("IM2TEXT: embed_dim=768, middle_dim=512")
    print("IM2TEXT: output_dim=768, n_layer=2")
    print("=" * 60)

    return {
        "epoch": checkpoint["epoch"],
        "clip_architecture": "ViT-L/14",
        "visual_transformer_blocks": len(visual_blocks),
        "text_transformer_blocks": len(text_blocks),
        "im2text_embed_dim": 768,
        "im2text_middle_dim": 512,
        "im2text_output_dim": 768,
        "im2text_n_layer": 2,
        "compatible": True,
    }

# --------------------------------------------------
# 9. KIỂM TRA NẠP CHECKPOINT VÀO MÔ HÌNH THỰC TẾ
# --------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=1800,
)
def test_checkpoint_loading():

    import os
    import sys
    import torch

    project_dir = "/root/project"
    model_file = os.path.join(project_dir, "model", "model.py")
    checkpoint_path = os.path.join(
        project_dir,
        "checkpoints",
        "pic2word_model.pt",
    )

    # Đảm bảo mã nguồn dự án được tìm thấy trước khi import.
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(
            f"Project directory not found: {project_dir}"
        )

    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"Model source not found: {model_file}. "
            "Check that the root model/ directory is included "
            "in image.add_local_dir()."
        )

    # Repository layout: model/ and third_party/ are at project root,
    # while the project's data.py is under src/.
    source_dir = os.path.join(project_dir, "src")
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    from model.model import IM2TEXT, build_model

    print("=" * 60)
    print("TEST CHECKPOINT LOADING")
    print("=" * 60)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("Device:", device)
    print("PyTorch:", torch.__version__)

    # 1. Đọc checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    sd = checkpoint["state_dict"]
    sd_img2text = checkpoint["state_dict_img2text"]

    # 2. Loại bỏ tiền tố module.
    def remove_module_prefix(state_dict):

        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    sd = remove_module_prefix(sd)
    sd_img2text = remove_module_prefix(sd_img2text)

    print("\n[1] State dictionaries prepared.")

    # 3. Khởi tạo CLIP từ chính checkpoint.
    print("\n[2] Building CLIP model...")

    clip_model = build_model(sd)

    clip_model = clip_model.to(device)
    clip_model.eval()

    print("CLIP model built successfully.")

    # 4. Khởi tạo IM2TEXT
    print("\n[3] Building IM2TEXT model...")

    img2text = IM2TEXT(
        embed_dim=768,
        middle_dim=512,
        output_dim=768,
        n_layer=2,
    )

    result = img2text.load_state_dict(
        sd_img2text,
        strict=True,
    )

    img2text = img2text.to(device)
    img2text.eval()

    print("IM2TEXT model loaded successfully.")
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    # 5. Kiểm tra forward IM2TEXT
    print("\n[4] Testing IM2TEXT forward pass...")

    with torch.no_grad():

        dummy_input = torch.randn(
            2,
            768,
            device=device,
        )

        output = img2text(dummy_input)

    print("Input shape:", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))

    if tuple(output.shape) != (2, 768):
        raise RuntimeError(
            f"Unexpected IM2TEXT output shape: {output.shape}"
        )

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "IM2TEXT output contains NaN or Inf."
        )

    # 6. Kết luận
    print("\n" + "=" * 60)
    print("CHECKPOINT LOADING TEST PASSED")
    print("CLIP: loaded successfully")
    print("IM2TEXT: loaded successfully")
    print("IM2TEXT forward pass: passed")
    print("=" * 60)

    return {
        "clip_loaded": True,
        "img2text_loaded": True,
        "forward_pass": True,
        "output_shape": tuple(output.shape),
    }



# --------------------------------------------------
# 11. KIỂM TRA END-TO-END PIC2WORD TRÊN FASHION-IQ
# --------------------------------------------------

@app.function(
    image=image,
    volumes={"/data": fashion_volume},
    cpu=8,
    memory=16384,
    timeout=3600,
)
def test_fashioniq_end_to_end(
    cloth="dress",
    num_queries=2,
    gallery_batch_size=32,
):
    """
    Smoke test toàn bộ luồng Fashion-IQ:
    ảnh tham chiếu -> CLIP -> IM2TEXT -> pseudo-token + caption
    -> CLIP text encoder -> so khớp với gallery.

    Đây là kiểm tra kỹ thuật trên một số query, KHÔNG phải kết quả
    đánh giá benchmark đầy đủ.
    """
    import os
    import sys
    import json
    import logging
    from pathlib import Path

    import torch
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader, Subset

    project_dir = "/root/project"
    project_src = os.path.join(project_dir, "src")
    checkpoint_path = os.path.join(
        project_dir, "checkpoints", "pic2word_model.pt"
    )
    data_root = "/data"
    fashion_root = os.path.join(data_root, "fashion-iq")

    # Repository modules are split across the project root and src/.
    # model/ and third_party/ are under project root; eval_utils.py and
    # utils.py are under src/ in this repository layout.
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    if cloth not in {"dress", "shirt", "toptee"}:
        raise ValueError("cloth must be dress, shirt, or toptee")
    if num_queries < 1:
        raise ValueError("num_queries must be at least 1")

    required_paths = [
        checkpoint_path,
        os.path.join(project_dir, "model", "model.py"),
        os.path.join(project_dir, "third_party", "open_clip", "clip.py"),
        os.path.join(project_dir, "src", "data.py"),
        os.path.join(project_src, "eval_utils.py"),
        os.path.join(project_src, "utils.py"),
    ]
    for path in required_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required project file missing: {path}")

    import src.data as project_data
    from model.model import IM2TEXT, build_model
    from third_party.open_clip.clip import tokenize, _transform
    from eval_utils import get_metrics_fashion

    print("=" * 72)
    print("FASHION-IQ END-TO-END SMOKE TEST")
    print("Category:", cloth)
    print("Requested query count:", num_queries)
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("PyTorch:", torch.__version__)

    # 1. Nạp đúng checkpoint đã dùng ở các bài test trước.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    def strip_module_prefix(state_dict):
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    clip_state = strip_module_prefix(checkpoint["state_dict"])
    img2text_state = strip_module_prefix(
        checkpoint["state_dict_img2text"]
    )

    clip_model = build_model(clip_state).to(device).float().eval()

    img2text = IM2TEXT(
        embed_dim=768,
        middle_dim=512,
        output_dim=768,
        n_layer=2,
    )
    img2text.load_state_dict(img2text_state, strict=True)
    img2text = img2text.to(device).float().eval()

    preprocess = _transform(clip_model.visual.input_resolution)

    # 2. Đọc gallery split và chỉ lấy các ảnh thực sự có trên Volume.
    #    Đây là smoke test có ghi nhận số ảnh thiếu, không âm thầm giả
    #    định rằng gallery đầy đủ.
    split_path = os.path.join(
        fashion_root,
        "image_splits",
        f"split.{cloth}.val.json",
    )
    caption_path_json = os.path.join(
        fashion_root, "json", f"cap.{cloth}.val.json"
    )
    caption_path_compat = os.path.join(
        fashion_root, "captions", f"cap.{cloth}.val.json"
    )
    caption_path = (
        caption_path_json
        if os.path.isfile(caption_path_json)
        else caption_path_compat
    )

    for path in [split_path, caption_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    with open(split_path, "r", encoding="utf-8") as f:
        gallery_ids = json.load(f)

    image_dir = os.path.join(fashion_root, "images")

    def resolve_image(image_id):
        image_id = str(image_id)
        suffix = Path(image_id).suffix.lower()
        extensions = [suffix] if suffix in {".png", ".jpg", ".jpeg"} else [
            ".png", ".jpg", ".jpeg"
        ]
        stem = image_id if suffix else image_id
        for ext in extensions:
            filename = stem if suffix else stem + ext
            # Fashion-IQ data on the Modal volume may be stored either
            # directly under images/ or inside category folders
            # images/dress, images/shirt, images/toptee.
            candidates = [
                os.path.join(image_dir, filename),
                os.path.join(image_dir, cloth, filename),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    return candidate
        return None

    gallery_items = []
    missing_gallery = []
    unreadable_gallery = []

    for image_id in gallery_ids:
        path = resolve_image(image_id)
        if path is None:
            missing_gallery.append(str(image_id))
            continue
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            unreadable_gallery.append(str(image_id))
            continue
        gallery_items.append((path, str(path)))

    print("\n[Gallery]")
    print("IDs in official split:", len(gallery_ids))
    print("Readable gallery images used:", len(gallery_items))
    print("Missing gallery images:", len(missing_gallery))
    print("Unreadable gallery images:", len(unreadable_gallery))

    if not gallery_items:
        raise RuntimeError("No readable gallery images were found.")

    class ExistingGallery(Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            path, name = self.items[index]
            with Image.open(path) as im:
                tensor = self.transform(im.convert("RGB"))
            return tensor, name

    gallery_dataset = ExistingGallery(gallery_items, preprocess)
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=gallery_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 3. Dùng chính FashionIQ loader của project để đọc query/caption.
    #    Loader này tự bỏ qua query pair có ảnh ref/target bị thiếu.
    # The project FashionIQ loader originally searches only images/.
    # The Modal volume stores images in images/<category>/, so extend
    # its resolver for this dataset construction while preserving the
    # original annotation parsing, caption tokenization, and __getitem__.
    original_resolve_image = project_data.FashionIQ._resolve_image

    def resolve_image_with_category(self, image_id):
        resolved = original_resolve_image(self, image_id)
        if resolved is not None:
            return resolved

        image_id = str(image_id)
        suffix = Path(image_id).suffix.lower()
        extensions = (
            [suffix]
            if suffix in {".png", ".jpg", ".jpeg"}
            else [".png", ".jpg", ".jpeg"]
        )
        filename = image_id
        for ext in extensions:
            candidate_name = (
                filename if suffix else filename + ext
            )
            candidate = os.path.join(
                self.root_img, self.cloth, candidate_name
            )
            if os.path.isfile(candidate):
                return candidate
        return None

    project_data.FashionIQ._resolve_image = resolve_image_with_category
    try:
        query_dataset = project_data.FashionIQ(
            cloth=cloth,
            transforms=preprocess,
            is_train=False,
            vis_mode=False,
            mode="caps",
            is_return_target_path=True,
            root=data_root,
        )
    finally:
        project_data.FashionIQ._resolve_image = original_resolve_image

    gallery_name_set = {name for _, name in gallery_items}
    eligible_indices = []
    for index in range(len(query_dataset)):
        target_path = query_dataset.target_imgs[index]
        if str(target_path) in gallery_name_set:
            eligible_indices.append(index)
        if len(eligible_indices) >= num_queries:
            break

    if len(eligible_indices) < num_queries:
        raise RuntimeError(
            f"Only {len(eligible_indices)} eligible query pair(s) found "
            f"for requested num_queries={num_queries}. "
            "A query target must also exist in the available gallery."
        )

    query_subset = Subset(query_dataset, eligible_indices)
    query_loader = DataLoader(
        query_subset,
        batch_size=min(4, num_queries),
        shuffle=False,
        num_workers=0,
    )

    print("\n[Queries]")
    print("Valid query pairs in dataset:", len(query_dataset))
    print("Selected query indices:", eligible_indices)
    print("Selected query count:", len(eligible_indices))

    # 4. Trích xuất feature gallery theo đúng evaluate_fashion().
    all_gallery_features = []
    all_gallery_names = []

    with torch.inference_mode():
        for images, names in gallery_loader:
            images = images.to(device)
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            if not torch.isfinite(features).all():
                raise RuntimeError("Gallery features contain NaN or Inf.")
            all_gallery_features.append(features.cpu())
            all_gallery_names.extend(list(names))

    gallery_features = torch.cat(all_gallery_features, dim=0)

    # 5. Trích xuất bốn loại query feature như eval_utils.evaluate_fashion.
    composed_features = []
    image_query_features = []
    caption_features_all = []
    mixture_features = []
    answer_names = []

    split_token_id = int(tokenize(["*"])[0][1].item())
    print("\n[Tokenizer]")
    print("Placeholder '*' token ID:", split_token_id)

    with torch.inference_mode():
        for batch in query_loader:
            (
                ref_images,
                target_images,
                target_caption,
                caption_only,
                answer_paths,
                ref_names,
                captions,
            ) = batch

            ref_images = ref_images.to(device)
            target_caption = target_caption.to(device)
            caption_only = caption_only.to(device)

            # Tương ứng evaluate_fashion() trong eval_utils.py.
            query_image_features_raw = clip_model.encode_image(ref_images)
            caption_features_raw = clip_model.encode_text(target_caption)
            query_image_tokens = img2text(query_image_features_raw)

            composed = clip_model.encode_text_img_retrieval(
                target_caption,
                query_image_tokens,
                split_ind=split_token_id,
                repeat=False,
            )

            query_image = (
                query_image_features_raw
                / query_image_features_raw.norm(dim=-1, keepdim=True)
            )
            caption = (
                caption_features_raw
                / caption_features_raw.norm(dim=-1, keepdim=True)
            )
            composed = composed / composed.norm(dim=-1, keepdim=True)

            mixture = query_image + caption
            mixture = mixture / mixture.norm(dim=-1, keepdim=True)

            for tensor_name, tensor in [
                ("query image", query_image),
                ("caption", caption),
                ("composed", composed),
                ("mixture", mixture),
            ]:
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(
                        f"{tensor_name} features contain NaN or Inf."
                    )

            image_query_features.append(query_image.cpu())
            caption_features_all.append(caption.cpu())
            composed_features.append(composed.cpu())
            mixture_features.append(mixture.cpu())
            answer_names.extend([str(path) for path in answer_paths])

    feature_sets = {
        "composed": torch.cat(composed_features, dim=0),
        "image": torch.cat(image_query_features, dim=0),
        "text": torch.cat(caption_features_all, dim=0),
        "mixture": torch.cat(mixture_features, dim=0),
    }

    # 6. Gọi đúng metric Fashion-IQ trong eval_utils.py.
    print("\n[Metrics]")
    print("Gallery feature shape:", tuple(gallery_features.shape))
    print("Answer count:", len(answer_names))

    metrics_by_method = {}
    for method, query_features in feature_sets.items():
        metrics = get_metrics_fashion(
            image_features=gallery_features,
            ref_features=query_features,
            target_names=all_gallery_names,
            answer_names=answer_names,
        )
        metrics_by_method[method] = metrics
        print(
            method,
            "|",
            " | ".join(
                f"{key}: {value:.2f}"
                for key, value in metrics.items()
            ),
        )

    print("\n" + "=" * 72)
    print("FASHION-IQ END-TO-END SMOKE TEST PASSED")
    print("CLIP image encoding: passed")
    print("IM2TEXT mapping: passed")
    print("Pseudo-token insertion + text encoding: passed")
    print("Gallery similarity ranking + Fashion-IQ metrics: passed")
    print(
        "NOTE: This is a small technical smoke test, "
        "not the full benchmark result."
    )
    print("=" * 72)

    return {
        "status": "passed",
        "category": cloth,
        "query_count": len(answer_names),
        "gallery_count_used": len(all_gallery_names),
        "gallery_count_in_official_split": len(gallery_ids),
        "missing_gallery_count": len(missing_gallery),
        "unreadable_gallery_count": len(unreadable_gallery),
        "metrics": metrics_by_method,
    }

# --------------------------------------------------
# 10. CIRR: KIỂM TRA DỮ LIỆU VÀ CHẠY VALIDATION
# --------------------------------------------------

@app.function(
    image=image,
    volumes={"/root/project/data": cirr_volume},
    cpu=4,
    memory=8192,
    timeout=1800,
)
def audit_cirr_data():
    """
    CPU-only audit of the standard CIRR Validation/Test1 resources.
    Does not modify the dataset or allocate a GPU.
    """
    import json
    from pathlib import Path

    root = Path("/root/project/data/CIRR")
    required = [
        root / "captions/cap.rc2.val.json",
        root / "captions/cap.rc2.test1.json",
        root / "image_splits/split.rc2.val.json",
        root / "image_splits/split.rc2.test1.json",
        root / "img_raw/dev",
        root / "img_raw/test1",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing CIRR resources in the mounted Volume: " + ", ".join(missing)
        )

    def image_index(folder):
        return {
            path.stem: path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        }

    def audit_split(split_name, caption_file, split_file, image_dir, is_test):
        with open(caption_file, "r", encoding="utf-8") as f:
            captions = json.load(f)
        with open(split_file, "r", encoding="utf-8") as f:
            split_ids = json.load(f)

        images = image_index(image_dir)
        split_ids = [str(value) for value in split_ids]
        split_set = set(split_ids)

        missing_split_images = sorted(split_set - set(images))
        extra_images = sorted(set(images) - split_set)
        pairids = [str(row["pairid"]) for row in captions]
        duplicate_pairids = len(pairids) - len(set(pairids))

        references = {
            str(row["reference"]).replace(".png", "")
            for row in captions
        }
        missing_references = sorted(references - set(images))

        report = {
            "split": split_name,
            "caption_records": len(captions),
            "image_files": len(images),
            "split_ids": len(split_ids),
            "missing_split_images": len(missing_split_images),
            "extra_images_not_in_split": len(extra_images),
            "unique_references": len(references),
            "missing_references": len(missing_references),
            "duplicate_pairids": duplicate_pairids,
        }

        if not is_test:
            targets = {
                str(row["target_hard"]).replace(".png", "")
                for row in captions
            }
            missing_targets = sorted(targets - set(images))
            report["unique_target_hard"] = len(targets)
            report["missing_target_hard"] = len(missing_targets)

        if (
            missing_split_images
            or extra_images
            or missing_references
            or duplicate_pairids
            or (not is_test and report.get("missing_target_hard", 0))
        ):
            raise RuntimeError(
                f"CIRR {split_name} audit failed: {report}"
            )

        print(f"\\n[{split_name}]")
        for key, value in report.items():
            print(f"{key}: {value}")
        return report

    print("CIRR Volume root:", root)
    print("Checkpoint exists:", Path(
        "/root/project/checkpoints/pic2word_model.pt"
    ).is_file())

    reports = [
        audit_split(
            "Validation",
            root / "captions/cap.rc2.val.json",
            root / "image_splits/split.rc2.val.json",
            root / "img_raw/dev",
            is_test=False,
        ),
        audit_split(
            "Test1",
            root / "captions/cap.rc2.test1.json",
            root / "image_splits/split.rc2.test1.json",
            root / "img_raw/test1",
            is_test=True,
        ),
    ]
    print("\\nCIRR DATA AUDIT PASSED")
    return reports


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=21600,
    volumes={
         "/root/project/data": cirr_volume,
        "/results": cirr_results_volume,
    },
)
def run_cirr_validation(batch_size: int = 64, workers: int = 1):
    """
    Full official CIRR Validation evaluation through the repository's
    src/eval_retrieval.py implementation.
    """
    import os
    import subprocess
    from pathlib import Path

    project = Path("/root/project")
    checkpoint = project / "checkpoints/pic2word_model.pt"
    cirr_root = project / "data/CIRR"
    log_root = Path("/results/cirr_validation")

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for relative in [
        "captions/cap.rc2.val.json",
        "image_splits/split.rc2.val.json",
        "img_raw/dev",
    ]:
        if not (cirr_root / relative).exists():
            raise FileNotFoundError(cirr_root / relative)

    log_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(project / "src")
        + os.pathsep
        + str(project)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    env["WANDB_MODE"] = "disabled"

    command = [
        "python", "src/eval_retrieval.py",
        "--eval-mode", "cirr",
        "--model", "ViT-L/14",
        "--resume", str(checkpoint),
        "--middle_dim", "512",
        "--n-layer", "2",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--workers", str(workers),
        "--precision", "amp",
        "--logs", str(log_root),
        "--name", "cirr_validation",
    ]

    print("Running official CIRR Validation implementation:")
    print(" ".join(command))
    try:
        subprocess.run(
            command,
            cwd=str(project),
            env=env,
            check=True,
        )
    finally:
        cirr_results_volume.commit()

    output_dir = log_root / "cirr_validation"
    print("\\nCIRR Validation evaluation completed.")
    print("Persistent output directory:", output_dir)
    for output_file in sorted(output_dir.glob("*")):
        if output_file.is_file():
            print(output_file.name, output_file.stat().st_size, "bytes")

    return {
        "status": "completed",
        "benchmark": "CIRR Validation",
        "results_dir": str(output_dir),
    }


# --------------------------------------------------
# 12. CHẠY CIRR TEST1 VÀ TẠO FILE SUBMISSION JSON
# --------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=21600,
    volumes={
         "/root/project/data": cirr_volume,
        "/results": cirr_results_volume,
    },
)
def run_cirr_test1(batch_size: int = 64, workers: int = 1):
    """
    Run the repository's CIRR Test1 prediction pipeline and persist
    the four JSON prediction files in the dedicated CIRR results Volume.
    """
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if workers < 0:
        raise ValueError("workers cannot be negative")

    project = Path("/root/project")
    checkpoint = project / "checkpoints" / "pic2word_model.pt"
    eval_script = project / "src" / "eval_retrieval.py"
    cirr_root = project / "data" / "CIRR"

    # Keep Test1 outputs separate from Validation outputs.
    result_root = Path("/results/cirr_test1")
    log_root = result_root / "logs"
    persistent_prediction_dir = result_root / "predictions"

    required_paths = [
        checkpoint,
        eval_script,
        cirr_root / "captions" / "cap.rc2.test1.json",
        cirr_root / "image_splits" / "split.rc2.test1.json",
        cirr_root / "img_raw" / "test1",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required CIRR Test1 resource is missing: {path}"
            )

    # Confirm the expected Test1 annotation structures before using GPU.
    with (cirr_root / "captions" / "cap.rc2.test1.json").open(
        "r", encoding="utf-8"
    ) as f:
        test_queries = json.load(f)

    with (cirr_root / "image_splits" / "split.rc2.test1.json").open(
        "r", encoding="utf-8"
    ) as f:
        test_gallery = json.load(f)

    if not isinstance(test_queries, list) or not test_queries:
        raise ValueError("CIRR Test1 caption file must contain a non-empty JSON list")
    if not isinstance(test_gallery, dict) or not test_gallery:
        raise ValueError("CIRR Test1 gallery split must contain a non-empty JSON object")

    missing_pairid = [
        index for index, item in enumerate(test_queries)
        if not isinstance(item, dict) or "pairid" not in item
    ]
    if missing_pairid:
        raise ValueError(
            "CIRR Test1 caption records are missing pairid; "
            f"sample indices: {missing_pairid[:10]}"
        )

    pairids = [str(item["pairid"]) for item in test_queries]
    if len(pairids) != len(set(pairids)):
        raise ValueError("CIRR Test1 contains duplicate pairid values")

    log_root.mkdir(parents=True, exist_ok=True)
    persistent_prediction_dir.mkdir(parents=True, exist_ok=True)

    # eval_retrieval.py writes to ./res_cirr. Remove only this temporary
    # project-local output directory so stale predictions cannot be copied.
    temporary_prediction_dir = project / "res_cirr"
    if temporary_prediction_dir.exists():
        shutil.rmtree(temporary_prediction_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(project / "src")
        + os.pathsep
        + str(project)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    env["WANDB_MODE"] = "disabled"

    command = [
        "python", str(eval_script),
        "--eval-mode", "cirr_test",
        "--model", "ViT-L/14",
        "--resume", str(checkpoint),
        "--middle_dim", "512",
        "--n-layer", "2",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--workers", str(workers),
        "--precision", "amp",
        "--logs", str(log_root),
        "--name", "cirr_test1",
    ]

    print("=" * 72, flush=True)
    print("CIRR TEST1 PREDICTION GENERATION", flush=True)
    print("Queries:", len(test_queries), flush=True)
    print("Gallery images:", len(test_gallery), flush=True)
    print("Checkpoint:", checkpoint, flush=True)
    print("Temporary predictions:", temporary_prediction_dir, flush=True)
    print("Persistent results:", result_root, flush=True)
    print("Command:", " ".join(command), flush=True)
    print("=" * 72, flush=True)

    expected_files = [
        "composed.json",
        "image.json",
        "text.json",
        "mixture.json",
    ]

    try:
        subprocess.run(
            command,
            cwd=str(project),
            env=env,
            check=True,
        )

        # Validate every generated JSON before copying it to persistent storage.
        for filename in expected_files:
            source = temporary_prediction_dir / filename
            if not source.is_file():
                raise FileNotFoundError(
                    f"Test1 did not generate the expected prediction file: {source}"
                )

            with source.open("r", encoding="utf-8") as f:
                prediction = json.load(f)

            if not isinstance(prediction, dict):
                raise TypeError(
                    f"{filename} must contain a JSON object"
                )
            if prediction.get("version") != "rc2":
                raise ValueError(
                    f"{filename} has unexpected version: "
                    f"{prediction.get('version')!r}"
                )
            if prediction.get("metric") != "recall":
                raise ValueError(
                    f"{filename} has unexpected metric: "
                    f"{prediction.get('metric')!r}"
                )

            prediction_pairids = set(prediction) - {"version", "metric"}
            expected_pairids = set(pairids)
            if prediction_pairids != expected_pairids:
                missing = sorted(expected_pairids - prediction_pairids)[:10]
                extra = sorted(prediction_pairids - expected_pairids)[:10]
                raise ValueError(
                    f"{filename} pairid coverage mismatch. "
                    f"Missing sample: {missing}; extra sample: {extra}"
                )

            malformed = [
                pairid for pairid in pairids
                if (
                    not isinstance(prediction[pairid], list)
                    or len(prediction[pairid]) != 50
                    or not all(isinstance(image_id, str)
                               for image_id in prediction[pairid])
                )
            ]
            if malformed:
                raise ValueError(
                    f"{filename} has invalid top-50 predictions for "
                    f"pairid sample: {malformed[:10]}"
                )

            shutil.copy2(
                source,
                persistent_prediction_dir / filename,
            )
            print(
                "Validated and copied:",
                filename,
                "size:",
                source.stat().st_size,
                "bytes",
                flush=True,
            )

    finally:
        # Persist logs and any successfully validated prediction files,
        # including partial outputs if the evaluation or validation fails.
        cirr_results_volume.commit()

    print("\\nCIRR TEST1 PREDICTIONS COMPLETED", flush=True)
    print("Persistent predictions:", persistent_prediction_dir, flush=True)
    print("Log directory:", log_root, flush=True)

    return {
        "status": "completed",
        "benchmark": "CIRR Test1",
        "query_count": len(test_queries),
        "gallery_count": len(test_gallery),
        "prediction_files": expected_files,
        "predictions_dir": str(persistent_prediction_dir),
        "logs_dir": str(log_root),
    }

