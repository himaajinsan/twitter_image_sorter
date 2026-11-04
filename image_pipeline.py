"""Local image tagging, embedding, clustering and natural-language search.

The database is deliberately SQLite-only so the pipeline works offline and can
be copied together with the image collection. Heavy model dependencies are
loaded only when their corresponding command is used.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Sequence
from threading import Event
from gpu_runtime import onnx_providers, torch_device

# Hugging Face's cache uses symlinks, which Windows only allows in Developer
# Mode or an elevated shell. Fall back to copying files so model downloads
# (LFM2, Florence-2) never fail with WinError 1314 there.
if os.name == "nt":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Bidirectional EN<->JP translator used for semantic search queries.
TRANSLATION_MODEL = "LiquidAI/LFM2-350M-ENJP-MT"
TRANSLATION_SYSTEM_PROMPT = "Translate to English."

# Models are expensive to load (seconds each), so keep them alive for the
# lifetime of the process. Searches are usually repeated against one database.
_CLIP_CACHE: dict[tuple[str, bool], tuple] = {}
_TRANSLATION_CACHE: dict[tuple[str, bool], tuple] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# Normalized embeddings are cached as one float32 matrix so a query is a single
# BLAS matrix-vector product instead of a Python loop over SQLite rows.
_EMBEDDING_CACHE_MAX_ROWS = 500_000
_EMBEDDING_CACHE: dict[str, object] = {}
_EMBEDDING_CACHE_LOCK = threading.Lock()


class _EmbeddingIndex:
    __slots__ = ("signature", "ids", "matrix", "rows")

    def __init__(self, signature, ids, matrix, rows):
        self.signature = signature
        self.ids = ids
        self.matrix = matrix
        self.rows = rows


def _database_signature(database: str | Path) -> tuple:
    """Cheap fingerprint that changes whenever SQLite writes to the database."""
    base = Path(database)
    signature = []
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(base) + suffix)
        try:
            stat = path.stat()
        except OSError:
            signature.append(None)
        else:
            signature.append((stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def clear_embedding_cache():
    with _EMBEDDING_CACHE_LOCK:
        _EMBEDDING_CACHE.clear()


def _build_embedding_index(database: str | Path, signature) -> _EmbeddingIndex:
    import numpy as np

    db = connect_database(database)
    try:
        rows = db.execute(
            "SELECT id, path, cluster_id, caption, embedding, sequence_group "
            "FROM images WHERE embedding IS NOT NULL"
        ).fetchall()
    finally:
        db.close()

    if not rows:
        empty = np.empty((0, 0), dtype=np.float32)
        return _EmbeddingIndex(signature, np.empty(0, dtype=np.int64), empty, [])

    dimension = len(rows[0]["embedding"]) // 4
    ids, vectors, metadata = [], [], []
    skipped = 0
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        if vector.size != dimension:
            skipped += 1
            continue
        ids.append(row["id"])
        vectors.append(vector)
        metadata.append({
            "path": row["path"],
            "cluster_id": row["cluster_id"],
            "caption": row["caption"],
            "sequence_group": row["sequence_group"],
        })
    if skipped:
        print(
            f"[SEARCH] skipped {skipped} embeddings with a different dimension",
            flush=True,
        )
    matrix = np.stack(vectors) if vectors else np.empty((0, dimension), dtype=np.float32)
    # Embeddings are normalized when stored; normalize defensively so the dot
    # product is always a cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = (matrix / norms).astype(np.float32, copy=False)
    return _EmbeddingIndex(
        signature,
        np.asarray(ids, dtype=np.int64),
        matrix,
        metadata,
    )


def _get_embedding_index(database: str | Path) -> _EmbeddingIndex:
    signature = _database_signature(database)
    with _EMBEDDING_CACHE_LOCK:
        cached = _EMBEDDING_CACHE.get("index")
        if isinstance(cached, _EmbeddingIndex) and cached.signature == signature:
            return cached

    index = _build_embedding_index(database, signature)
    if len(index.rows) <= _EMBEDDING_CACHE_MAX_ROWS:
        with _EMBEDDING_CACHE_LOCK:
            _EMBEDDING_CACHE["index"] = index
    return index


def connect_database(path: str | Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            nsfw_score REAL,
            caption TEXT NOT NULL DEFAULT '',
            embedding BLOB,
            embedding_model TEXT NOT NULL DEFAULT '',
            cluster_id INTEGER,
            sequence_group INTEGER,
            sequence_index INTEGER,
            sequence_score REAL,
            analyzed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_images_cluster ON images(cluster_id);
        """
    )
    columns = {row["name"] for row in db.execute("PRAGMA table_info(images)")}
    for name, definition in (
        ("sequence_group", "INTEGER"),
        ("sequence_index", "INTEGER"),
        ("sequence_score", "REAL"),
    ):
        if name not in columns:
            db.execute(f"ALTER TABLE images ADD COLUMN {name} {definition}")
    if "caption_en" in columns:
        db.execute("ALTER TABLE images DROP COLUMN caption_en")
        db.commit()
    return db


def _boundary_score(left, right, device="cpu") -> float:
    import numpy as np

    width = min(64, left.shape[1], right.shape[1])
    left_strip = left[:, -width:].astype("float32") / 255.0
    right_strip = right[:, :width].astype("float32") / 255.0
    left_gray = left_strip.mean(axis=2)
    right_gray = right_strip.mean(axis=2)
    left_content = np.logical_or(
        left_gray < 0.92,
        left_strip.max(axis=2) - left_strip.min(axis=2) > 0.10,
    )
    right_content = np.logical_or(
        right_gray < 0.92,
        right_strip.max(axis=2) - right_strip.min(axis=2) > 0.10,
    )
    left_density = float(left_content.mean())
    right_density = float(right_content.mean())
    if max(left_density, right_density) < 0.08:
        return 0.0
    if abs(left_density - right_density) > 0.35:
        return 0.0
    if device != "cpu":
        import torch
        a = torch.from_numpy(left_strip).to(device)
        b = torch.from_numpy(right_strip).to(device)
        color = torch.mean(torch.abs(a[:, -1:] - b[:, :1])).item()
        gradient = torch.mean(torch.abs(
            (a[:, 1:] - a[:, :-1])[:, -1:] -
            (b[:, 1:] - b[:, :-1])[:, :1]
        )).item()
    else:
        color = float(np.mean(np.abs(left_strip[:, -1:] - right_strip[:, :1])))
        gradient = float(np.mean(np.abs(
            (left_strip[:, 1:] - left_strip[:, :-1])[:, -1:] -
            (right_strip[:, 1:] - right_strip[:, :-1])[:, :1]
        )))
    return max(0.0, 1.0 - (color * 0.65 + gradient * 0.35))


def detect_sequences(
    database: str | Path,
    *,
    use_gpu: bool = True,
    threshold: float = 0.92,
    cancel_event: Event | None = None,
    progress=None,
) -> tuple[int, int]:
    """Find up to four-image sequences using source-file modification times.

    Neighboring images are considered height-compatible when their heights
    differ by no more than one percent.
    """
    import numpy as np

    db = connect_database(database)
    database_rows = db.execute(
        "SELECT id, path, width, height FROM images WHERE width IS NOT NULL "
        "AND height IS NOT NULL"
    ).fetchall()
    rows = []
    for row in database_rows:
        try:
            file_modified_at = Path(row["path"]).stat().st_mtime
        except OSError as exc:
            rows.append({**dict(row), "_stat_error": str(exc)})
            continue
        rows.append({**dict(row), "_modified_at": file_modified_at})
    rows.sort(key=lambda row: row.get("_modified_at", float("inf")))
    db.execute(
        "UPDATE images SET sequence_group = NULL, sequence_index = NULL, "
        "sequence_score = NULL"
    )
    db.commit()
    device, mode = torch_device(use_gpu)
    print(f"[GPU] Sequence checker device: {mode} ({device})", flush=True)
    image_cache = {}
    group_id = 1
    groups = errors = 0

    def pixels(row):
        if row["id"] not in image_cache:
            image = _require_pillow().open(row["path"]).convert("RGB")
            image_cache[row["id"]] = np.asarray(image)
        return image_cache[row["id"]]

    valid_rows = []
    for row in rows:
        if "_modified_at" in row:
            valid_rows.append(row)
        else:
            errors += 1
            if progress:
                progress(
                    len(valid_rows) + errors,
                    len(rows),
                    groups,
                    errors,
                    row["path"],
                    row["_stat_error"],
                )

    runs = []
    current = []
    for row in valid_rows:
        if (
            current
            and (
                abs(row["height"] - current[-1]["height"])
                / max(row["height"], current[-1]["height"])
                > 0.01
                or row["_modified_at"] - current[-1]["_modified_at"] > 10
            )
        ):
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)

    processed = 0
    for run in runs:
        for start in range(0, len(run), 4):
            chunk = run[start:start + 4]
            processed += len(chunk)
            if cancel_event and cancel_event.is_set():
                break
            if len(chunk) < 2:
                continue
            best = None
            try:
                for sequence in itertools.permutations(chunk):
                    scores = [
                        _boundary_score(pixels(a), pixels(b), device)
                        for a, b in zip(sequence, sequence[1:])
                    ]
                    score = float(np.mean(scores))
                    if best is None or score > best[0]:
                        best = (score, sequence)
            except Exception as exc:
                errors += 1
                if progress:
                    progress(processed, len(rows), groups, errors, chunk[0]["path"], str(exc))
                continue
            if best and best[0] >= threshold:
                score, sequence = best
                for sequence_index, row in enumerate(sequence):
                    db.execute(
                        "UPDATE images SET sequence_group = ?, "
                        "sequence_index = ?, sequence_score = ? WHERE id = ?",
                        (group_id, sequence_index, score, row["id"]),
                    )
                db.commit()
                group_id += 1
                groups += 1
                if progress:
                    log_parts = []
                    for left, right in zip(sequence, sequence[1:]):
                        log_parts.extend((
                            Path(left["path"]).name,
                            f"{_boundary_score(pixels(left), pixels(right), device):.3f}",
                        ))
                    log_parts.append(Path(sequence[-1]["path"]).name)
                    progress(
                        processed,
                        len(rows),
                        groups,
                        errors,
                        " || ".join(log_parts),
                        "",
                    )
            if progress:
                progress(processed, len(rows), groups, errors, "", "")
        if cancel_event and cancel_event.is_set():
            break
    db.close()
    return groups, errors


def list_sequences(database: str | Path) -> list[dict]:
    """Return detected sequence groups and their ordered image paths."""
    db = connect_database(database)
    rows = db.execute(
        "SELECT sequence_group, sequence_index, path FROM images "
        "WHERE sequence_group IS NOT NULL ORDER BY sequence_group, sequence_index"
    ).fetchall()
    db.close()
    groups = {}
    for row in rows:
        groups.setdefault(row["sequence_group"], []).append(row["path"])
    return [
        {"sequence_group": group, "paths": paths}
        for group, paths in sorted(groups.items())
    ]


def clear_sequence_group(database: str | Path, sequence_group: int) -> int:
    """Remove sequence registration without deleting image records."""
    db = connect_database(database)
    try:
        cursor = db.execute(
            "UPDATE images SET sequence_group = NULL, sequence_index = NULL, "
            "sequence_score = NULL "
            "WHERE sequence_group = ?",
            (sequence_group,),
        )
        db.commit()
        return cursor.rowcount
    finally:
        db.close()


def reorder_sequence_group(
    database: str | Path,
    sequence_group: int,
    ordered_paths: Sequence[str | Path],
) -> None:
    """Persist a new image order for an existing sequence group."""
    paths = [str(Path(path)) for path in ordered_paths]
    if not 2 <= len(paths) <= 4 or len(set(paths)) != len(paths):
        raise ValueError("グループ順は重複しない2～4枚で指定してください。")
    db = connect_database(database)
    try:
        rows = db.execute(
            "SELECT path FROM images WHERE sequence_group = ? "
            "ORDER BY sequence_index",
            (sequence_group,),
        ).fetchall()
        if {row["path"] for row in rows} != set(paths):
            raise ValueError("指定した画像が現在のグループと一致しません。")
        db.executemany(
            "UPDATE images SET sequence_index = ? "
            "WHERE sequence_group = ? AND path = ?",
            [(index, sequence_group, path) for index, path in enumerate(paths)],
        )
        db.commit()
    finally:
        db.close()


def add_sequence_group(database: str | Path, paths: Sequence[str | Path]) -> int:
    """Register two to four existing image rows in the supplied order."""
    ordered_paths = [str(Path(path)) for path in paths]
    if not 2 <= len(ordered_paths) <= 4:
        raise ValueError("手動グループは2枚から4枚まで選択してください。")
    db = connect_database(database)
    try:
        placeholders = ",".join("?" for _ in ordered_paths)
        rows = db.execute(
            f"SELECT path, sequence_group FROM images WHERE path IN ({placeholders})",
            ordered_paths,
        ).fetchall()
        by_path = {row["path"]: row for row in rows}
        missing = [path for path in ordered_paths if path not in by_path]
        if missing:
            raise ValueError(
                "SQLiteに登録されていない画像があります: "
                + ", ".join(Path(path).name for path in missing)
            )
        already_grouped = [
            row for row in rows if row["sequence_group"] is not None
        ]
        if already_grouped:
            raise ValueError(
                "すでにグループ登録済みの画像があります: "
                + ", ".join(Path(row["path"]).name for row in already_grouped)
            )
        next_group = db.execute(
            "SELECT COALESCE(MAX(sequence_group), 0) + 1 "
            "FROM images WHERE sequence_group IS NOT NULL"
        ).fetchone()[0]
        db.executemany(
            "UPDATE images SET sequence_group = ?, sequence_index = ?, "
            "sequence_score = 1.0 WHERE path = ?",
            [(next_group, index, path) for index, path in enumerate(ordered_paths)],
        )
        db.commit()
        return int(next_group)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def image_sequence_status(database: str | Path, path: str | Path) -> int | None:
    """Return the registered sequence group for an image, if any."""
    db = connect_database(database)
    try:
        row = db.execute(
            "SELECT sequence_group FROM images WHERE path = ?",
            (str(Path(path)),),
        ).fetchone()
        return row["sequence_group"] if row else None
    finally:
        db.close()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images(folder: str | Path) -> Iterable[Path]:
    root = Path(folder).expanduser().resolve()
    yield from (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def _require_pillow():
    try:
        from PIL import Image
        return Image
    except ImportError as exc:
        raise RuntimeError("Pillow が必要です。requirements.txt をインストールしてください。") from exc


def _load_clip(model_name: str, use_gpu=True):
    key = (model_name, bool(use_gpu))
    cached = _CLIP_CACHE.get(key)
    if cached is not None:
        return cached
    with _MODEL_CACHE_LOCK:
        cached = _CLIP_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP には transformers と torch が必要です。"
                " `pip install transformers torch` を実行してください。"
            ) from exc
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        device, mode = torch_device(use_gpu)
        model.to(device)
        model.eval()
        cached = (processor, model, device, mode)
        _CLIP_CACHE[key] = cached
        return cached


def _clip_embedding_from_image(image, processor, model, device):
    import torch
    inputs = {
        key: value.to(device)
        for key, value in processor(images=image, return_tensors="pt").items()
    }
    with torch.no_grad():
        vector = _clip_image_features(model, inputs)
        vector = vector / vector.norm(dim=-1, keepdim=True)
    return vector[0].cpu().numpy().astype("float32")


def _clip_embedding(path: Path, processor, model, device):
    image = _require_pillow().open(path).convert("RGB")
    try:
        return _clip_embedding_from_image(image, processor, model, device)
    finally:
        image.close()


def _clip_embeddings_from_images(images, processor, model, device):
    """Encode several images in one GPU call; far faster than one at a time."""
    import torch
    inputs = {
        key: value.to(device)
        for key, value in processor(images=images, return_tensors="pt").items()
    }
    with torch.no_grad():
        vectors = _clip_image_features(model, inputs)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    return vectors.cpu().numpy().astype("float32")


def _load_translation(model_name: str = TRANSLATION_MODEL, use_gpu=True):
    key = (model_name, bool(use_gpu))
    cached = _TRANSLATION_CACHE.get(key)
    if cached is not None:
        return cached
    with _MODEL_CACHE_LOCK:
        cached = _TRANSLATION_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("翻訳には transformers と torch が必要です。") from exc
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device, mode = torch_device(use_gpu)
        # LFM2 checkpoints are stored in bfloat16, which older GPUs cannot use.
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
        print(f"[GPU] Translation device: {mode} ({device})", flush=True)
        cached = (tokenizer, model, device)
        _TRANSLATION_CACHE[key] = cached
        return cached


def _translate(
    text: str,
    translator,
    max_new_tokens=256,
    system_prompt: str = TRANSLATION_SYSTEM_PROMPT,
) -> str:
    if not text.strip():
        return ""
    import torch
    tokenizer, model, device = translator
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=1.0,
            min_p=0.1,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _clip_image_features(model, inputs):
    """Handle both legacy tensor and newer structured Transformers outputs."""
    output = model.get_image_features(**inputs)
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return _project_clip_output(model, output.pooler_output, "visual_projection")
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        pooled = output.last_hidden_state[:, 0]
        return _project_clip_output(model, pooled, "visual_projection")
    return output


def _clip_text_features(model, inputs):
    output = model.get_text_features(**inputs)
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return _project_clip_output(model, output.pooler_output, "text_projection")
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        pooled = output.last_hidden_state[:, 0]
        return _project_clip_output(model, pooled, "text_projection")
    return output


def _project_clip_output(model, value, projection_name):
    projection = getattr(model, projection_name, None)
    if projection is None:
        return value
    expected_width = getattr(projection, "in_features", None)
    if expected_width is None or value.shape[-1] == expected_width:
        return projection(value)
    # Some Transformers versions return the already projected 512-D vector
    # inside pooler_output, so applying the 768->512 layer again is invalid.
    return value


def analyze_folder(
    folder: str | Path,
    database: str | Path,
    *,
    clip_model: str = "openai/clip-vit-base-patch32",
    resume: bool = True,
    use_gpu: bool = True,
    batch_size: int = 16,
    cancel_event: Event | None = None,
    progress=None,
) -> tuple[int, int]:
    """Analyze images incrementally. Returns (processed, errors).

    CLIP runs on batches of ``batch_size`` images because the GPU is mostly
    idle when encoding one image per call.
    """
    db = connect_database(database)
    clip_processor = clip = None
    clip_device = "cpu"
    if clip_model:
        clip_processor, clip, clip_device, clip_mode = _load_clip(clip_model, use_gpu=use_gpu)
        print(f"[GPU] CLIP device: {clip_mode} ({clip_device})", flush=True)

    files = list(iter_images(folder))
    pending = []
    for index, path in enumerate(files, 1):
        if cancel_event and cancel_event.is_set():
            break
        if resume and db.execute("SELECT 1 FROM images WHERE path = ?", (str(path),)).fetchone():
            continue
        pending.append((index, path))

    processed = errors = 0
    batch_size = max(1, batch_size)
    for start in range(0, len(pending), batch_size):
        if cancel_event and cancel_event.is_set():
            break
        loaded = []
        for index, path in pending[start:start + batch_size]:
            if cancel_event and cancel_event.is_set():
                break
            try:
                image = _require_pillow().open(path).convert("RGB")
            except Exception as exc:
                errors += 1
                if progress:
                    progress(index, len(files), processed, errors, str(path), str(exc))
                    progress(index, len(files), processed, errors, "", "")
                continue
            loaded.append((index, path, image))

        embeddings = {}
        if clip and loaded:
            try:
                vectors = _clip_embeddings_from_images(
                    [image for _, _, image in loaded], clip_processor, clip, clip_device
                )
                embeddings = {
                    path: vector for (_, path, _), vector in zip(loaded, vectors)
                }
            except Exception as exc:
                # A corrupt or unusual image must not lose the whole batch.
                print(
                    f"[ANALYZE] batch embedding failed ({exc}); retrying one by one",
                    flush=True,
                )
                for _, path, image in loaded:
                    try:
                        embeddings[path] = _clip_embedding_from_image(
                            image, clip_processor, clip, clip_device
                        )
                    except Exception as item_exc:
                        embeddings[path] = item_exc

        inserted = False
        for index, path, image in loaded:
            try:
                embedding = embeddings.get(path) if clip else None
                if isinstance(embedding, Exception):
                    raise embedding
                db.execute(
                    """INSERT INTO images(path, sha256, width, height, embedding, embedding_model)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (str(path), _sha256(path), image.width, image.height,
                     sqlite3.Binary(embedding.astype("float32").tobytes())
                     if embedding is not None else None,
                     clip_model if embedding is not None else ""),
                )
                inserted = True
                processed += 1
            except Exception as exc:
                errors += 1
                if progress:
                    progress(index, len(files), processed, errors, str(path), str(exc))
            finally:
                image.close()
            if progress:
                progress(index, len(files), processed, errors, "", "")
        if inserted:
            db.commit()
    db.close()
    clear_embedding_cache()
    return processed, errors


def score_nsfw_folder(
    folder: str | Path,
    database: str | Path,
    *,
    use_gpu: bool = True,
    cancel_event: Event | None = None,
    progress=None,
) -> tuple[int, int]:
    """Run NudeNet and save the highest adult-class score for each image."""
    try:
        from nudenet import NudeDetector
    except ImportError as exc:
        raise RuntimeError(
            "NSFWスコア解析には nudenet と onnxruntime が必要です。"
        ) from exc

    providers, mode = onnx_providers(use_gpu)
    try:
        detector = NudeDetector(providers=providers)
        print(f"[GPU] NudeNet score providers: {detector.onnx_session.get_providers()}", flush=True)
    except Exception as exc:
        if providers == ["CPUExecutionProvider"]:
            raise
        print(f"[GPU] NudeNet score CUDA initialization failed; using CPU: {exc}", flush=True)
        detector = NudeDetector(providers=["CPUExecutionProvider"])
        mode = "CPU (CUDA初期化失敗)"

    adult_labels = {
        "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED", "ANUS_EXPOSED",
        "MALE_BREAST_EXPOSED",
    }
    db = connect_database(database)
    files = list(iter_images(folder))
    done = errors = 0
    print(f"[GPU] NudeNet score mode: {mode}", flush=True)
    for index, path in enumerate(files, 1):
        if cancel_event and cancel_event.is_set():
            break
        try:
            detections = detector.detect(str(path))
            score = max(
                (float(item.get("score", 0.0)) for item in detections
                 if str(item.get("class", item.get("label", ""))).upper() in adult_labels),
                default=0.0,
            )
            db.execute(
                "UPDATE images SET nsfw_score = ? WHERE path = ?",
                (score, str(path)),
            )
            db.commit()
            done += 1
        except Exception as exc:
            errors += 1
            if progress:
                progress(index, len(files), done, errors, str(path), str(exc))
        if progress:
            progress(index, len(files), done, errors, "", "")
    db.close()
    return done, errors


def assign_clusters(
    database: str | Path,
    *,
    clusters: int = 20,
    keep_sequence_groups: bool = True,
) -> int:
    """Assign cluster labels to analyzed images.

    Sequence groups are the image sets that the continuity checker matched by
    boundary score.  When ``keep_sequence_groups`` is true, every group is
    collapsed into one weighted centroid before clustering, so all of its
    images always receive the same cluster label.  The centroid weight is the
    group's stored match score multiplied by its member count: strongly
    matched groups have more influence, while a group still counts roughly
    like the individual images it contains.
    """
    try:
        import numpy as np
        from sklearn.cluster import KMeans, MiniBatchKMeans
    except ImportError as exc:
        raise RuntimeError("クラスタリングには numpy と scikit-learn が必要です。") from exc
    db = connect_database(database)
    rows = db.execute(
        "SELECT id, embedding, sequence_group, sequence_score FROM images "
        "WHERE embedding IS NOT NULL ORDER BY id"
    ).fetchall()
    if not rows:
        db.close()
        raise RuntimeError("埋め込みがありません。先に analyze を実行してください。")
    matrix = np.stack([
        np.frombuffer(row["embedding"], dtype=np.float32) for row in rows
    ]).astype(np.float32)
    # Embeddings are normally L2-normalized already. Normalize defensively so
    # older databases and partial imports still behave as cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    unit_vectors = []
    unit_ids = []
    weights = []
    if keep_sequence_groups:
        group_order = []
        group_indices = {}
        for index, row in enumerate(rows):
            group = row["sequence_group"]
            if group is None:
                continue
            group = int(group)
            if group not in group_indices:
                group_indices[group] = []
                group_order.append(group)
            group_indices[group].append(index)

        for index, row in enumerate(rows):
            group = row["sequence_group"]
            if group is None:
                unit_vectors.append(matrix[index])
                unit_ids.append([row["id"]])
                weights.append(1.0)

        for group in group_order:
            indices = group_indices[group]
            vectors = matrix[indices]
            centroid = vectors.mean(axis=0)
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            stored_scores = [
                float(rows[index]["sequence_score"])
                for index in indices
                if rows[index]["sequence_score"] is not None
            ]
            if stored_scores:
                match = float(np.mean(stored_scores))
            else:
                # Groups created before sequence_score was stored fall back
                # to the group's own embedding coherence.
                match = float(np.mean(vectors @ centroid))
            match = min(1.0, max(0.0, match))
            unit_vectors.append(centroid)
            unit_ids.append([rows[index]["id"] for index in indices])
            weights.append(max(1e-3, len(indices) * max(match, 0.05)))
    else:
        for index, row in enumerate(rows):
            unit_vectors.append(matrix[index])
            unit_ids.append([row["id"]])
            weights.append(1.0)

    unit_matrix = np.stack(unit_vectors).astype(np.float32)
    unit_norms = np.linalg.norm(unit_matrix, axis=1, keepdims=True)
    unit_matrix = unit_matrix / np.maximum(unit_norms, 1e-12)
    count = min(max(1, int(clusters)), len(unit_matrix))
    weight_array = np.asarray(weights, dtype=np.float64)

    if count <= 1:
        labels = np.zeros(len(unit_matrix), dtype=np.int32)
    elif count >= len(unit_matrix):
        # More requested clusters than matcheable units: each unit becomes
        # its own cluster instead of creating empty clusters.
        labels = np.arange(len(unit_matrix), dtype=np.int32)
    elif len(unit_matrix) <= 20000:
        labels = KMeans(
            n_clusters=count, random_state=42, n_init="auto"
        ).fit_predict(unit_matrix, sample_weight=weight_array).astype(np.int32)
    else:
        # MiniBatch keeps very large libraries practical. It is still
        # similarity-based because the vectors are normalized.
        labels = MiniBatchKMeans(
            n_clusters=count, random_state=42, n_init="auto", batch_size=256
        ).fit_predict(unit_matrix, sample_weight=weight_array).astype(np.int32)

    # Rows without an embedding must not keep a label from an older run.
    db.execute("UPDATE images SET cluster_id = NULL")
    db.executemany(
        "UPDATE images SET cluster_id = ? WHERE id = ?",
        [
            (int(label), image_id)
            for label, ids in zip(labels, unit_ids)
            for image_id in ids
        ],
    )
    db.commit()
    db.close()
    clear_embedding_cache()
    return int(len(np.unique(labels)))


def _expand_sequence_results(db, rows, limit):
    results = []
    seen_groups = set()
    for row in rows:
        group = row["sequence_group"]
        if group is None:
            results.append({
                "path": row["path"],
                "paths": [row["path"]],
                "cluster_id": row["cluster_id"],
                "caption": row["caption"],
                "confidence": row["confidence"] or 0.0,
            })
            continue
        if group in seen_groups:
            continue
        seen_groups.add(group)
        members = db.execute(
            "SELECT path FROM images WHERE sequence_group = ? "
            "ORDER BY sequence_index",
            (group,),
        ).fetchall()
        paths = [member["path"] for member in members]
        results.append({
            "path": paths[0],
            "paths": paths,
            "cluster_id": row["cluster_id"],
            "caption": row["caption"],
            "confidence": row["confidence"] or 0.0,
        })
        if len(results) >= limit:
            break
    return results[:limit]


def caption_folder(
    folder: str | Path,
    database: str | Path,
    *,
    model_name: str = "florence-community/Florence-2-base",
    resume: bool = True,
    use_gpu: bool = True,
    cancel_event: Event | None = None,
    progress=None,
) -> tuple[int, int]:
    """Generate local Florence-2 captions and store them in the existing DB."""
    try:
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Florence-2 には transformers と torch が必要です。"
            " `pip install transformers torch` を実行してください。"
        ) from exc
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        device, _ = torch_device(use_gpu)
        print(f"[GPU] Florence-2 device: {device}", flush=True)
        model = Florence2ForConditionalGeneration.from_pretrained(model_name).to(device)
    except AttributeError as exc:
        if "forced_bos_token_id" in str(exc):
            raise RuntimeError(
                "Florence-2とTransformersの互換性がありません。"
                " 起動し直してTransformers 4.57.6へ更新してください。"
            ) from exc
        raise
    model.eval()
    db = connect_database(database)
    files = list(iter_images(folder))
    done = errors = 0
    for index, path in enumerate(files, 1):
        if cancel_event and cancel_event.is_set():
            break
        row = db.execute("SELECT id, caption FROM images WHERE path = ?", (str(path),)).fetchone()
        if not row or (resume and row["caption"]):
            continue
        try:
            image = _require_pillow().open(path).convert("RGB")
            inputs = {
                key: value.to(device)
                for key, value in processor(
                    text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt"
                ).items()
            }
            with torch.no_grad():
                generated = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=128,
                    num_beams=3,
                )
            result = processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                result, task="<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height)
            )
            caption = parsed.get("<MORE_DETAILED_CAPTION>", result).strip()
            db.execute(
                "UPDATE images SET caption = ? WHERE id = ?",
                (caption, row["id"]),
            )
            db.commit()
            done += 1
        except Exception as exc:
            errors += 1
            if progress:
                progress(index, len(files), done, errors, str(path), str(exc))
        if progress:
            progress(index, len(files), done, errors, "", "")
    db.close()
    clear_embedding_cache()
    return done, errors


def search(
    database: str | Path,
    query: str,
    *,
    limit: int = 50,
    clip_model: str | None = None,
    use_gpu: bool = True,
    translation_callback=None,
) -> list[dict]:
    """Search Japanese/English natural-language terms locally.

    Keyword matching is always available. If query embedding support is added
    later, this stable result shape can be extended without changing the DB.
    """
    db = connect_database(database)
    if clip_model:
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("意味検索には numpy と torch が必要です。") from exc
        processor, model, device, mode = _load_clip(clip_model, use_gpu=use_gpu)
        print(f"[GPU] CLIP search device: {mode} ({device})", flush=True)
        if any("\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in query):
            original_query = query
            translator = _load_translation(use_gpu=use_gpu)
            query = _translate(query, translator)
            print(f"[SEARCH] translated query: {query}", flush=True)
            if translation_callback:
                translation_callback(original_query, query)
        with torch.no_grad():
            inputs = {
                key: value.to(device)
                for key, value in processor(text=[query], return_tensors="pt", padding=True).items()
            }
            vector = _clip_text_features(model, inputs)
            vector = (vector / vector.norm(dim=-1, keepdim=True))[0].cpu().numpy().astype("float32")
        index = _get_embedding_index(database)
        if index.matrix.shape[0] == 0:
            db.close()
            return []
        scores = index.matrix @ vector
        total = scores.shape[0]
        wanted = min(total, max(limit * 8, 64))
        while True:
            if wanted >= total:
                candidate_indices = np.argsort(-scores)
            else:
                candidate_indices = np.argpartition(-scores, wanted - 1)[:wanted]
                candidate_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
            result_rows = [
                {
                    "path": index.rows[position]["path"],
                    "cluster_id": index.rows[position]["cluster_id"],
                    "caption": index.rows[position]["caption"],
                    "confidence": float(scores[position]),
                    "sequence_group": index.rows[position]["sequence_group"],
                }
                for position in candidate_indices
            ]
            result = _expand_sequence_results(db, result_rows, limit)
            if len(result) >= limit or wanted >= total:
                break
            # Sequence groups can collapse several candidates into one result,
            # so widen the candidate window until enough rows are found.
            wanted = min(total, wanted * 4)
        db.close()
        return result
    original_query = query
    translated_query = ""
    if any("\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in query):
        original_query = query
        translator = _load_translation(use_gpu=use_gpu)
        translated_query = _translate(query, translator)
        print(f"[SEARCH] translated keyword query: {translated_query}", flush=True)
        if translation_callback:
            translation_callback(original_query, translated_query)

    stopwords = {
        "a", "an", "and", "are", "as", "at", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "the", "to", "with",
    }
    terms = []
    for text in (original_query, translated_query):
        for term in re.findall(r"[A-Za-z0-9_]+|[一-龯ぁ-んァ-ン]+", text.lower()):
            if term not in stopwords and (len(term) > 1 or not term.isascii()):
                if term not in terms:
                    terms.append(term)
    if not terms:
        db.close()
        return []

    clauses = []
    params: list[str] = []
    for term in terms:
        pattern = f"%{term}%"
        clauses.append(
            "(lower(i.path) LIKE ? OR lower(i.caption) LIKE ?)"
        )
        params.extend([pattern, pattern])
    rows = db.execute(
        f"""SELECT i.path, i.cluster_id, i.caption, i.sequence_group, 0.0 AS confidence
            FROM images i
            WHERE {' OR '.join(clauses)}
            ORDER BY i.path LIMIT ?""",
        (*params, limit),
    ).fetchall()
    result = _expand_sequence_results(db, rows, limit)
    db.close()
    return result
