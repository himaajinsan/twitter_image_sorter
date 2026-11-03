import os
import shutil
import threading
import queue
import traceback
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from gpu_runtime import onnx_providers

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
TRASH_PREFIX = ".trashed-"
DUPLICATE_SUFFIX = re.compile(r"\(\d+\)$")
def build_detector(use_gpu=True):
    import onnxruntime
    import nudenet
    from nudenet import NudeDetector

    providers, mode = onnx_providers(use_gpu)
    can_use_gpu = mode == "GPU"
    print(f"[GPU] NudeNet requested mode: {mode}", flush=True)

    if not getattr(NudeDetector.__init__, "_gpu_aware_patched", False):
        def _gpu_aware_init(self, model_path=None, providers=None, inference_resolution=320):
            if providers is None:
                available_providers = onnxruntime.get_available_providers()
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]
            model_path = model_path or os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx")
            self.onnx_session = onnxruntime.InferenceSession(model_path, providers=providers)
            active_providers = self.onnx_session.get_providers()
            if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in active_providers:
                raise RuntimeError(
                    f"CUDAExecutionProvider could not be activated; active providers: {active_providers}"
                )
            model_inputs = self.onnx_session.get_inputs()
            self.input_width = inference_resolution
            self.input_height = inference_resolution
            self.input_name = model_inputs[0].name

        _gpu_aware_init._gpu_aware_patched = True
        NudeDetector.__init__ = _gpu_aware_init

    try:
        detector = NudeDetector(providers=providers)
        print(f"[GPU] NudeNet active providers: {detector.onnx_session.get_providers()}", flush=True)
        return detector, mode
    except RuntimeError as exc:
        if not can_use_gpu or "CUDAExecutionProvider" not in str(exc):
            raise
        # The provider can be advertised by ONNX Runtime while its CUDA DLLs
        # or driver are unusable. Keep the sorter usable and report the mode.
        detector = NudeDetector(providers=["CPUExecutionProvider"])
        print(f"[GPU] NudeNet active providers: {detector.onnx_session.get_providers()}", flush=True)
        return detector, "CPU (CUDA初期化失敗)"


class App:
    def __init__(self, root):
        self.root = root
        root.title("画像あれこれツール")
        root.geometry("900x760")
        root.minsize(760, 600)
        self.events = queue.Queue()
        self.stop_requested = False
        self.pipeline_running = False
        self.pipeline_cancel = threading.Event()
        self.sequence_running = False
        self.sequence_cancel = threading.Event()
        self.last_search_translation = None
        self.pipeline_use_gpu = tk.BooleanVar(value=True)
        self.source = tk.StringVar()
        self.output = tk.StringVar()
        self.threshold = tk.StringVar(value="0.30")
        self.mode = tk.StringVar(value="copy")
        self.use_gpu = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="フォルダを選択してね")
        self.progress = tk.DoubleVar(value=0)
        self.pipeline_folder = tk.StringVar()
        self.database = tk.StringVar()
        self.clip_model = tk.StringVar(value="openai/clip-vit-base-patch32")
        self.florence_model = tk.StringVar(value="florence-community/Florence-2-base")
        self.pipeline_threshold = tk.StringVar(value="0.35")
        self.cluster_count = tk.StringVar(value="20")
        self.search_query = tk.StringVar()
        self.pipeline_status = tk.StringVar(value="解析フォルダとSQLiteデータベースを指定してください")

        f = ttk.Frame(root, padding=14)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)
        tabs = ttk.Notebook(f)
        tabs.grid(row=0, column=0, columnspan=3, sticky="nsew")
        sorter_tab = ttk.Frame(tabs, padding=10)
        pipeline_tab = ttk.Frame(tabs, padding=10)
        sequence_tab = ttk.Frame(tabs, padding=10)
        thumbnail_tab = ttk.Frame(tabs, padding=10)
        tabs.add(sorter_tab, text="NudeNet仕分け")
        tabs.add(pipeline_tab, text="タグ・特徴量・検索")
        tabs.add(sequence_tab, text="連続画像チェッカー")
        tabs.add(thumbnail_tab, text="画像サムネイル一覧")
        self._build_sorter_ui(sorter_tab)
        self._build_pipeline_ui(pipeline_tab)
        self._build_sequence_ui(sequence_tab)
        self._build_thumbnail_ui(thumbnail_tab)
        f.rowconfigure(0, weight=1)
        root.after(150, self.poll)

    def _build_sorter_ui(self, f):
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="変更元フォルダ").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.source).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(f, text="フォルダを開く…", command=self.pick_source).grid(row=0, column=2)
        ttk.Label(f, text="処理結果フォルダ").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.output).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(f, text="フォルダを開く…", command=self.pick_output).grid(row=1, column=2)

        ttk.Label(f, text="成人向け閾値 (0–1)").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(f, textvariable=self.threshold, width=10).grid(row=2, column=1, sticky="w", padx=8)
        ttk.Label(f, text="数値が高い = より明らかなえっち画像を選別 (ヒューリスティックだよ)").grid(row=2, column=1, sticky="w", padx=(100, 0))

        gpu_state = self._gpu_available()
        self.gpu_toggle = ttk.Checkbutton(
            f,
            text="GPU を使う（利用可能なら）",
            variable=self.use_gpu,
            state="normal" if gpu_state else "disabled",
        )
        self.gpu_toggle.grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 8))
        if not gpu_state:
            self.use_gpu.set(False)
            self.status.set("GPU が使えないため CPU で動作します")

        lf = ttk.LabelFrame(f, text="変更元フォルダの扱い", padding=6)
        lf.grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Radiobutton(lf, text="コピー (安全、オリジナルは残るよ)", variable=self.mode, value="copy").pack(anchor="w")
        ttk.Radiobutton(lf, text="移動 (オリジナルは消えるよ)", variable=self.mode, value="move").pack(anchor="w")

        actions = ttk.LabelFrame(f, text="前処理", padding=8)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(actions, text='以下の文字列のあるファイルを消すよ: ".trashed- **" or  ** "(number)"').pack(anchor="w")
        self.cleanup_btn = ttk.Button(actions, text="前処理を実行", command=self.cleanup)
        self.cleanup_btn.pack(anchor="w", pady=(6, 0))
        ttk.Label(actions, text="例だよ: .trashed-123.jpg,（ゴミ箱ファイル） image(1).png, photo(23).webp（重複ファイル）").pack(anchor="w")

        self.start_btn = ttk.Button(f, text="仕分け開始", command=self.start)
        self.start_btn.grid(row=6, column=0, sticky="w", pady=7)
        self.stop_btn = ttk.Button(f, text="中断", command=self.stop, state="disabled")
        self.stop_btn.grid(row=6, column=1, sticky="w", padx=8, pady=7)
        ttk.Progressbar(f, variable=self.progress, maximum=100).grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(f, textvariable=self.status, wraplength=700).grid(row=8, column=0, columnspan=3, sticky="w")
        ttk.Label(f, text="Log").grid(row=9, column=0, sticky="w", pady=(8, 2))
        self.log = tk.Text(f, height=10, wrap="word", state="disabled")
        self.log.grid(row=10, column=0, columnspan=3, sticky="nsew")
        f.rowconfigure(10, weight=1)

    def _build_pipeline_ui(self, f):
        f.columnconfigure(1, weight=1)
        f.rowconfigure(13, weight=1)
        ttk.Label(f, text="解析する画像フォルダ").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.pipeline_folder).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(f, text="フォルダを開く…", command=self.pick_pipeline_folder).grid(row=0, column=2)
        ttk.Label(f, text="SQLiteデータベース").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.database).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(f, text="保存先…", command=self.pick_database).grid(row=1, column=2)

        model_frame = ttk.LabelFrame(f, text="モデル設定（初回はモデルのダウンロードが必要）", padding=8)
        model_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=8)
        model_frame.columnconfigure(1, weight=1)
        ttk.Label(model_frame, text="CLIPモデル").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(model_frame, textvariable=self.clip_model).grid(row=0, column=1, columnspan=2, sticky="ew", padx=8)
        ttk.Label(model_frame, text="Florence-2モデル").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(model_frame, textvariable=self.florence_model).grid(row=1, column=1, columnspan=2, sticky="ew", padx=8)
        settings = ttk.Frame(f)
        settings.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        self.pipeline_gpu_toggle = ttk.Checkbutton(
            settings, text="GPUを使う", variable=self.pipeline_use_gpu
        )
        self.pipeline_gpu_toggle.pack(side="left", padx=(0, 10))
        ttk.Label(settings, text="クラスタ数").pack(side="left")
        ttk.Entry(settings, textvariable=self.cluster_count, width=8).pack(side="left", padx=(5, 15))
        self.analyze_btn = ttk.Button(settings, text="CLIP解析", command=self.start_pipeline_analysis)
        self.analyze_btn.pack(side="left", padx=3)
        self.caption_btn = ttk.Button(settings, text="Florence-2キャプション", command=self.start_caption)
        self.caption_btn.pack(side="left", padx=3)
        self.nsfw_score_btn = ttk.Button(settings, text="NSFWスコア解析", command=self.start_nsfw_scoring)
        self.nsfw_score_btn.pack(side="left", padx=3)
        self.cluster_btn = ttk.Button(settings, text="クラスタリング", command=self.start_clustering)
        self.cluster_btn.pack(side="left", padx=3)
        self.pipeline_stop_btn = ttk.Button(settings, text="解析を中断", command=self.stop_pipeline, state="disabled")
        self.pipeline_stop_btn.pack(side="left", padx=8)

        search_frame = ttk.LabelFrame(f, text="自然言語検索", padding=8)
        search_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)
        search_frame.columnconfigure(0, weight=1)
        ttk.Entry(search_frame, textvariable=self.search_query).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.keyword_search_btn = ttk.Button(search_frame, text="タグ・キャプション検索", command=lambda: self.start_search(False))
        self.keyword_search_btn.grid(row=0, column=1, padx=3)
        self.semantic_search_btn = ttk.Button(search_frame, text="CLIP意味検索", command=lambda: self.start_search(True))
        self.semantic_search_btn.grid(row=0, column=2, padx=3)
        ttk.Label(search_frame, text="例: ケモノ 背景 線画 / a furry character in a forest").grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        ttk.Label(f, textvariable=self.pipeline_status, wraplength=820).grid(row=5, column=0, columnspan=3, sticky="w", pady=4)
        self.pipeline_progress = ttk.Progressbar(f, maximum=100)
        self.pipeline_progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(f, text="検索結果 / 解析ログ").grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 2))
        self.pipeline_log = tk.Text(f, height=12, wrap="word", state="disabled")
        self.pipeline_log.grid(row=8, column=0, columnspan=3, sticky="nsew")
        f.rowconfigure(8, weight=1)

    def _build_sequence_ui(self, f):
        f.columnconfigure(1, weight=1)
        f.rowconfigure(6, weight=1)
        ttk.Label(f, text="解析する画像フォルダ").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.pipeline_folder).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(f, text="フォルダを開く…", command=self.pick_pipeline_folder).grid(row=0, column=2)
        ttk.Label(f, text="SQLiteデータベース").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.database).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(f, text="保存先…", command=self.pick_database).grid(row=1, column=2)
        settings = ttk.LabelFrame(f, text="判定設定", padding=8)
        settings.grid(row=2, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(settings, text="境界連続スコア").pack(side="left")
        self.sequence_threshold = tk.StringVar(value="0.92")
        ttk.Entry(settings, textvariable=self.sequence_threshold, width=8).pack(side="left", padx=6)
        self.sequence_gpu_toggle = ttk.Checkbutton(
            settings, text="GPUを使う", variable=self.pipeline_use_gpu
        )
        self.sequence_gpu_toggle.pack(side="left", padx=8)
        self.sequence_start_btn = ttk.Button(
            settings, text="連続画像を判定", command=self.start_sequence_check
        )
        self.sequence_start_btn.pack(side="left", padx=3)
        self.sequence_stop_btn = ttk.Button(
            settings, text="中断", command=self.stop_sequence_check, state="disabled"
        )
        self.sequence_stop_btn.pack(side="left", padx=3)
        self.sequence_status = tk.StringVar(
            value="ファイル更新日時順・隣接10秒以内・高さ差1%以内の候補を全探索します"
        )
        ttk.Label(f, textvariable=self.sequence_status, wraplength=820).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=4
        )
        self.sequence_progress = ttk.Progressbar(f, maximum=100)
        self.sequence_progress.grid(row=4, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(f, text="連続画像チェッカーログ").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(6, 2)
        )
        self.sequence_log = tk.Text(f, height=12, wrap="word", state="disabled")
        self.sequence_log.grid(row=6, column=0, columnspan=3, sticky="nsew")
        list_frame = ttk.LabelFrame(f, text="判定済み画像一覧", padding=6)
        list_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.sequence_list = ttk.Treeview(
            list_frame, columns=("group", "count", "paths"), show="headings"
        )
        self.sequence_list.heading("group", text="グループ")
        self.sequence_list.heading("count", text="枚数")
        self.sequence_list.heading("paths", text="画像（順番）")
        self.sequence_list.column("group", width=80, anchor="center")
        self.sequence_list.column("count", width=60, anchor="center")
        self.sequence_list.column("paths", width=700, anchor="w")
        self.sequence_list.grid(row=0, column=0, sticky="nsew")
        sequence_scroll = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.sequence_list.yview
        )
        sequence_scroll.grid(row=0, column=1, sticky="ns")
        self.sequence_list.configure(yscrollcommand=sequence_scroll.set)
        ttk.Button(
            list_frame, text="一覧を更新", command=self.refresh_sequence_list
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(
            list_frame, text="グループを確認", command=self.show_sequence_previews
        ).grid(row=1, column=0, sticky="w", padx=(100, 0), pady=(6, 0))
        ttk.Button(
            list_frame, text="手動でグループ追加", command=self.add_manual_sequence
        ).grid(row=1, column=0, sticky="w", padx=(220, 0), pady=(6, 0))
        f.rowconfigure(7, weight=1)

    def _build_thumbnail_ui(self, f):
        try:
            from PIL import Image, ImageTk
        except ImportError as exc:
            ttk.Label(
                f, text=f"サムネイル表示には Pillow が必要です: {exc}"
            ).grid(row=0, column=0, sticky="w")
            return
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=1)
        folder_var = self.pipeline_folder
        folder_controls = ttk.Frame(f)
        folder_controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        folder_controls.columnconfigure(1, weight=1)
        ttk.Label(folder_controls, text="表示フォルダ").grid(
            row=0, column=0, sticky="w"
        )
        folder_entry = ttk.Entry(folder_controls, textvariable=folder_var)
        folder_entry.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(
            f,
            text="1つのフォルダ内の画像を、横幅いっぱいの隙間なしサムネイルで表示します",
        ).grid(row=1, column=0, sticky="w", pady=(0, 6))
        canvas = tk.Canvas(f, background="#202020", highlightthickness=0)
        canvas.grid(row=2, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=2, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        selection_status = tk.StringVar(
            value="画像をクリックすると番号とファイル名を表示します"
        )
        ttk.Label(f, textvariable=selection_status, anchor="center").grid(
            row=3, column=0, sticky="ew", pady=(4, 0)
        )
        from collections import OrderedDict

        files = []
        columns = 8
        layout_size = 0
        empty_item = None
        scan_generation = 0
        thumb_generation = 0
        last_folder = None
        scan_in_flight = False
        photo_cache = OrderedDict()
        pending = set()
        items = {}
        jobs = queue.Queue()
        results = queue.Queue()
        render_job = None
        resize_job = None
        scan_job = None
        poll_job = None

        def safe_mtime(path):
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        def thumb_worker():
            while True:
                path, size, gen, key = jobs.get()
                try:
                    with Image.open(path) as source:
                        # JPEG can be decoded at a reduced scale, which is far
                        # cheaper than loading a full-resolution photo.
                        source.draft("RGB", (size * 2, size * 2))
                        image = source.convert("RGB")
                    width, height = image.size
                    side = min(width, height)
                    left = (width - side) // 2
                    top = (height - side) // 2
                    cropped = image.crop(
                        (left, top, left + side, top + side)
                    )
                    image.close()
                    # Center-crop to a square so cells touch like a phone
                    # album instead of showing letterbox bands.
                    cropped = cropped.resize(
                        (size, size), Image.Resampling.BILINEAR
                    )
                    results.put(("thumb", gen, key, cropped))
                except Exception:
                    results.put(("thumb", gen, key, None))

        for _ in range(8):
            threading.Thread(target=thumb_worker, daemon=True).start()

        def clear_cells():
            for cell in items.values():
                for item in cell:
                    canvas.delete(item)
            items.clear()

        def trim_cache(first_index, last_index, size, count):
            keep_lo = first_index - count * 2
            keep_hi = last_index + count * 2
            for key in list(photo_cache):
                index, cached_size = key
                if cached_size != size or not keep_lo <= index < keep_hi:
                    del photo_cache[key]

        def schedule_poll():
            nonlocal poll_job
            if poll_job is None:
                poll_job = canvas.after(20, poll_results)

        def request_thumb(index, size):
            key = (index, size)
            if key in photo_cache or key in pending:
                return
            if len(pending) >= 256 or index >= len(files):
                return
            pending.add(key)
            jobs.put((files[index], size, thumb_generation, key))
            schedule_poll()

        def apply_scan(found):
            nonlocal files, scan_in_flight, thumb_generation
            scan_in_flight = False
            changed = [str(path) for path in found] != [
                str(path) for path in files
            ]
            files = found
            if not changed:
                render()
                return
            # File list changed: drop every cached bitmap and start over.
            thumb_generation += 1
            clear_cells()
            photo_cache.clear()
            pending.clear()
            while True:
                try:
                    jobs.get_nowait()
                except queue.Empty:
                    break
            canvas.yview_moveto(0)
            render(force=True)

        def poll_results():
            nonlocal poll_job
            poll_job = None
            try:
                while True:
                    message = results.get_nowait()
                    if message[0] == "scan":
                        _, gen, found = message
                        if gen == scan_generation:
                            apply_scan(found)
                        continue
                    _, gen, key, image = message
                    pending.discard(key)
                    if gen != thumb_generation:
                        continue
                    index, size = key
                    if image is None:
                        photo_cache[key] = None
                        continue
                    try:
                        photo = ImageTk.PhotoImage(image)
                    finally:
                        image.close()
                    photo_cache[key] = photo
                    cell = items.get(index)
                    if cell is not None and size == layout_size:
                        canvas.itemconfigure(cell[1], image=photo)
            except queue.Empty:
                pass
            if pending or scan_in_flight or not results.empty():
                schedule_poll()

        def render(force=False):
            nonlocal render_job, layout_size, empty_item
            render_job = None
            width = max(canvas.winfo_width(), 240)
            count = max(1, min(12, columns))
            size = max(48, width // count)
            total_rows = (len(files) + count - 1) // count if files else 0
            canvas.configure(
                scrollregion=(0, 0, width, max(1, total_rows * size))
            )

            if not files:
                clear_cells()
                canvas.delete("all")
                if scan_in_flight:
                    message = "読み込み中…"
                elif not folder_var.get().strip():
                    message = "表示フォルダを選択してください"
                else:
                    message = "画像がありません"
                empty_item = canvas.create_text(
                    width // 2, 30, text=message, fill="#aaaaaa"
                )
                layout_size = size
                return

            if empty_item is not None:
                canvas.delete(empty_item)
                empty_item = None

            size_changed = size != layout_size
            if force or size_changed:
                clear_cells()
                for key in list(photo_cache):
                    if key[1] != size:
                        del photo_cache[key]
                layout_size = size
            if size_changed:
                canvas.yview_moveto(0)

            top_row = max(0, int(canvas.canvasy(0) // size))
            view_rows = max(1, int(canvas.winfo_height() // size) + 1)
            first = max(0, top_row - 1)
            last = min(total_rows, top_row + view_rows + 2)
            first_index = first * count
            last_index = min(len(files), last * count)
            desired = set(range(first_index, last_index))

            for index in list(items):
                if index not in desired:
                    for item in items.pop(index):
                        canvas.delete(item)

            for index in range(first_index, last_index):
                if index in items:
                    continue
                row, column = divmod(index, count)
                x, y = column * size, row * size
                rect = canvas.create_rectangle(
                    x, y, x + size, y + size, fill="#202020", outline=""
                )
                image_item = canvas.create_image(x, y, anchor="nw")
                items[index] = (rect, image_item)
                key = (index, size)
                if key in photo_cache:
                    photo = photo_cache[key]
                    if photo is not None:
                        canvas.itemconfigure(image_item, image=photo)
                else:
                    request_thumb(index, size)

            trim_cache(first_index, last_index, size, count)

        def select_image(index):
            selection_status.set(
                f"{index + 1} / {len(files)} 番目: {files[index]}"
            )

        def schedule_render(*_):
            nonlocal render_job
            if render_job is not None:
                canvas.after_cancel(render_job)
            render_job = canvas.after(20, render)

        def set_columns(event=None):
            nonlocal columns
            try:
                columns = max(1, min(12, int(column_var.get())))
            except ValueError:
                columns = 8
            column_var.set(str(columns))
            render(force=True)

        def scan_folder(*_):
            nonlocal files, scan_generation, last_folder, scan_in_flight
            folder_text = folder_var.get().strip()
            folder = Path(folder_text).expanduser() if folder_text else None
            folder_name = str(folder) if folder else ""
            folder_changed = folder_name != last_folder
            last_folder = folder_name
            scan_generation += 1
            gen = scan_generation
            scan_in_flight = bool(folder)
            if folder_changed:
                files = []
                clear_cells()
                photo_cache.clear()
                pending.clear()
                while True:
                    try:
                        jobs.get_nowait()
                    except queue.Empty:
                        break
                canvas.delete("all")
                canvas.yview_moveto(0)
                render(force=True)
            if folder is None:
                render(force=True)
                return

            def job():
                found = []
                try:
                    if folder.is_dir():
                        found = [
                            path
                            for path in folder.iterdir()
                            if path.is_file() and path.suffix.lower() in EXTS
                        ]
                        found.sort(
                            key=lambda path: (
                                safe_mtime(path), str(path).lower()
                            )
                        )
                except Exception:
                    found = []
                results.put(("scan", gen, found))

            threading.Thread(target=job, daemon=True).start()
            schedule_poll()

        def on_click(event):
            if not files or layout_size <= 0:
                return
            count = max(1, min(12, columns))
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            if x < 0 or y < 0:
                return
            row = int(y // layout_size)
            column = int(x // layout_size)
            if column >= count:
                return
            index = row * count + column
            if 0 <= index < len(files):
                select_image(index)

        def on_yscroll(first, last):
            # Keep the scrollbar in sync and repaint when the view moves by any
            # means, including dragging the scrollbar itself.
            scrollbar.set(first, last)
            schedule_render()

        def on_wheel(event):
            canvas.yview_scroll(-3 if event.delta > 0 else 3, "units")
            schedule_render()

        def on_configure(event):
            nonlocal resize_job
            if resize_job is not None:
                canvas.after_cancel(resize_job)
            resize_job = canvas.after(120, lambda: render(force=True))

        def on_folder_change(*_):
            nonlocal scan_job
            if scan_job is not None:
                canvas.after_cancel(scan_job)
            scan_job = canvas.after(400, scan_folder)

        def on_map(event):
            if event.widget is not f:
                return
            if folder_var.get().strip():
                # Re-scan when the tab is opened so files added or removed
                # outside the app are reflected.
                scan_folder()
            else:
                canvas.after(50, lambda: render(force=True))

        column_var = tk.StringVar(value="8")
        controls = ttk.Frame(f)
        controls.grid(row=1, column=0, sticky="e")
        ttk.Button(
            folder_controls,
            text="フォルダを1つ選ぶ…",
            command=lambda: choose_thumbnail_folder(),
        ).grid(row=0, column=2)
        ttk.Label(controls, text="1列の画像数").pack(side="left")
        ttk.Spinbox(
            controls, from_=1, to=12, width=5, textvariable=column_var,
            command=set_columns,
        ).pack(side="left", padx=4)
        controls.winfo_children()[-1].bind("<Return>", set_columns)
        ttk.Button(controls, text="更新", command=scan_folder).pack(side="left")
        folder_entry.bind("<Return>", lambda event: scan_folder())
        folder_var.trace_add("write", on_folder_change)
        f.bind("<Map>", on_map)
        canvas.configure(yscrollcommand=on_yscroll)
        canvas.bind("<Configure>", on_configure)
        canvas.bind("<Button-1>", on_click)
        canvas.bind("<MouseWheel>", on_wheel)
        canvas.bind(
            "<Button-4>",
            lambda event: (
                canvas.yview_scroll(-3, "units"), schedule_render()
            ),
        )
        canvas.bind(
            "<Button-5>",
            lambda event: (
                canvas.yview_scroll(3, "units"), schedule_render()
            ),
        )
        canvas.bind(
            "<Up>",
            lambda event: (
                canvas.yview_scroll(-3, "units"), schedule_render()
            ),
        )
        canvas.bind(
            "<Down>",
            lambda event: (
                canvas.yview_scroll(3, "units"), schedule_render()
            ),
        )
        canvas.focus_set()
        render(force=True)

        def choose_thumbnail_folder():
            folder = filedialog.askdirectory(
                title="サムネイル表示用フォルダを1つ選択"
            )
            if folder:
                folder_var.set(folder)
                scan_folder()

    def pick_pipeline_folder(self):
        p = filedialog.askdirectory(title="解析する画像フォルダ")
        if p:
            self.pipeline_folder.set(p)

    def pick_database(self):
        p = filedialog.asksaveasfilename(
            title="SQLiteデータベースの保存先", defaultextension=".sqlite3",
            filetypes=[("SQLite database", "*.sqlite3"), ("All files", "*.*")]
        )
        if p:
            self.database.set(p)

    @staticmethod
    def pick_file(variable, title, filetypes):
        p = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if p:
            variable.set(p)

    def pipeline_log_line(self, text):
        self.pipeline_log.configure(state="normal")
        self.pipeline_log.insert("end", text + "\n")
        self.pipeline_log.see("end")
        self.pipeline_log.configure(state="disabled")

    def _validate_pipeline_paths(self, need_folder=True):
        folder = Path(self.pipeline_folder.get()).expanduser()
        database = Path(self.database.get()).expanduser()
        if need_folder and not folder.is_dir():
            messagebox.showerror("解析フォルダ", "存在する画像フォルダを指定してください")
            return None
        if not self.database.get().strip():
            messagebox.showerror("SQLiteデータベース", "保存先を指定してください")
            return None
        database.parent.mkdir(parents=True, exist_ok=True)
        return folder, database

    def _set_pipeline_buttons(self, enabled):
        state = "normal" if enabled else "disabled"
        for button in (self.analyze_btn, self.caption_btn, self.nsfw_score_btn, self.cluster_btn,
                       self.keyword_search_btn, self.semantic_search_btn):
            button.configure(state=state)
        self.pipeline_gpu_toggle.configure(state=state)
        self.pipeline_stop_btn.configure(state="disabled" if enabled else "normal")

    def _start_pipeline_worker(self, operation, *args):
        if self.pipeline_running:
            return
        self.pipeline_running = True
        self.pipeline_cancel.clear()
        self._set_pipeline_buttons(False)
        self.pipeline_progress.configure(value=0)
        threading.Thread(target=self._run_pipeline_worker, args=(operation, args), daemon=True).start()

    def stop_pipeline(self):
        if self.pipeline_running:
            self.pipeline_cancel.set()
            self.pipeline_status.set("解析を中断しています。現在の画像の処理後に停止します…")
            self.pipeline_log_line("[STOP] 中断要求を受け付けました")

    def start_pipeline_analysis(self):
        paths = self._validate_pipeline_paths()
        if not paths:
            return
        self._start_pipeline_worker("analyze", paths[0], paths[1])

    def start_caption(self):
        paths = self._validate_pipeline_paths()
        if paths:
            self._start_pipeline_worker("caption", paths[0], paths[1])

    def start_nsfw_scoring(self):
        paths = self._validate_pipeline_paths()
        if paths:
            self._start_pipeline_worker("nsfw_score", paths[0], paths[1])

    def start_clustering(self):
        paths = self._validate_pipeline_paths(need_folder=False)
        if not paths:
            return
        try:
            clusters = int(self.cluster_count.get())
            if clusters < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("クラスタ数", "1以上の整数を入力してください")
            return
        self._start_pipeline_worker("cluster", paths[1], clusters)

    def start_search(self, semantic):
        paths = self._validate_pipeline_paths(need_folder=False)
        query = self.search_query.get().strip()
        if not paths:
            return
        if not query:
            messagebox.showerror("検索語", "検索語を入力してください")
            return
        self.last_search_translation = None
        self._start_pipeline_worker("search", paths[1], query, semantic)

    def sequence_log_line(self, text):
        self.sequence_log.configure(state="normal")
        self.sequence_log.insert("end", text + "\n")
        self.sequence_log.see("end")
        self.sequence_log.configure(state="disabled")

    def refresh_sequence_list(self):
        paths = self._validate_pipeline_paths(need_folder=False)
        if not paths:
            return
        try:
            from image_pipeline import list_sequences
            groups = list_sequences(paths[1])
            for item in self.sequence_list.get_children():
                self.sequence_list.delete(item)
            for group in groups:
                self.sequence_list.insert(
                    "", "end",
                    values=(
                        group["sequence_group"],
                        len(group["paths"]),
                        " | ".join(group["paths"]),
                    ),
                )
            self.sequence_status.set(f"判定済みグループ: {len(groups)}")
        except Exception as exc:
            self.sequence_log_line(f"[ERROR] 一覧更新: {exc}")
            messagebox.showerror("一覧更新エラー", str(exc))

    def show_sequence_previews(self):
        paths = self._validate_pipeline_paths(need_folder=False)
        if not paths:
            return
        try:
            from image_pipeline import list_sequences
            from PIL import Image, ImageTk
            groups = list_sequences(paths[1])
        except Exception as exc:
            self.sequence_log_line(f"[ERROR] プレビュー: {exc}")
            messagebox.showerror("プレビューエラー", str(exc))
            return
        if not groups:
            messagebox.showinfo("グループ確認", "判定済みグループがありません")
            return

        window = tk.Toplevel(self.root)
        window.title(f"連続画像グループ確認 (全{len(groups)}グループ)")
        window.geometry("1100x700")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        position = tk.IntVar(value=0)
        status = tk.StringVar()
        content = ttk.Frame(window, padding=8)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        cache = {}
        card = ttk.Frame(content)
        card.grid(row=0, column=0, sticky="nsew")
        order_list = tk.Listbox(content, height=4, exportselection=False)
        order_list.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        def make_preview(index):
            if index in cache:
                return cache[index]
            group = groups[index]
            source_images = []
            try:
                for path in group["paths"]:
                    with Image.open(path) as image:
                        source_images.append(image.convert("RGB"))
                preview_height = min(520, max(image.height for image in source_images))
                resized = []
                for image in source_images:
                    width = max(1, round(image.width * preview_height / image.height))
                    resized.append(image.resize((width, preview_height)))
                    image.close()
                combined = Image.new(
                    "RGB", (sum(image.width for image in resized), preview_height)
                )
                offset = 0
                for image in resized:
                    combined.paste(image, (offset, 0))
                    offset += image.width
                    image.close()
                combined.thumbnail((1050, 520))
                photo = ImageTk.PhotoImage(combined)
                combined.close()
                cache[index] = (photo, "\n".join(group["paths"]))
                return cache[index]
            except Exception as exc:
                cache[index] = (None, f"画像を表示できません: {exc}")
                return cache[index]

        def release_outside(index):
            for cached_index in list(cache):
                if abs(cached_index - index) > 1:
                    del cache[cached_index]

        def render(index):
            index = max(0, min(index, len(groups) - 1))
            position.set(index)
            release_outside(index)
            for neighbor in (index - 1, index, index + 1):
                if 0 <= neighbor < len(groups):
                    make_preview(neighbor)
            for child in card.winfo_children():
                child.destroy()
            photo, paths_text = cache[index]
            header = ttk.Label(
                card,
                text=f"グループ {groups[index]['sequence_group']} "
                     f"({len(groups[index]['paths'])}枚)",
                font=("", 12, "bold"),
            )
            header.pack(anchor="w", pady=(0, 6))
            if photo:
                ttk.Label(card, image=photo).pack(anchor="center")
            else:
                # Paths are listed in the reorder box below, so only show them
                # here when the preview itself cannot be rendered.
                ttk.Label(card, text=paths_text, wraplength=1050).pack(anchor="w")
            order_list.delete(0, "end")
            for path in groups[index]["paths"]:
                order_list.insert("end", Path(path).name)
            status.set(f"{index + 1} / {len(groups)}")

        def move_order(delta):
            selected_index = order_list.curselection()
            if not selected_index:
                return
            old_index = selected_index[0]
            new_index = old_index + delta
            if not 0 <= new_index < order_list.size():
                return
            group = groups[position.get()]
            group["paths"][old_index], group["paths"][new_index] = (
                group["paths"][new_index],
                group["paths"][old_index],
            )
            cache.pop(position.get(), None)
            render(position.get())
            order_list.selection_set(new_index)

        def save_order():
            index = position.get()
            group = groups[index]
            try:
                from image_pipeline import reorder_sequence_group
                reorder_sequence_group(
                    paths[1], group["sequence_group"], group["paths"]
                )
            except Exception as exc:
                self.sequence_log_line(f"[ERROR] 並び順保存: {exc}")
                messagebox.showerror("並び順保存エラー", str(exc), parent=window)
                return
            cache.clear()
            self.refresh_sequence_list()
            self.sequence_log_line(
                f"[REORDER] グループ {group['sequence_group']} の並び順を保存しました"
            )
            render(index)
            messagebox.showinfo("並び順保存", "並び順を保存しました", parent=window)

        def clear_current_group():
            index = position.get()
            group = groups[index]
            if not messagebox.askyesno(
                "グループ解除",
                f"グループ {group['sequence_group']} の登録を解除しますか？\n"
                "画像ファイルと画像情報は削除されません。",
                parent=window,
            ):
                return
            try:
                from image_pipeline import clear_sequence_group
                removed = clear_sequence_group(paths[1], group["sequence_group"])
            except Exception as exc:
                self.sequence_log_line(f"[ERROR] グループ解除: {exc}")
                messagebox.showerror("グループ解除エラー", str(exc), parent=window)
                return
            self.sequence_log_line(
                f"[UNGROUP] グループ {group['sequence_group']} "
                f"({removed}枚) の登録を解除しました"
            )
            groups.pop(index)
            cache.clear()
            self.refresh_sequence_list()
            if not groups:
                window.destroy()
                return
            render(min(index, len(groups) - 1))

        controls = ttk.Frame(window, padding=(8, 4))
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Button(
            controls, text="◀ 前のグループ",
            command=lambda: render(position.get() - 1),
        ).grid(row=0, column=0, padx=4)
        ttk.Label(controls, textvariable=status, anchor="center").grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(
            controls, text="次のグループ ▶",
            command=lambda: render(position.get() + 1),
        ).grid(row=0, column=2, padx=4)
        ttk.Button(
            controls, text="このグループを解除",
            command=clear_current_group,
        ).grid(row=0, column=3, padx=(12, 4))
        ttk.Button(
            controls, text="選択画像を上へ",
            command=lambda: move_order(-1),
        ).grid(row=1, column=0, padx=4, pady=(6, 0))
        ttk.Button(
            controls, text="選択画像を下へ",
            command=lambda: move_order(1),
        ).grid(row=1, column=1, padx=4, pady=(6, 0))
        ttk.Button(
            controls, text="並び順を保存",
            command=save_order,
        ).grid(row=1, column=2, padx=4, pady=(6, 0))
        window.bind("<Left>", lambda event: render(position.get() - 1))
        window.bind("<Right>", lambda event: render(position.get() + 1))
        window.bind("<MouseWheel>", lambda event: render(
            position.get() - 1 if event.delta > 0 else position.get() + 1
        ))
        window.bind("<Button-4>", lambda event: render(position.get() - 1))
        window.bind("<Button-5>", lambda event: render(position.get() + 1))
        window.focus_set()
        render(0)

    def add_manual_sequence(self):
        paths = self._validate_pipeline_paths(need_folder=True)
        if not paths:
            return
        try:
            from PIL import Image, ImageTk
        except ImportError as exc:
            messagebox.showerror("手動グループ追加", str(exc))
            return

        window = tk.Toplevel(self.root)
        window.title("手動で連続画像グループを追加")
        window.geometry("1200x650")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        folder = paths[0]
        files = sorted(
            (path for path in folder.rglob("*") if path.suffix.lower() in EXTS),
            key=lambda path: (path.stat().st_mtime, str(path).lower()),
        )
        if not files:
            window.destroy()
            messagebox.showinfo("手動グループ追加", "選択中フォルダに画像がありません")
            return
        selected = set()
        start = 0
        slots = []
        preview_cache = {}
        preload_job = None
        status = tk.StringVar(
            value="フォルダ内の画像を4枚ずつ表示します。チェック後に前後へ移動できます"
        )
        header = ttk.Frame(window, padding=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, textvariable=status).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(header, text="先頭に表示する画像番号").grid(
            row=0, column=1, padx=(12, 4)
        )
        image_number = tk.StringVar(value="1")
        image_number_input = ttk.Spinbox(
            header,
            from_=1,
            to=len(files),
            textvariable=image_number,
            width=8,
        )
        image_number_input.grid(row=0, column=2)
        cards = ttk.Frame(window, padding=8)
        cards.grid(row=1, column=0, sticky="nsew")
        for index in range(4):
            cards.columnconfigure(index, weight=1)
            card = ttk.LabelFrame(cards, text=f"{index + 1}枚目", padding=6)
            card.grid(row=0, column=index, sticky="nsew", padx=4)
            preview = ttk.Label(card, text="未選択", anchor="center")
            preview.pack(fill="both", expand=True, pady=(0, 6))
            path_var = tk.StringVar()
            registered_var = tk.StringVar(value="未選択")
            selected_var = tk.BooleanVar(value=False)
            ttk.Label(card, textvariable=registered_var, wraplength=250).pack(
                anchor="w"
            )
            ttk.Checkbutton(
                card, text="この画像を登録", variable=selected_var
            ).pack(anchor="w")
            slots.append((path_var, registered_var, selected_var, preview))

        def open_thumbnail_browser():
            browser = tk.Toplevel(window)
            browser.title("フォルダ画像サムネイル一覧")
            browser.geometry("1100x760")
            browser.columnconfigure(0, weight=1)
            browser.rowconfigure(1, weight=1)
            columns = tk.IntVar(value=8)
            page_start = 0
            thumbnail_cache = {}
            selected_number = tk.StringVar(value="画像をクリックすると通し番号を表示します")
            grid = ttk.Frame(browser, padding=8)
            grid.grid(row=1, column=0, sticky="nsew")

            def load_thumbnail(index):
                if index in thumbnail_cache:
                    return thumbnail_cache[index]
                try:
                    with Image.open(files[index]) as image:
                        image = image.convert("RGB")
                        image.thumbnail((120, 120))
                        square = Image.new("RGB", (120, 120), "white")
                        square.paste(
                            image,
                            ((120 - image.width) // 2, (120 - image.height) // 2),
                        )
                        photo = ImageTk.PhotoImage(square)
                        square.close()
                    thumbnail_cache[index] = (photo, "")
                except Exception as exc:
                    thumbnail_cache[index] = (None, f"表示できません: {exc}")
                return thumbnail_cache[index]

            def render_thumbnails():
                nonlocal page_start
                count = max(1, min(12, int(columns.get())))
                page_size = count * 3
                page_start = max(
                    0, min(page_start, max(0, len(files) - page_size))
                )
                cache_start = max(0, page_start - page_size * 3)
                cache_end = min(len(files), page_start + page_size * 4)
                keep = set(range(cache_start, cache_end))
                for index in list(thumbnail_cache):
                    if index not in keep:
                        del thumbnail_cache[index]
                for child in grid.winfo_children():
                    child.destroy()
                for column in range(count):
                    grid.columnconfigure(column, weight=1)
                for offset, index in enumerate(
                    range(page_start, min(page_start + page_size, len(files)))
                ):
                    row, column = divmod(offset, count)
                    photo, error = load_thumbnail(index)
                    label = ttk.Label(
                        grid,
                        image=photo,
                        text=error,
                        compound="top",
                        anchor="center",
                    )
                    label.grid(row=row, column=column, padx=3, pady=3)
                    label.bind(
                        "<Button-1>",
                        lambda event, number=index + 1: selected_number.set(
                            f"{number} / {len(files)} 番目"
                        ),
                    )
                selected_number.set(
                    f"{page_start + 1}-{min(page_start + page_size, len(files))}"
                    f" / {len(files)} | 前後3ページを先読み"
                )

            def move_page(delta):
                nonlocal page_start
                page_size = max(1, min(12, int(columns.get()))) * 3
                page_start = max(
                    0,
                    min(page_start + delta * page_size, max(0, len(files) - page_size)),
                )
                render_thumbnails()

            header = ttk.Frame(browser, padding=8)
            header.grid(row=0, column=0, sticky="ew")
            ttk.Label(header, text="1列の画像数").pack(side="left")
            columns_box = ttk.Spinbox(
                header,
                from_=1,
                to=12,
                width=5,
                textvariable=columns,
                command=render_thumbnails,
            )
            columns_box.pack(side="left", padx=5)
            columns_box.bind("<Return>", lambda event: render_thumbnails())
            ttk.Button(
                header, text="更新", command=render_thumbnails
            ).pack(side="left")
            ttk.Button(
                header, text="◀ 前の範囲",
                command=lambda: move_page(-1),
            ).pack(side="left", padx=(18, 3))
            ttk.Button(
                header, text="次の範囲 ▶",
                command=lambda: move_page(1),
            ).pack(side="left", padx=3)
            ttk.Label(
                browser, textvariable=selected_number, anchor="center"
            ).grid(row=2, column=0, sticky="ew", pady=(0, 8))
            browser.bind("<MouseWheel>", lambda event: move_page(
                -1 if event.delta > 0 else 1
            ))
            browser.bind("<Button-4>", lambda event: move_page(-1))
            browser.bind("<Button-5>", lambda event: move_page(1))
            render_thumbnails()

        def remember_selection():
            for path_var, _, selected_var, _ in slots:
                path = path_var.get()
                if path:
                    if selected_var.get():
                        selected.add(path)
                    else:
                        selected.discard(path)

        def load_preview(path):
            cached = preview_cache.get(str(path))
            if cached is not None:
                return cached
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((260, 440))
                    photo = ImageTk.PhotoImage(image)
                preview_cache[str(path)] = (photo, "")
            except Exception as exc:
                preview_cache[str(path)] = (None, f"表示できません: {exc}")
            return preview_cache[str(path)]

        def preload_neighbors():
            nonlocal preload_job
            preload_job = None
            cache_start = max(0, start - 4)
            cache_end = min(len(files), start + 8)
            for path in files[cache_start:cache_end]:
                load_preview(path)
            for path in list(preview_cache):
                if path not in {str(item) for item in files[cache_start:cache_end]}:
                    del preview_cache[path]

        def schedule_preload():
            nonlocal preload_job
            if preload_job is not None:
                window.after_cancel(preload_job)
            preload_job = window.after_idle(preload_neighbors)

        def render():
            remember_selection()
            for index, (path_var, registered_var, selected_var, preview) in enumerate(slots):
                file_index = start + index
                if file_index >= len(files):
                    path_var.set("")
                    registered_var.set("表示なし")
                    selected_var.set(False)
                    preview.configure(image="", text="表示なし")
                    continue
                path = files[file_index]
                path_var.set(str(path))
                selected_var.set(str(path) in selected)
                photo, error = load_preview(path)
                preview.configure(image=photo or "", text=error)
                try:
                    from image_pipeline import image_sequence_status
                    group = image_sequence_status(paths[1], path)
                    registered_var.set(
                        f"登録済み: グループ {group}" if group is not None else "未登録"
                    )
                except Exception as exc:
                    registered_var.set(f"登録状態を確認できません: {exc}")
            end = min(start + 4, len(files))
            status.set(
                f"{start + 1}-{end} / {len(files)} | "
                f"選択中: {len(selected)}枚 | 周辺画像を先読み中"
            )
            schedule_preload()

        def move(delta):
            nonlocal start
            remember_selection()
            start = max(0, min(start + delta, max(0, len(files) - 4)))
            image_number.set(str(start + 1))
            render()

        def jump_to_image():
            nonlocal start
            try:
                requested = int(image_number.get())
            except ValueError:
                messagebox.showerror(
                    "表示位置",
                    f"1から{len(files)}までの整数を入力してください",
                    parent=window,
                )
                image_number.set(str(start + 1))
                return
            if not 1 <= requested <= len(files):
                messagebox.showerror(
                    "表示位置",
                    f"1から{len(files)}までの整数を入力してください",
                    parent=window,
                )
                image_number.set(str(start + 1))
                return
            remember_selection()
            start = min(requested - 1, max(0, len(files) - 4))
            image_number.set(str(start + 1))
            render()

        def register():
            remember_selection()
            ordered = [str(path) for path in files if str(path) in selected]
            if len(ordered) < 2:
                messagebox.showerror("手動グループ追加", "2枚以上をチェックしてください", parent=window)
                return
            if len(ordered) > 4:
                messagebox.showerror(
                    "手動グループ追加",
                    "一度に登録できるのは4枚までです。チェックを4枚以内にしてください",
                    parent=window,
                )
                return
            try:
                from image_pipeline import add_sequence_group
                group = add_sequence_group(paths[1], ordered)
            except Exception as exc:
                self.sequence_log_line(f"[ERROR] 手動グループ追加: {exc}")
                messagebox.showerror("手動グループ追加エラー", str(exc), parent=window)
                return
            self.sequence_log_line(
                f"[GROUP] 手動登録: グループ {group} | "
                + " || ".join(Path(path).name for path in ordered)
            )
            self.refresh_sequence_list()
            status.set(f"グループ {group} として登録しました")
            selected.clear()
            # render() re-reads the checkbox states, so clear them first or the
            # previous selection would be restored immediately.
            for _, _, selected_var, _ in slots:
                selected_var.set(False)
            render()
            messagebox.showinfo(
                "手動グループ追加",
                f"グループ {group} の登録が完了しました。\n\n"
                "並び順は自動では変更されません。\n"
                "「グループを確認」画面で並べ替えを行ってください。",
                parent=window,
            )

        controls = ttk.Frame(window, padding=6)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Button(controls, text="◀ 前の4枚", command=lambda: move(-1)).grid(
            row=0, column=0, padx=4
        )
        ttk.Label(controls, text="1枚ずつスライド", anchor="center").grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(controls, text="次の4枚 ▶", command=lambda: move(1)).grid(
            row=0, column=2, padx=4
        )
        ttk.Button(
            window, text="チェックした画像をグループ登録",
            command=register,
        ).grid(row=3, column=0, pady=10)
        window.bind("<Left>", lambda event: move(-1))
        window.bind("<Right>", lambda event: move(1))
        window.bind("<MouseWheel>", lambda event: move(
            -1 if event.delta > 0 else 1
        ))
        window.bind("<Button-4>", lambda event: move(-1))
        window.bind("<Button-5>", lambda event: move(1))
        image_number_input.bind("<Return>", lambda event: jump_to_image())
        ttk.Button(
            header, text="表示", command=jump_to_image
        ).grid(row=0, column=3, padx=(4, 0))
        window.focus_set()
        render()

    def start_sequence_check(self):
        paths = self._validate_pipeline_paths(need_folder=False)
        if not paths:
            return
        try:
            threshold = float(self.sequence_threshold.get())
            if not 0 <= threshold <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("境界連続スコア", "0から1の数値を入力してください")
            return
        if self.sequence_running or self.pipeline_running:
            return
        self.sequence_running = True
        self.sequence_cancel.clear()
        self.sequence_start_btn.configure(state="disabled")
        self.sequence_stop_btn.configure(state="normal")
        self.sequence_gpu_toggle.configure(state="disabled")
        self.sequence_progress.configure(value=0)
        threading.Thread(
            target=self._run_sequence_worker,
            args=(paths[1], threshold),
            daemon=True,
        ).start()

    def stop_sequence_check(self):
        if self.sequence_running:
            self.sequence_cancel.set()
            self.sequence_status.set("中断しています。現在の候補の処理後に停止します…")
            self.sequence_log_line("[STOP] 中断要求を受け付けました")

    def _run_sequence_worker(self, database, threshold):
        try:
            from image_pipeline import detect_sequences
            result = detect_sequences(
                database, use_gpu=self.pipeline_use_gpu.get(),
                threshold=threshold, cancel_event=self.sequence_cancel,
                progress=lambda i, total, groups, errors, path, error:
                self.events.put(("sequence_progress", i, total, groups, errors, path, error)),
            )
            label = "中断" if self.sequence_cancel.is_set() else "連続画像判定完了"
            self.events.put(("sequence_done", f"{label}: {result[0]}グループ, エラー {result[1]}"))
        except Exception as exc:
            self.events.put(("sequence_error", f"{exc}\n\n{traceback.format_exc()}"))

    def _run_pipeline_worker(self, operation, args):
        try:
            from image_pipeline import (
                analyze_folder, assign_clusters, caption_folder, score_nsfw_folder, search
            )
            print(f"[APP] Pipeline operation started: {operation}", flush=True)
            if operation == "analyze":
                folder, database = args
                result = analyze_folder(
                    folder, database, clip_model=self.clip_model.get(),
                    use_gpu=self.pipeline_use_gpu.get(),
                    cancel_event=self.pipeline_cancel,
                    progress=self._pipeline_progress_event,
                )
                label = "中断" if self.pipeline_cancel.is_set() else "解析完了"
                self.events.put(("pipeline_done", f"{label}: 保存 {result[0]}, エラー {result[1]}"))
            elif operation == "caption":
                folder, database = args
                result = caption_folder(
                    folder, database, model_name=self.florence_model.get(),
                    use_gpu=self.pipeline_use_gpu.get(),
                    cancel_event=self.pipeline_cancel, progress=self._pipeline_progress_event,
                )
                label = "中断" if self.pipeline_cancel.is_set() else "キャプション完了"
                self.events.put(("pipeline_done", f"{label}: 保存 {result[0]}, エラー {result[1]}"))
            elif operation == "nsfw_score":
                folder, database = args
                result = score_nsfw_folder(
                    folder, database, use_gpu=self.pipeline_use_gpu.get(),
                    cancel_event=self.pipeline_cancel,
                    progress=self._pipeline_progress_event,
                )
                label = "中断" if self.pipeline_cancel.is_set() else "NSFWスコア解析完了"
                self.events.put(("pipeline_done", f"{label}: 更新 {result[0]}, エラー {result[1]}"))
            elif operation == "cluster":
                database, clusters = args
                count = assign_clusters(database, clusters=clusters)
                self.events.put(("pipeline_done", f"{count}クラスタに分類しました"))
            else:
                database, query, semantic = args
                rows = search(database, query, limit=20,
                              clip_model=self.clip_model.get() if semantic else None,
                              use_gpu=self.pipeline_use_gpu.get(),
                              translation_callback=lambda original, translated: self.events.put(
                                  ("search_translation", original, translated)
                              ))
                self.events.put(("pipeline_result", rows))
        except Exception as exc:
            self.events.put(("pipeline_error", f"{exc}\n\n{traceback.format_exc()}"))

    def _pipeline_progress_event(self, index, total, done, errors, path="", error=""):
        self.events.put(("pipeline_progress", index, total, done, errors, path, error))

    def show_search_results(self, rows):
        window = tk.Toplevel(self.root)
        window.title(f"検索結果 ({len(rows)}件)")
        window.geometry("760x680")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        if not rows:
            ttk.Label(window, text="検索結果がありません").grid(
                row=0, column=0, padx=20, pady=20
            )
            return
        image_label = ttk.Label(window, anchor="center")
        image_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        info = ttk.Label(window, anchor="center", justify="center", wraplength=720)
        info.grid(row=1, column=0, sticky="ew", padx=12)
        controls = ttk.Frame(window)
        controls.grid(row=2, column=0, pady=10)
        previous = ttk.Button(controls, text="← 前へ")
        previous.pack(side="left", padx=5)
        counter = ttk.Label(controls)
        counter.pack(side="left", padx=12)
        following = ttk.Button(controls, text="次へ →")
        following.pack(side="left", padx=5)
        photos = []

        def display(index):
            index %= len(rows)
            row = rows[index]
            try:
                from PIL import Image, ImageTk
                source_images = [
                    Image.open(path).convert("RGB")
                    for path in row.get("paths", [row["path"]])
                ]
                height = min(500, max(image.height for image in source_images))
                resized = []
                for image in source_images:
                    width = max(1, round(image.width * height / image.height))
                    resized.append(image.resize((width, height)))
                image = Image.new("RGB", (
                    sum(item.width for item in resized), height
                ))
                offset = 0
                for item in resized:
                    image.paste(item, (offset, 0))
                    offset += item.width
                image.thumbnail((700, 500))
                photo = ImageTk.PhotoImage(image)
                photos[:] = [photo]
                image_label.configure(image=photo, text="")
            except Exception:
                photos.clear()
                image_label.configure(image="", text="画像を表示できません")
            score = row.get("confidence") or 0.0
            info.configure(
                text=f"{score:.3f}\n"
                f"{len(row.get('paths', [row['path']]))}枚\n{row['path']}"
            )
            counter.configure(text=f"{index + 1} / {len(rows)}")
            window.current_result_index = index

        previous.configure(command=lambda: display(window.current_result_index - 1))
        following.configure(command=lambda: display(window.current_result_index + 1))
        window.bind("<MouseWheel>", lambda event: display(
            window.current_result_index - (1 if event.delta > 0 else -1)
        ))
        window.bind("<Button-4>", lambda event: display(window.current_result_index - 1))
        window.bind("<Button-5>", lambda event: display(window.current_result_index + 1))
        window.current_result_index = 0
        display(0)

    def pick_source(self):
        p = filedialog.askdirectory(title="変更元フォルダを指定してね")
        if p: self.source.set(p)

    def pick_output(self):
        p = filedialog.askdirectory(title="処理結果フォルダを指定してね")
        if p: self.output.set(p)

    @staticmethod
    def _gpu_available():
        try:
            import onnxruntime
            return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        except Exception:
            return False

    def log_line(self, s):
        self.log.configure(state="normal")
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    @staticmethod
    def is_junk_name(path):
        name = path.name
        if name.lower().startswith(TRASH_PREFIX):
            return True
        return bool(DUPLICATE_SUFFIX.search(path.stem))

    def cleanup(self):
        source = Path(self.source.get()).expanduser()
        if not source.is_dir():
            messagebox.showerror("変更元フォルダ", "存在する変更元フォルダを指定してね")
            return
        candidates = []
        try:
            output = Path(self.output.get()).expanduser() if self.output.get().strip() else None
            out_resolved = output.resolve() if output else None
            for p in source.rglob("*"):
                if not p.is_file() or not self.is_junk_name(p):
                    continue
                try:
                    if out_resolved and (p.resolve() == out_resolved or out_resolved in p.resolve().parents):
                        continue
                except Exception:
                    pass
                candidates.append(p)
        except Exception as e:
            messagebox.showerror("前処理エラー", str(e))
            return

        if not candidates:
            messagebox.showinfo("前処理完了", '該当ファイルはなかったよ ".trashed-*" or "*(number)" ')
            return

        preview = "\n".join(str(p.relative_to(source)) for p in candidates[:20])
        if len(candidates) > 20:
            preview += f"\n… and {len(candidates) - 20} more files"
        ok = messagebox.askyesno(
            "削除の確認だよ",
            f" {len(candidates)} 個のファイルが見つかりました\n{source}\n\n"
            f"例:\n{preview}\n\n"
            "これらのファイルは完全に削除されます（Windowsのゴミ箱には移動しません）\n"
            "続行しますか？"
        )
        if not ok:
            return

        deleted = errors = 0
        for p in candidates:
            try:
                p.unlink()
                deleted += 1
                self.log_line(f"[DELETED] {p}")
            except Exception as e:
                errors += 1
                self.log_line(f"[ERROR] {p}: {e}")
        summary = f"前処理完了: 検知 {deleted}, エラー {errors}."
        self.status.set(summary)
        self.log_line(summary)
        messagebox.showinfo("前処理完了", summary)

    def start(self):
        src, out = Path(self.source.get()).expanduser(), Path(self.output.get()).expanduser()
        try:
            threshold = float(self.threshold.get())
            if not 0 <= threshold <= 1: raise ValueError
        except ValueError:
            messagebox.showerror("不正な閾値", "0から1の間の数値を入力してね")
            return
        if not src.is_dir():
            messagebox.showerror("変更元フォルダ", "存在する変更元フォルダを指定してね")
            return
        if not self.output.get().strip():
            messagebox.showerror("処理結果フォルダ", "処理結果フォルダを指定してね")
            return
        try:
            if src.resolve() == out.resolve() or src.resolve() in out.resolve().parents:
                messagebox.showerror("フォルダ選択", "出力フォルダは変更元フォルダの中に指定できません")
                return
        except Exception: pass
        if not messagebox.askyesno("受付完了", "処理を開始しますか？\n\n初回実行時はモデルのダウンロードが行われます。"):
            return
        self.stop_requested = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.cleanup_btn.configure(state="disabled")
        self.progress.set(0)
        use_gpu = self.use_gpu.get() and self._gpu_available()
        self.status.set(f"ローカルの分類機を読み込み中（{'GPU' if use_gpu else 'CPU'} で実行）")
        threading.Thread(target=self.run_sort, args=(src, out, threshold, self.mode.get(), use_gpu), daemon=True).start()

    def stop(self):
        self.stop_requested = True
        self.status.set("中断されました。しばらく待ってね…")

    def run_sort(self, src, out, threshold, mode, use_gpu):
        try:
            detector, execution_mode = build_detector(use_gpu=use_gpu)
            self.events.put(("status", f"{execution_mode} モードで分類機を起動中…"))
            if execution_mode != "GPU":
                self.events.put(("log", "GPUを使用できないためCPUモードで処理します。NVIDIAドライバー/CUDA DLLを確認してください。"))
            files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
            try:
                out_resolved = out.resolve()
                files = [p for p in files if out_resolved not in p.resolve().parents and p.resolve() != out_resolved]
            except Exception: pass
            files = [p for p in files if not self.is_junk_name(p)]
            total = len(files)
            if not total:
                self.events.put(("done", "サポートされている画像ファイルが見つかりませんでした。", 0, 0, 0))
                return
            adult = other = errors = 0
            self.events.put(("status", f"画像 {total} 枚を見つけたよ。処理中…"))
            adult_labels = {
                "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
                "MALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED", "ANUS_EXPOSED",
                "MALE_BREAST_EXPOSED"
            }
            cpu_fallback_detector = None
            for i, src_file in enumerate(files, 1):
                if self.stop_requested: break
                try:
                    try:
                        detections = detector.detect(str(src_file))
                    except Exception as detection_error:
                        message = str(detection_error)
                        if (
                            execution_mode == "GPU"
                            and ("CUDNN" in message.upper() or "CUDA" in message.upper())
                        ):
                            self.events.put((
                                "log",
                                "[GPU] cuDNN/CUDA inference failed; switching NudeNet to CPU "
                                f"for the remaining images: {message}",
                            ))
                            if cpu_fallback_detector is None:
                                cpu_fallback_detector, _ = build_detector(use_gpu=False)
                            detector = cpu_fallback_detector
                            execution_mode = "CPU (GPU推論失敗)"
                            detections = detector.detect(str(src_file))
                        else:
                            raise
                    unsafe = max(
                        (float(item.get("score", 0.0)) for item in detections
                         if str(item.get("class", item.get("label", ""))).upper() in adult_labels),
                        default=0.0
                    )
                    category = "18禁" if unsafe >= threshold else "その他"
                    folder = out / category
                    folder.mkdir(parents=True, exist_ok=True)
                    dest = folder / src_file.name
                    n = 1
                    while dest.exists():
                        dest = folder / f"{src_file.stem}_{n}{src_file.suffix}"
                        n += 1
                    if mode == "move": shutil.move(str(src_file), str(dest))
                    else: shutil.copy2(str(src_file), str(dest))
                    if category == "18禁": adult += 1
                    else: other += 1
                    self.events.put(("log", f"[{i}/{total}] {category} (score={unsafe:.3f}) {src_file.name}"))
                except Exception as e:
                    errors += 1
                    self.events.put(("log", f"[ERROR] {src_file.name}: {e}"))
                self.events.put(("progress", i, total, adult, other, errors))
            state = "中断" if self.stop_requested else "完了"
            self.events.put(("完了", f"{state}: 18禁 {adult}, その他 {other}, エラー {errors}。", adult, other, errors))
        except Exception as e:
            self.events.put(("失敗", f"{e}\n\n{traceback.format_exc()}"))

    def poll(self):
        try:
            while True:
                e = self.events.get_nowait()
                if e[0] == "log": self.log_line(e[1])
                elif e[0] == "status": self.status.set(e[1])
                elif e[0] == "progress":
                    _, i, total, a, o, err = e
                    self.progress.set(i / total * 100)
                    self.status.set(f"処理中 {i}/{total} | 18禁: {a} | その他: {o} | エラー: {err}")
                elif e[0] in ("done", "完了"):
                    self.status.set(e[1]); self.log_line(e[1])
                    self.start_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
                    self.cleanup_btn.configure(state="normal")
                    messagebox.showinfo("処理完了", e[1])
                elif e[0] in ("fatal", "失敗"):
                    self.status.set("エラー"); self.log_line(e[1])
                    self.start_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
                    self.cleanup_btn.configure(state="normal")
                    messagebox.showerror("処理エラー", e[1])
                elif e[0] == "pipeline_progress":
                    _, index, total, done, errors, path, error = e
                    self.pipeline_progress.configure(value=index / total * 100 if total else 0)
                    self.pipeline_status.set(
                        f"解析中 {index}/{total} | 保存: {done} | エラー: {errors}"
                    )
                    if path and error:
                        self.pipeline_log_line(f"[ERROR] {path}: {error}")
                elif e[0] == "pipeline_done":
                    self.pipeline_running = False
                    self._set_pipeline_buttons(True)
                    self.pipeline_status.set(e[1])
                    self.pipeline_log_line(e[1])
                    self.pipeline_progress.configure(value=100)
                elif e[0] == "search_translation":
                    _, original, translated = e
                    self.last_search_translation = (original, translated)
                    self.pipeline_log_line(
                        f"検索文（原文）: {original}\n検索文（英訳）: {translated}"
                    )
                elif e[0] == "pipeline_result":
                    self.pipeline_running = False
                    self._set_pipeline_buttons(True)
                    if self.last_search_translation:
                        original, translated = self.last_search_translation
                        self.pipeline_status.set(
                            f"検索完了 | 原文: {original} | 英訳: {translated}"
                        )
                    else:
                        self.pipeline_status.set("検索完了")
                    rows = e[1]
                    self.pipeline_log_line(f"検索結果: {len(rows)}件")
                    for row in rows[:20]:
                        score = row.get("confidence") or 0.0
                        self.pipeline_log_line(f"[{score:.3f}] {row['path']}")
                    self.show_search_results(rows)
                elif e[0] == "pipeline_error":
                    self.pipeline_running = False
                    self._set_pipeline_buttons(True)
                    self.pipeline_status.set("解析エラー")
                    self.pipeline_log_line(e[1])
                    messagebox.showerror("解析エラー", e[1])
                elif e[0] == "sequence_progress":
                    _, index, total, groups, errors, path, error = e
                    self.sequence_progress.configure(
                        value=index / total * 100 if total else 0
                    )
                    self.sequence_status.set(
                        f"判定中 {index}/{total} | グループ: {groups} | エラー: {errors}"
                    )
                    if path and error:
                        self.sequence_log_line(f"[ERROR] {path}: {error}")
                    elif path:
                        self.sequence_log_line(f"[GROUP] {path}")
                elif e[0] == "sequence_done":
                    self.sequence_running = False
                    self.sequence_start_btn.configure(state="normal")
                    self.sequence_stop_btn.configure(state="disabled")
                    self.sequence_gpu_toggle.configure(state="normal")
                    self.sequence_progress.configure(value=100)
                    self.sequence_status.set(e[1])
                    self.sequence_log_line(e[1])
                    self.refresh_sequence_list()
                elif e[0] == "sequence_error":
                    self.sequence_running = False
                    self.sequence_start_btn.configure(state="normal")
                    self.sequence_stop_btn.configure(state="disabled")
                    self.sequence_gpu_toggle.configure(state="normal")
                    self.sequence_status.set("連続画像判定エラー")
                    self.sequence_log_line(e[1])
                    messagebox.showerror("連続画像判定エラー", e[1])
        except queue.Empty: pass
        self.root.after(150, self.poll)

if __name__ == "__main__":
    root = tk.Tk()
    try:
        if "vista" in ttk.Style().theme_names(): ttk.Style().theme_use("vista")
    except Exception: pass
    App(root)
    root.mainloop()
