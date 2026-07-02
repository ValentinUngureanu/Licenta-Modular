from __future__ import annotations

import contextlib
import importlib
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
    from PIL import Image
except (
    Exception
) as exc:  # pragma: no cover - doar fallback vizual pentru dependinte lipsa
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Dependinte lipsa",
        "Interfata moderna are nevoie de doua pachete:\n\n"
        "pip install customtkinter pillow\n\n"
        f"Eroare initiala:\n{exc}",
    )
    raise SystemExit(1)

# ============================================================
# INTERFATA MODERNA - PROIECT PLEURA
# ============================================================
# Instalare dependinte:
#   pip install customtkinter pillow
#
# Utilizare:
#   1. Pune fisierul in folderul proiectului, langa main.py si config.py.
#   2. Redenumeste-l in interface.py.
#   3. Ruleaza:
#        python interface.py
#
# Ce face:
#   - ruleaza pipeline-ul existent din main.py;
#   - nu modifica main.py;
#   - seteaza temporar config.INPUT_DIR / RESULTS_DIR / RUN_SINGLE_IMAGE;
#   - afiseaza logul;
#   - afiseaza galerie cu rezultatele finale din folderul 14;
#   - deschide rapid folderele importante.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FINAL_FOLDER_RELATIVE = (
    Path("FINAL_CONTOUR_TEST") / "14_VARIANTA_FINALA_PLEURA_INTRERUPERI_NODULI"
)
NODULE_DEBUG_RELATIVE = Path("FINAL_CONTOUR_TEST") / "13_PLEURAL_NODULE_MARKING"
INTERRUPTION_DEBUG_RELATIVE = (
    Path("FINAL_CONTOUR_TEST") / "12_PLEURAL_INTERRUPTION_MARKING"
)

PROJECT_MODULES_TO_RELOAD = [
    "crop",
    "final_contour",
    "gap_rescue",
    "horizontal_rescue",
    "image_io",
    "pleural_interruptions",
    "pleural_nodules",
    "postprocessing",
    "preprocessing",
    "principal_component",
    "principal_selector",
    "secondary_component",
    "secondary_rescue",
    "top2_final_contour",
    "top2_pleura",
    "traveler",
    "unification",
    "main",
]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class QueueWriter:
    def __init__(self, output_queue: queue.Queue[str]) -> None:
        self.output_queue = output_queue

    def write(self, text: str) -> None:
        if text:
            self.output_queue.put(text)

    def flush(self) -> None:
        pass


def safe_int(value: str, fallback: int) -> int:
    try:
        return int(float(str(value).replace(",", ".").strip()))
    except Exception:
        return fallback


def open_path(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        messagebox.showwarning("Folder lipsa", f"Nu exista:\n{path}")
        return

    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def numeric_image_indices(input_dir: Path) -> list[int]:
    if not input_dir.exists():
        return []

    indices: list[int] = []
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem.isdigit():
            indices.append(int(path.stem))

    return sorted(set(indices))


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    result: list[Path] = []
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            result.append(path)

    def sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem.split("_")[0]
        if stem.isdigit():
            return int(stem), path.name
        return 999999, path.name

    return sorted(result, key=sort_key)


def fit_image_size(
    width: int, height: int, max_width: int, max_height: int
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return max_width, max_height
    scale = min(max_width / width, max_height / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


class PleuraDashboard(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Pleura Analysis Dashboard")
        self.geometry("1320x820")
        self.minsize(1180, 720)

        self.output_queue: queue.Queue[str] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.running = False
        self.start_time = 0.0

        self.input_dir_var = ctk.StringVar(value=str(PROJECT_DIR / "ORIGINAL_IMAGES"))
        self.results_dir_var = ctk.StringVar(value=str(PROJECT_DIR / "RESULTS"))
        self.single_idx_var = ctk.StringVar(value="0")
        self.start_idx_var = ctk.StringVar(value="0")
        self.end_idx_var = ctk.StringVar(value="60")
        self.status_var = ctk.StringVar(value="Gata de rulare")
        self.summary_var = ctk.StringVar(
            value="Alege imaginile si ruleaza pipeline-ul."
        )
        self.preview_title_var = ctk.StringVar(value="Nu exista rezultate incarcate.")

        self.gallery_paths: list[Path] = []
        self.gallery_index = 0
        self.preview_image: ctk.CTkImage | None = None

        self._build_layout()
        self._refresh_detected_indices()
        self.after(120, self._poll_output_queue)

    # --------------------------------------------------------
    # UI BUILDERS
    # --------------------------------------------------------
    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(12, weight=1)

        self.content = ctk.CTkFrame(
            self, corner_radius=0, fg_color=("#F3F6FB", "#0C111A")
        )
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=3)
        self.content.grid_columnconfigure(1, weight=2)
        self.content.grid_rowconfigure(1, weight=1)

        self._build_sidebar()
        self._build_top_bar()
        self._build_preview_card()
        self._build_log_card()
        self._build_bottom_bar()

    def _build_sidebar(self) -> None:
        title = ctk.CTkLabel(
            self.sidebar,
            text="PLEURA\nANALYSIS",
            font=ctk.CTkFont(size=25, weight="bold"),
            justify="left",
        )
        title.grid(row=0, column=0, padx=24, pady=(28, 6), sticky="w")

        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Detectare pleura • intreruperi • noduli",
            font=ctk.CTkFont(size=12),
            text_color=("#5B6470", "#9AA6B2"),
            justify="left",
        )
        subtitle.grid(row=1, column=0, padx=24, pady=(0, 24), sticky="w")

        self.run_all_button = self._side_button(
            "Ruleaza toate imaginile", self._run_all
        )
        self.run_all_button.grid(row=2, column=0, padx=18, pady=(4, 8), sticky="ew")

        self.run_single_button = self._side_button(
            "Ruleaza imaginea aleasa", self._run_single
        )
        self.run_single_button.grid(row=3, column=0, padx=18, pady=8, sticky="ew")

        self.run_range_button = self._side_button("Ruleaza interval", self._run_range)
        self.run_range_button.grid(row=4, column=0, padx=18, pady=8, sticky="ew")

        separator = ctk.CTkFrame(
            self.sidebar, height=1, fg_color=("#D4DAE3", "#263243")
        )
        separator.grid(row=5, column=0, padx=22, pady=18, sticky="ew")

        self._side_button("Deschide folder final 14", self._open_final_folder).grid(
            row=6, column=0, padx=18, pady=8, sticky="ew"
        )
        self._side_button("Debug noduli", self._open_nodule_debug).grid(
            row=7, column=0, padx=18, pady=8, sticky="ew"
        )
        self._side_button("Debug intreruperi", self._open_interruption_debug).grid(
            row=8, column=0, padx=18, pady=8, sticky="ew"
        )
        self._side_button("Reincarca galeria", self._load_gallery).grid(
            row=9, column=0, padx=18, pady=8, sticky="ew"
        )

        self.progress = ctk.CTkProgressBar(self.sidebar, mode="indeterminate")
        self.progress.grid(row=13, column=0, padx=22, pady=(0, 8), sticky="ew")
        self.progress.set(0)

        self.status_label = ctk.CTkLabel(
            self.sidebar,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12),
            text_color=("#4A5565", "#AEB8C5"),
            wraplength=200,
            justify="left",
        )
        self.status_label.grid(row=14, column=0, padx=22, pady=(0, 22), sticky="w")

    def _side_button(self, text: str, command) -> ctk.CTkButton:
        return ctk.CTkButton(
            self.sidebar,
            text=text,
            command=command,
            height=42,
            corner_radius=14,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#2563EB", "#1D4ED8"),
            hover_color=("#1D4ED8", "#2563EB"),
        )

    def _build_top_bar(self) -> None:
        top = ctk.CTkFrame(
            self.content, corner_radius=18, fg_color=("#FFFFFF", "#111827")
        )
        top.grid(row=0, column=0, columnspan=2, padx=22, pady=(22, 12), sticky="ew")
        top.grid_columnconfigure(1, weight=1)

        header = ctk.CTkLabel(
            top,
            text="Dashboard procesare imagini ecografice",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header.grid(row=0, column=0, padx=22, pady=(18, 4), sticky="w")

        summary = ctk.CTkLabel(
            top,
            textvariable=self.summary_var,
            font=ctk.CTkFont(size=13),
            text_color=("#5B6470", "#AEB8C5"),
        )
        summary.grid(row=1, column=0, columnspan=3, padx=22, pady=(0, 12), sticky="w")

        self._path_row(
            top,
            row=2,
            label="Imagini",
            variable=self.input_dir_var,
            browse=self._browse_input,
        )
        self._path_row(
            top,
            row=3,
            label="Rezultate",
            variable=self.results_dir_var,
            browse=self._browse_results,
        )

        controls = ctk.CTkFrame(top, fg_color="transparent")
        controls.grid(row=4, column=0, columnspan=3, padx=22, pady=(8, 18), sticky="ew")
        controls.grid_columnconfigure((1, 3, 5), weight=1)

        ctk.CTkLabel(controls, text="Imagine", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=(0, 8), sticky="w"
        )
        ctk.CTkEntry(
            controls, textvariable=self.single_idx_var, width=80, height=34
        ).grid(row=0, column=1, padx=(0, 18), sticky="w")

        ctk.CTkLabel(controls, text="Start", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=2, padx=(0, 8), sticky="w"
        )
        ctk.CTkEntry(
            controls, textvariable=self.start_idx_var, width=80, height=34
        ).grid(row=0, column=3, padx=(0, 18), sticky="w")

        ctk.CTkLabel(controls, text="End", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=4, padx=(0, 8), sticky="w"
        )
        ctk.CTkEntry(controls, textvariable=self.end_idx_var, width=80, height=34).grid(
            row=0, column=5, padx=(0, 18), sticky="w"
        )

        ctk.CTkButton(
            controls,
            text="Detecteaza indicii",
            command=self._refresh_detected_indices,
            width=150,
            height=34,
            corner_radius=12,
            fg_color=("#0F766E", "#0F766E"),
            hover_color=("#115E59", "#14B8A6"),
        ).grid(row=0, column=6, padx=(8, 0), sticky="e")

    def _path_row(
        self, parent, row: int, label: str, variable: ctk.StringVar, browse
    ) -> None:
        ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=row, column=0, padx=(22, 12), pady=6, sticky="w"
        )
        ctk.CTkEntry(parent, textvariable=variable, height=36).grid(
            row=row, column=1, padx=(0, 10), pady=6, sticky="ew"
        )
        ctk.CTkButton(
            parent,
            text="Alege",
            command=browse,
            width=90,
            height=36,
            corner_radius=12,
        ).grid(row=row, column=2, padx=(0, 22), pady=6, sticky="e")

    def _build_preview_card(self) -> None:
        card = ctk.CTkFrame(
            self.content, corner_radius=18, fg_color=("#FFFFFF", "#111827")
        )
        card.grid(row=1, column=0, padx=(22, 10), pady=(0, 12), sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.grid(row=0, column=0, padx=18, pady=(18, 8), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Rezultat final",
            font=ctk.CTkFont(size=19, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        nav = ctk.CTkFrame(header, fg_color="transparent")
        nav.grid(row=0, column=1, sticky="e")
        ctk.CTkButton(nav, text="◀", width=44, command=self._previous_image).grid(
            row=0, column=0, padx=4
        )
        ctk.CTkButton(nav, text="▶", width=44, command=self._next_image).grid(
            row=0, column=1, padx=4
        )

        self.preview_label = ctk.CTkLabel(
            card,
            text="Ruleaza pipeline-ul sau incarca galeria.",
            font=ctk.CTkFont(size=15),
            text_color=("#475569", "#CBD5E1"),
            corner_radius=16,
            fg_color=("#EEF2F7", "#0B1220"),
        )
        self.preview_label.grid(row=1, column=0, padx=18, pady=(0, 10), sticky="nsew")

        self.preview_caption = ctk.CTkLabel(
            card,
            textvariable=self.preview_title_var,
            font=ctk.CTkFont(size=13),
            text_color=("#475569", "#AEB8C5"),
            anchor="w",
        )
        self.preview_caption.grid(row=2, column=0, padx=20, pady=(0, 16), sticky="ew")

    def _build_log_card(self) -> None:
        card = ctk.CTkFrame(
            self.content, corner_radius=18, fg_color=("#FFFFFF", "#111827")
        )
        card.grid(row=1, column=1, padx=(10, 22), pady=(0, 12), sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.grid(row=0, column=0, padx=18, pady=(18, 8), sticky="ew")
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            top, text="Log rulare", font=ctk.CTkFont(size=19, weight="bold")
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            top,
            text="Curata",
            width=80,
            height=32,
            corner_radius=12,
            command=self._clear_log,
        ).grid(row=0, column=1, sticky="e")

        self.log_box = ctk.CTkTextbox(
            card,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=14,
        )
        self.log_box.grid(row=1, column=0, padx=18, pady=(0, 18), sticky="nsew")
        self._append_log("Interfata pornita.\n")

    def _build_bottom_bar(self) -> None:
        bottom = ctk.CTkFrame(
            self.content, corner_radius=18, fg_color=("#FFFFFF", "#111827")
        )
        bottom.grid(row=2, column=0, columnspan=2, padx=22, pady=(0, 22), sticky="ew")
        bottom.grid_columnconfigure((0, 1, 2), weight=1)

        self.stat_images = self._stat_card(bottom, 0, "Imagini gasite", "0")
        self.stat_results = self._stat_card(bottom, 1, "Rezultate finale", "0")
        self.stat_folder = self._stat_card(bottom, 2, "Folder final", "14")

    def _stat_card(self, parent, column: int, label: str, value: str) -> ctk.CTkLabel:
        frame = ctk.CTkFrame(parent, corner_radius=14, fg_color=("#F8FAFC", "#0B1220"))
        frame.grid(row=0, column=column, padx=12, pady=14, sticky="ew")
        ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(size=12),
            text_color=("#64748B", "#94A3B8"),
        ).pack(anchor="w", padx=16, pady=(12, 2))
        value_label = ctk.CTkLabel(
            frame, text=value, font=ctk.CTkFont(size=22, weight="bold")
        )
        value_label.pack(anchor="w", padx=16, pady=(0, 12))
        return value_label

    # --------------------------------------------------------
    # PATHS / GALLERY
    # --------------------------------------------------------
    def _browse_input(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.input_dir_var.get())
        if selected:
            self.input_dir_var.set(selected)
            self._refresh_detected_indices()

    def _browse_results(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.results_dir_var.get())
        if selected:
            self.results_dir_var.set(selected)
            self._load_gallery()

    def _refresh_detected_indices(self) -> None:
        indices = numeric_image_indices(Path(self.input_dir_var.get()))
        if indices:
            self.start_idx_var.set(str(indices[0]))
            self.end_idx_var.set(str(indices[-1]))
            self.single_idx_var.set(str(indices[0]))
            self.summary_var.set(
                f"Am gasit {len(indices)} imagini numerotate: {indices[0]}..{indices[-1]}."
            )
        else:
            self.summary_var.set("Nu am gasit imagini numerotate in folderul selectat.")
        self.stat_images.configure(text=str(len(indices)))

    def _final_folder(self) -> Path:
        return Path(self.results_dir_var.get()) / FINAL_FOLDER_RELATIVE

    def _nodule_debug_folder(self) -> Path:
        return Path(self.results_dir_var.get()) / NODULE_DEBUG_RELATIVE

    def _interruption_debug_folder(self) -> Path:
        return Path(self.results_dir_var.get()) / INTERRUPTION_DEBUG_RELATIVE

    def _open_final_folder(self) -> None:
        open_path(self._final_folder())

    def _open_nodule_debug(self) -> None:
        open_path(self._nodule_debug_folder())

    def _open_interruption_debug(self) -> None:
        open_path(self._interruption_debug_folder())

    def _load_gallery(self) -> None:
        self.gallery_paths = image_files(self._final_folder())
        self.gallery_index = 0
        self.stat_results.configure(text=str(len(self.gallery_paths)))

        if not self.gallery_paths:
            self.preview_image = None
            self.preview_label.configure(
                image=None, text="Nu exista imagini in folderul final 14."
            )
            self.preview_title_var.set(str(self._final_folder()))
            return

        self._show_gallery_image(0)

    def _show_gallery_image(self, index: int) -> None:
        if not self.gallery_paths:
            return

        self.gallery_index = max(0, min(index, len(self.gallery_paths) - 1))
        path = self.gallery_paths[self.gallery_index]

        try:
            pil_image = Image.open(path).convert("RGB")
            max_w = max(520, self.preview_label.winfo_width() - 24)
            max_h = max(360, self.preview_label.winfo_height() - 24)
            new_w, new_h = fit_image_size(
                pil_image.width, pil_image.height, max_w, max_h
            )
            display = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.preview_image = ctk.CTkImage(
                light_image=display, dark_image=display, size=(new_w, new_h)
            )
            self.preview_label.configure(image=self.preview_image, text="")
            self.preview_title_var.set(
                f"{self.gallery_index + 1}/{len(self.gallery_paths)}  •  {path.name}"
            )
        except Exception as exc:
            self.preview_label.configure(
                image=None, text=f"Nu pot incarca imaginea:\n{path}\n\n{exc}"
            )
            self.preview_title_var.set(path.name)

    def _previous_image(self) -> None:
        if self.gallery_paths:
            self._show_gallery_image(self.gallery_index - 1)

    def _next_image(self) -> None:
        if self.gallery_paths:
            self._show_gallery_image(self.gallery_index + 1)

    # --------------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------------
    def _run_all(self) -> None:
        indices = numeric_image_indices(Path(self.input_dir_var.get()))
        if not indices:
            messagebox.showwarning("Lipsa imagini", "Nu am gasit imagini numerotate.")
            return
        self._start_run(mode="range", start_idx=indices[0], end_idx=indices[-1])

    def _run_single(self) -> None:
        index = safe_int(self.single_idx_var.get(), 0)
        self._start_run(mode="single", single_idx=index)

    def _run_range(self) -> None:
        start = safe_int(self.start_idx_var.get(), 0)
        end = safe_int(self.end_idx_var.get(), start)
        if end < start:
            messagebox.showwarning("Interval invalid", "End trebuie sa fie >= Start.")
            return
        self._start_run(mode="range", start_idx=start, end_idx=end)

    def _start_run(
        self,
        mode: str,
        single_idx: int | None = None,
        start_idx: int | None = None,
        end_idx: int | None = None,
    ) -> None:
        if self.running:
            messagebox.showinfo("Rulare activa", "Pipeline-ul ruleaza deja.")
            return

        input_dir = Path(self.input_dir_var.get())
        if not input_dir.exists():
            messagebox.showwarning(
                "Folder invalid", f"Nu exista folderul de imagini:\n{input_dir}"
            )
            return

        results_dir = Path(self.results_dir_var.get())
        results_dir.mkdir(parents=True, exist_ok=True)

        self.running = True
        self.start_time = time.time()
        self.status_var.set("Ruleaza...")
        self.progress.start()
        self._set_run_buttons_state(False)
        self._append_log("\n" + "=" * 70 + "\n")
        self._append_log("Pornesc pipeline-ul...\n")

        self.worker_thread = threading.Thread(
            target=self._worker_run,
            kwargs={
                "mode": mode,
                "single_idx": single_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "input_dir": input_dir,
                "results_dir": results_dir,
            },
            daemon=True,
        )
        self.worker_thread.start()

    def _set_run_buttons_state(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in [
            self.run_all_button,
            self.run_single_button,
            self.run_range_button,
        ]:
            button.configure(state=state)

    def _worker_run(
        self,
        mode: str,
        single_idx: int | None,
        start_idx: int | None,
        end_idx: int | None,
        input_dir: Path,
        results_dir: Path,
    ) -> None:
        writer = QueueWriter(self.output_queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                main_module = self._prepare_project_modules(
                    mode=mode,
                    single_idx=single_idx,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    input_dir=input_dir,
                    results_dir=results_dir,
                )
                main_module.main()
            self.output_queue.put("\n[DONE] Rulare terminata.\n")
        except Exception:
            self.output_queue.put("\n[EROARE]\n")
            self.output_queue.put(traceback.format_exc())
        finally:
            self.output_queue.put("__RUN_FINISHED__")

    def _prepare_project_modules(
        self,
        mode: str,
        single_idx: int | None,
        start_idx: int | None,
        end_idx: int | None,
        input_dir: Path,
        results_dir: Path,
    ):
        config = importlib.import_module("config")
        config.INPUT_DIR = input_dir
        config.RESULTS_DIR = results_dir

        if mode == "single":
            config.RUN_SINGLE_IMAGE = True
            config.SINGLE_IMAGE_IDX = int(single_idx if single_idx is not None else 0)
            print(f"Mod rulare: imagine unica {config.SINGLE_IMAGE_IDX}")
        else:
            config.RUN_SINGLE_IMAGE = False
            config.START_IDX = int(start_idx if start_idx is not None else 0)
            # In config, END_IDX este exclusiv. In interfata, end este inclusiv.
            config.END_IDX = (
                int(end_idx if end_idx is not None else config.START_IDX) + 1
            )
            print(f"Mod rulare: interval {config.START_IDX}..{config.END_IDX - 1}")

        print(f"Input : {config.INPUT_DIR}")
        print(f"Output: {config.RESULTS_DIR}")

        for module_name in PROJECT_MODULES_TO_RELOAD:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])

        return importlib.import_module("main")

    def _poll_output_queue(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item == "__RUN_FINISHED__":
                    self.running = False
                    self.progress.stop()
                    elapsed = time.time() - self.start_time
                    self.status_var.set(f"Gata. Timp rulare: {elapsed:.1f}s")
                    self._set_run_buttons_state(True)
                    self._load_gallery()
                    continue
                self._append_log(item)
        except queue.Empty:
            pass

        self.after(120, self._poll_output_queue)

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------
    def _append_log(self, text: str) -> None:
        self.log_box.insert("end", text)
        self.log_box.see("end")

    def _clear_log(self) -> None:
        self.log_box.delete("1.0", "end")


def main() -> None:
    app = PleuraDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
