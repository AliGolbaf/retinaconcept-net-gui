#!/usr/bin/env python3
"""RetinaXAI — local research desktop for explainable diabetic retinopathy.

Launch: python app.py
Models: supply the genuine fold checkpoints from the training pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from model_download import download_models
from model_runtime import (
    CONCEPT_CODES, CONCEPT_NAMES, GRADE_NAMES, RuntimeModels, decode_fundus,
    discover_checkpoints,
    margin_contributions, scenario_probabilities,
)

APP_VERSION = "1.5"


COLORS = {
    "bg": "#0C1322", "sidebar": "#101A2D", "card": "#172237",
    "card_alt": "#1D2A40", "canvas": "#0C1728", "border": "#29384F",
    "text": "#F3F7FF", "muted": "#A5B3C8", "dim": "#73849D",
    "teal": "#27D5C3", "blue": "#5A9BFF", "orange": "#F7B85A",
    "red": "#FC777C", "green": "#63DBA6", "purple": "#B79AFD",
}


def label(parent, text="", *, size=11, bold=False, color="text", bg=None,
          anchor="w", wraplength=0, **kwargs):
    return tk.Label(parent, text=text, font=("Segoe UI", size, "bold" if bold else "normal"),
                    foreground=COLORS.get(color, color), background=bg or COLORS["card"],
                    anchor=anchor, justify="left", wraplength=wraplength, **kwargs)


def button(parent, text, command, *, primary=False, small=False):
    """Classic Tk button reserved for the custom sidebar navigation."""
    color = COLORS["teal"] if primary else COLORS["card_alt"]
    foreground = COLORS["bg"] if primary else COLORS["text"]
    widget = tk.Button(parent, text=text, command=command,
                       font=("Segoe UI", 10 if small else 11, "bold"),
                       background=color, foreground=foreground,
                       activebackground=COLORS["blue"] if primary else COLORS["border"],
                       activeforeground=COLORS["bg"] if primary else COLORS["text"],
                       highlightthickness=0, borderwidth=0, relief="flat", cursor="hand2",
                       padx=18, pady=10 if not small else 7,
                       disabledforeground=COLORS["dim"])
    return widget


def action_button(parent, text, command, *, primary=False):
    """Native ttk control styled by the bundled Azure theme."""
    return ttk.Button(parent, text=text, command=command,
                      style="Accent.TButton" if primary else "TButton")


def card(parent, title, subtitle="", *, accent="blue"):
    frame = tk.Frame(parent, background=COLORS["card"], highlightthickness=1,
                     highlightbackground=COLORS["border"])
    tk.Frame(frame, background=COLORS.get(accent, accent), height=4).pack(fill="x")
    heading = tk.Frame(frame, background=COLORS["card"])
    heading.pack(fill="x", padx=20, pady=(16, 11))
    label(heading, title, size=13, bold=True).pack(anchor="w")
    if subtitle:
        label(heading, subtitle, size=9, color="muted", wraplength=650).pack(anchor="w", pady=(4, 0))
    body = tk.Frame(frame, background=COLORS["card"])
    body.pack(fill="both", expand=True, padx=20, pady=(0, 18))
    return frame, body


class ScrollablePage(tk.Frame):
    """Full-width page with independent scrolling and a fixed navigation bar."""

    def __init__(self, parent):
        super().__init__(parent, background=COLORS["bg"])
        self.footer = tk.Frame(self, background=COLORS["sidebar"], height=52)
        self.footer.pack(side="bottom", fill="x", pady=(12, 0))
        self.footer.pack_propagate(False)

        area = tk.Frame(self, background=COLORS["bg"])
        area.pack(fill="both", expand=True)
        self.scrollbar = ttk.Scrollbar(area, orient="vertical")
        self.scrollbar.pack(side="right", fill="y")
        self.canvas = tk.Canvas(area, background=COLORS["bg"],
                                highlightthickness=0, borderwidth=0,
                                yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.configure(command=self.canvas.yview)
        self.body = tk.Frame(self.canvas, background=COLORS["bg"])
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self._resize_job = None
        self._last_content_size = (0, 0)
        self._last_viewport_size = (0, 0)
        self.on_viewport_resize = None
        self.body.bind("<Configure>", self._schedule_resize)
        self.canvas.bind("<Configure>", self._schedule_resize)

    def _schedule_resize(self, _event=None):
        if self._resize_job is None:
            self._resize_job = self.after_idle(self._resize)

    def _resize(self):
        self._resize_job = None
        width = max(1, self.canvas.winfo_width())
        viewport_height = max(1, self.canvas.winfo_height())
        viewport_size = (width, viewport_height)
        if viewport_size != self._last_viewport_size:
            self._last_viewport_size = viewport_size
            if self.on_viewport_resize is not None:
                self.on_viewport_resize(width, viewport_height)
        height = max(self.canvas.winfo_height(), self.body.winfo_reqheight())
        if (width, height) != self._last_content_size:
            self.canvas.itemconfigure(self._window, width=width, height=height)
            self._last_content_size = (width, height)
        self.canvas.configure(scrollregion=(0, 0, width, height))

    def scroll(self, units):
        self.canvas.yview_scroll(units, "units")

    def scroll_to_top(self):
        self.canvas.yview_moveto(0)


class ProbabilityChart(tk.Canvas):
    """Compact, dependency-free five-class probability bars."""

    def __init__(self, parent, *, height=215):
        super().__init__(parent, height=height, background=COLORS["card"],
                         borderwidth=0, highlightthickness=0)
        self.values = None
        self.comparison = None
        self.bind("<Configure>", lambda _e: self.draw())

    def show(self, probabilities, comparison=None):
        self.values = None if probabilities is None else np.asarray(probabilities, dtype=float)
        self.comparison = None if comparison is None else np.asarray(comparison, dtype=float)
        self.draw()

    def draw(self):
        self.delete("all")
        width = max(200, self.winfo_width())
        if self.values is None:
            self.create_text(width / 2, max(35, self.winfo_height() / 2),
                             text="Results appear after running inference",
                             fill=COLORS["dim"], font=("Segoe UI", 10))
            return
        label_width, right = 128, width - 58
        track_width = max(45, right - label_width)
        height = max(100, self.winfo_height())
        step = min(37.0, max(19.0, (height - 38.0) / 4))
        top = max(18.0, (height - 4 * step) / 2)
        for i, (grade, value) in enumerate(zip(GRADE_NAMES, self.values)):
            y = top + step * i
            self.create_text(2, y, text=grade, fill=COLORS["muted"],
                             font=("Segoe UI", 9), anchor="w")
            self.create_rectangle(label_width, y - 9, right, y + 9,
                                  fill=COLORS["card_alt"], outline="")
            fill_width = max(0.0, track_width * float(np.clip(value, 0, 1)))
            active = i == int(np.argmax(self.values))
            first_top, first_bottom = (y - 9, y - 1) if self.comparison is not None else (y - 8, y + 8)
            self.create_rectangle(label_width, first_top, label_width + fill_width, first_bottom,
                                  fill=COLORS["teal"] if active else COLORS["blue"], outline="")
            if self.comparison is not None:
                other_width = track_width * float(np.clip(self.comparison[i], 0, 1))
                self.create_rectangle(label_width, y + 2, label_width + other_width, y + 9,
                                      fill=COLORS["purple"], outline="")
            self.create_text(width - 4, y, text=f"{value * 100:.1f}%",
                             fill=COLORS["text"] if active else COLORS["muted"],
                             font=("Segoe UI", 9, "bold" if active else "normal"), anchor="e")


class ContributionChart(tk.Canvas):
    """Signed concept contributions to a fixed two-grade logit contrast."""

    def __init__(self, parent):
        super().__init__(parent, background=COLORS["card"],
                         borderwidth=0, highlightthickness=0, height=250)
        self.values = None
        self.bind("<Configure>", lambda _e: self.draw())

    def show(self, values):
        self.values = None if values is None else np.asarray(values, dtype=float)
        self.draw()

    def draw(self):
        self.delete("all")
        height = max(120, self.winfo_height())
        if self.values is None:
            self.create_text(max(180, self.winfo_width()) / 2, height / 2,
                             text="Generate a CBM result to inspect contributions",
                             fill=COLORS["dim"], font=("Segoe UI", 10))
            return
        width = max(420, self.winfo_width())
        axis = width * 0.59
        available = min(axis - 158, width - axis - 50)
        available = max(35, available)
        max_abs = max(float(np.max(np.abs(self.values))), 1e-8)
        row_step = min(39.0, max(22.0, (height - 28.0) / 7))
        first_y = (height - row_step * 6) / 2
        self.create_line(axis, max(8, first_y - 19), axis,
                         min(height - 8, first_y + row_step * 6 + 19),
                         fill=COLORS["dim"], width=1)
        for row, index in enumerate(np.argsort(-np.abs(self.values))):
            value = float(self.values[index])
            y = first_y + row * row_step
            self.create_text(8, y, text=CONCEPT_CODES[index], fill=COLORS["muted"],
                             font=("Segoe UI", 10, "bold"), anchor="w")
            if row_step >= 32:
                self.create_text(axis - 12, y - 15, text=CONCEPT_NAMES[index],
                                 fill=COLORS["dim"], font=("Segoe UI", 8), anchor="e")
            magnitude = abs(value) / max_abs * available
            if value >= 0:
                self.create_rectangle(axis, y - 7, axis + magnitude, y + 7,
                                      fill=COLORS["teal"], outline="")
                self.create_text(min(width - 4, axis + magnitude + 5), y,
                                 text=f"{value:+.3f}", anchor="w", fill=COLORS["text"],
                                 font=("Segoe UI", 9))
            else:
                self.create_rectangle(axis - magnitude, y - 7, axis, y + 7,
                                      fill=COLORS["red"], outline="")
                self.create_text(max(155, axis - magnitude - 5), y,
                                 text=f"{value:+.3f}", anchor="e", fill=COLORS["text"],
                                 font=("Segoe UI", 9))


class RetinaXAI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.azure_loaded = self._initialise_azure_theme()
        self.title(f"RetinaConcept-Net v{APP_VERSION} | Explainable Diabetic Retinopathy Assessment")
        self.geometry("1420x880")
        self.minsize(1080, 680)
        self.configure(background=COLORS["bg"])

        self.engine = RuntimeModels()
        self.pending = queue.Queue()
        self.busy = False
        self.image_bytes: bytes | None = None
        self.image: Image.Image | None = None
        self.image_name = ""
        self.results = {}
        self.selected_concept = tk.IntVar(value=0)
        self.opacity = tk.DoubleVar(value=65)
        self.threshold = tk.DoubleVar(value=25)
        self.opacity_text = tk.StringVar(value="65%")
        self.threshold_text = tk.StringVar(value="25%")
        self.whatif_vars: list[tk.DoubleVar] = []
        self.whatif_value_labels: list[tk.Label] = []
        self._updating_sliders = False
        self._photo_references: dict[tk.Canvas, ImageTk.PhotoImage] = {}
        self.model_vars = {kind: tk.StringVar(value="Waiting for automatic loading…")
                           for kind in ("blackbox", "cbm")}
        self.model_load_buttons: dict[str, ttk.Button] = {}
        self._download_offered = False
        self.status_var = tk.StringVar(value=f"RetinaXAI v{APP_VERSION} — checking the models/ folder…")
        self._nav_buttons: dict[str, tk.Button] = {}
        self._pages: dict[str, tk.Frame] = {}
        self._page_shells: dict[str, ScrollablePage] = {}
        self._active_page = "Overview"
        self._build_ui()
        self.show_page("Overview")
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.after(80, self._poll_workers)
        self.after(180, self.autoload_models)

    def _initialise_azure_theme(self) -> bool:
        """Load the bundled Azure ttk theme; keep a usable fallback if unavailable."""
        theme_file = Path(__file__).resolve().parent / "azure" / "azure.tcl"
        if not theme_file.is_file():
            return False
        try:
            self.tk.call("source", str(theme_file))
            self.tk.call("set_theme", "dark")
            style = ttk.Style(self)
            style.configure("TButton", font=("Segoe UI", 10))
            style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
            style.configure("TRadiobutton", font=("Segoe UI", 9),
                            background=COLORS["card"], foreground=COLORS["text"])
            style.map("TRadiobutton", background=[("active", COLORS["card"])])
            style.configure("Tick.TScale", background=COLORS["card"])
            return True
        except tk.TclError:
            # A missing/unsupported theme must never prevent the app launching.
            return False

    def _build_ui(self):
        header = tk.Frame(self, background=COLORS["sidebar"], height=82)
        header.pack(fill="x")
        header.pack_propagate(False)

        logo = tk.Canvas(header, width=48, height=48, background=COLORS["sidebar"],
                         highlightthickness=0)
        logo.pack(side="left", padx=(30, 12), pady=16)
        logo.create_oval(6, 6, 42, 42, outline=COLORS["teal"], width=3)
        logo.create_oval(17, 17, 31, 31, fill=COLORS["blue"], outline="")
        logo.create_arc(10, 10, 38, 38, start=225, extent=130, outline=COLORS["orange"], width=3)

        identity = tk.Frame(header, background=COLORS["sidebar"])
        identity.pack(side="left", fill="y", pady=13)
        label(identity, "RetinaConcept-Net", size=21, bold=True, bg=COLORS["sidebar"]).pack(anchor="w")
        label(identity, "Transparent retinal AI  •  Research interface", size=10,
              color="muted", bg=COLORS["sidebar"]).pack(anchor="w")
        tk.Label(header, text="RESEARCH USE ONLY", foreground=COLORS["teal"],
                 background=COLORS["card_alt"], font=("Segoe UI", 9, "bold"),
                 padx=14, pady=9).pack(side="right", padx=28)
        tk.Label(header, text=f"v{APP_VERSION} · AZURE DARK" if self.azure_loaded else f"v{APP_VERSION} · STANDARD THEME",
                 foreground=COLORS["blue"],
                 background=COLORS["sidebar"], font=("Segoe UI", 9, "bold"),
                 padx=8).pack(side="right")

        shell = tk.Frame(self, background=COLORS["bg"])
        shell.pack(fill="both", expand=True)
        sidebar = tk.Frame(shell, width=210, background=COLORS["sidebar"])
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        label(sidebar, "WORKSPACE", size=9, bold=True, color="dim",
              bg=COLORS["sidebar"]).pack(anchor="w", padx=24, pady=(30, 13))

        for page, caption in (("Overview", "01   Overview"),
                              ("Lesion Evidence", "02   Lesion evidence"),
                              ("Decision Explorer", "03   Decision explorer")):
            nav = button(sidebar, caption, lambda name=page: self.show_page(name))
            nav.configure(anchor="w", background=COLORS["sidebar"],
                          activebackground=COLORS["card_alt"], pady=13)
            nav.pack(fill="x", padx=9, pady=3)
            self._nav_buttons[page] = nav
        label(sidebar, "5 DR grades\n7 lesion concepts\nResNet-152 backbone",
              size=10, color="dim", bg=COLORS["sidebar"]).pack(
                  side="bottom", anchor="w", padx=25, pady=28)

        self.content_host = tk.Frame(shell, background=COLORS["bg"])
        self.content_host.pack(side="left", fill="both", expand=True, padx=22, pady=18)
        page_names = ("Overview", "Lesion Evidence", "Decision Explorer")
        for index, name in enumerate(page_names):
            page_shell = ScrollablePage(self.content_host)
            page_shell.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._pages[name] = page_shell.body
            self._page_shells[name] = page_shell

            toolbar = page_shell.footer
            label(toolbar, f"STEP {index + 1:02d} / 03", size=9, bold=True,
                  color="teal", bg=COLORS["sidebar"]).pack(side="left", padx=18)
            label(toolbar, name, size=10, color="muted",
                  bg=COLORS["sidebar"]).pack(side="left", padx=(6, 0))
            if index < len(page_names) - 1:
                action_button(toolbar, "Next  →", lambda n=page_names[index + 1]: self.show_page(n),
                              primary=True).pack(side="right", padx=(6, 18), pady=7)
            if index > 0:
                previous = action_button(toolbar, "←  Previous",
                                         lambda n=page_names[index - 1]: self.show_page(n))
                previous.pack(side="right", padx=6, pady=7)

        self._build_overview(self._pages["Overview"])
        self._build_evidence(self._pages["Lesion Evidence"])
        self._build_decision(self._pages["Decision Explorer"])
        self._page_shells["Overview"].on_viewport_resize = self._fit_overview_page
        self._page_shells["Decision Explorer"].on_viewport_resize = self._fit_decision_page

        foot = tk.Frame(self, height=36, background=COLORS["sidebar"])
        foot.pack(fill="x")
        label(foot, "●", color="teal", bg=COLORS["sidebar"], size=11).pack(side="left", padx=(24, 7))
        label(foot, text="", bg=COLORS["sidebar"], size=10, color="muted",
              textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(foot, mode="indeterminate", length=120)

    def show_page(self, name):
        self._active_page = name
        self._page_shells[name].tkraise()
        self._page_shells[name].scroll_to_top()
        for page, item in self._nav_buttons.items():
            selected = page == name
            item.configure(background=COLORS["card_alt"] if selected else COLORS["sidebar"],
                           foreground=COLORS["teal"] if selected else COLORS["muted"])

    def _on_mousewheel(self, event):
        if self._active_page in self._page_shells and event.delta:
            amount = -1 if event.delta > 0 else 1
            self._page_shells[self._active_page].scroll(amount * 3)

    def _build_overview(self, page):
        page.grid_columnconfigure(0, weight=3)
        page.grid_columnconfigure(1, weight=2)
        # Let both cards fit the viewport naturally; scroll only if their
        # actual requested size exceeds the available height.
        page.grid_rowconfigure(1, weight=1)
        page.grid_rowconfigure(2, weight=1)
        label(page, "Fundus analysis dashboard", size=20, bold=True,
              bg=COLORS["bg"]).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        viewer_card, viewer = card(page, "Fundus image", "Load a colour fundus photograph (JPG or PNG).",
                                   accent="teal")
        viewer_card.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(0, 8))
        viewer.grid_rowconfigure(0, weight=1)
        viewer.grid_columnconfigure(0, weight=1)
        self.overview_canvas = tk.Canvas(viewer, background=COLORS["canvas"],
                                         highlightthickness=0)
        self.overview_canvas.grid(row=0, column=0, sticky="nsew")
        self.overview_canvas.bind("<Configure>", lambda _e: self._draw_image(self.overview_canvas))
        actions = tk.Frame(viewer, background=COLORS["card"])
        actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self.open_button = action_button(actions, "Choose fundus image", self.choose_image, primary=True)
        self.open_button.pack(side="left")
        self.image_info = tk.StringVar(value="No image selected")
        label(actions, "", textvariable=self.image_info, size=9, color="muted").pack(side="left", padx=15)

        model_card, model_body = card(page, "Model configuration",
                                     "Auto-load from models/ or choose a checkpoint manually.", accent="blue")
        model_card.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        for kind, title in (("blackbox", "Black-box ResNet-152"),
                            ("cbm", "Concept Bottleneck Model")):
            line = tk.Frame(model_body, background=COLORS["card"])
            line.pack(fill="x", pady=(0, 5))
            heading = tk.Frame(line, background=COLORS["card"])
            heading.pack(fill="x")
            label(heading, title, size=10, bold=True).pack(side="left", anchor="w")
            load_button = action_button(heading, "Load…",
                                        lambda name=kind: self.choose_model_checkpoint(name))
            load_button.pack(side="right", padx=(8, 0))
            self.model_load_buttons[kind] = load_button
            label(line, "", textvariable=self.model_vars[kind],
                  size=9, color="muted", wraplength=260).pack(anchor="w", pady=(2, 0))
        model_actions = tk.Frame(model_body, background=COLORS["card"])
        model_actions.pack(fill="x", pady=(6, 0))
        self.reload_button = action_button(model_actions, "Reload models folder", self.autoload_models)
        self.reload_button.pack(side="left", padx=(0, 8))
        self.download_button = action_button(model_actions, "Download weights",
                                             self.download_model_weights)
        self.download_button.pack(side="left")
        self.run_button = action_button(model_body, "Run retinal analysis", self.run_analysis,
                                        primary=True)
        self.run_button.pack(fill="x", pady=(9, 0), ipady=3)
        self.run_button.configure(state="disabled")

        result_card, result_body = card(page, "DR grading result",
                                        "Model softmax scores are not calibrated clinical probabilities.",
                                        accent="purple")
        result_card.grid(row=2, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))
        self.grade_summary = tk.StringVar(value="Awaiting model inference")
        label(result_body, "", textvariable=self.grade_summary,
              size=12, bold=True, color="teal").pack(anchor="w", pady=(0, 4))
        self.chart_title = tk.StringVar(value="Concept Bottleneck Model")
        label(result_body, "", textvariable=self.chart_title,
              size=9, color="muted").pack(anchor="w")
        self.overview_chart = ProbabilityChart(result_body, height=155)
        self.overview_chart.pack(fill="both", expand=True, pady=(6, 0))

    def _fit_overview_page(self, _width: int, viewport_height: int):
        """Fit the grading chart to the window while keeping the image flexible.

        On smaller displays, the page scrollbar remains available rather than
        hiding the model controls or compressing the chart labels too far.
        """
        chart_height = max(135, min(215, int(viewport_height * 0.20)))
        if int(self.overview_chart.cget("height")) != chart_height:
            self.overview_chart.configure(height=chart_height)

    def _build_evidence(self, page):
        page.grid_columnconfigure(0, weight=3)
        page.grid_columnconfigure(1, weight=2)
        page.grid_rowconfigure(1, weight=1)
        label(page, "Lesion evidence explorer", size=20, bold=True,
              bg=COLORS["bg"]).grid(row=0, column=0, columnspan=2,
                                      sticky="w", pady=(0, 14))
        viewer_card, viewer = card(page, "Spatial concept evidence",
                                   "Model evidence activations, not validated pixel-wise lesion segmentation.",
                                   accent="orange")
        viewer_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        viewer.grid_columnconfigure(0, weight=1)
        viewer.grid_rowconfigure(0, weight=1)
        self.evidence_canvas = tk.Canvas(viewer, background=COLORS["canvas"], highlightthickness=0)
        self.evidence_canvas.grid(row=0, column=0, sticky="nsew")
        self.evidence_canvas.bind("<Configure>", lambda _e: self._draw_image(self.evidence_canvas, evidence=True))

        controls = tk.Frame(viewer, background=COLORS["card"])
        controls.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        controls.grid_columnconfigure(1, weight=1)
        label(controls, "Overlay opacity", size=9, color="muted").grid(row=0, column=0, sticky="w")
        opacity_scale = ttk.Scale(controls, from_=0, to=100, variable=self.opacity,
                                  orient="horizontal", style="Tick.TScale" if self.azure_loaded else "TScale",
                                  command=self._on_overlay_control_change)
        opacity_scale.grid(row=0, column=1, sticky="ew", padx=12)
        label(controls, "", textvariable=self.opacity_text, size=9, color="teal",
              width=4).grid(row=0, column=2, sticky="e")
        label(controls, "Evidence threshold", size=9, color="muted").grid(row=1, column=0, sticky="w")
        threshold_scale = ttk.Scale(controls, from_=0, to=95, variable=self.threshold,
                                    orient="horizontal", style="Tick.TScale" if self.azure_loaded else "TScale",
                                    command=self._on_overlay_control_change)
        threshold_scale.grid(row=1, column=1, sticky="ew", padx=12)
        label(controls, "", textvariable=self.threshold_text, size=9, color="teal",
              width=4).grid(row=1, column=2, sticky="e")

        side_card, side = card(page, "Seven learned lesion concepts",
                              "Choose a concept to view its evidence heatmap and image-level score.",
                              accent="teal")
        side_card.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        self.concept_radios = []
        self.concept_value_labels = []
        for i, (code, title) in enumerate(zip(CONCEPT_CODES, CONCEPT_NAMES)):
            row = tk.Frame(side, background=COLORS["card"])
            row.pack(fill="x", pady=5)
            radio = ttk.Radiobutton(row, text=f"{code}  {title}", variable=self.selected_concept,
                                    value=i, command=self._refresh_evidence, style="TRadiobutton")
            radio.pack(side="left", fill="x", expand=True)
            score = label(row, "—", size=10, bold=True, color="teal")
            score.pack(side="right", padx=6)
            self.concept_radios.append(radio)
            self.concept_value_labels.append(score)
        self.evidence_info = tk.StringVar(value="Run the Concept Bottleneck Model to inspect evidence.")
        label(side, "", textvariable=self.evidence_info, color="muted",
              size=10, wraplength=300).pack(anchor="w", pady=(14, 0))

    def _build_decision(self, page):
        page.grid_columnconfigure(0, weight=5)
        page.grid_columnconfigure(1, weight=3)
        page.grid_rowconfigure(1, weight=1)
        label(page, "Concept bottleneck decision explorer", size=20,
              bold=True, bg=COLORS["bg"]).grid(row=0, column=0, columnspan=2,
                                                 sticky="w", pady=(0, 14))
        left = tk.Frame(page, background=COLORS["bg"])
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        # Give the written explanation the full left-hand width, while keeping
        # the contribution chart more compact within the upper row.
        left.grid_columnconfigure(0, weight=4)
        left.grid_columnconfigure(1, weight=1)
        left.grid_rowconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=3)

        contribution_card, contribution_body = card(
            left, "Signed concept contributions",
            "Exact additive contributions to the predicted-versus-runner-up logit margin.",
            accent="purple")
        contribution_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.contribution_caption = tk.StringVar(value="Awaiting CBM inference")
        label(contribution_body, "", textvariable=self.contribution_caption,
              size=9, color="muted").pack(anchor="w")
        self.contribution_chart = ContributionChart(contribution_body)
        self.contribution_chart.pack(fill="both", expand=True)

        explanation_card, explanation_body = card(left, "Case-level explanation",
                                                   "Faithful explanation of the linear grading decision.",
                                                   accent="green")
        explanation_card.grid(row=1, column=0, columnspan=2,
                              sticky="nsew", pady=(8, 0))
        self.explanation = tk.Text(explanation_body, wrap="word", borderwidth=0,
                                   highlightthickness=0, background=COLORS["card"],
                                   foreground=COLORS["text"], font=("Segoe UI", 10),
                                   height=10, insertbackground=COLORS["teal"])
        self.explanation.pack(fill="both", expand=True)
        self._set_explanation("Run CBM inference to generate an explanation based on the trained grading weights.")

        scenario_card, scenario = card(page, "What-if concept sensitivity",
                                       "Adjust the seven classifier inputs; the image model is NOT rerun.",
                                       accent="orange")
        scenario_card.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        self.scenario_title = tk.StringVar(value="Awaiting CBM inference")
        label(scenario, "", textvariable=self.scenario_title, size=11,
              bold=True, color="teal").pack(anchor="w", pady=(0, 8))
        self.scenario_chart = ProbabilityChart(scenario, height=215)
        self.scenario_chart.pack(fill="x")

        self.scenario_vars_container = tk.Frame(scenario, background=COLORS["card"])
        self.scenario_vars_container.pack(fill="both", expand=True, pady=(12, 0))
        for index, code in enumerate(CONCEPT_CODES):
            line = tk.Frame(self.scenario_vars_container, background=COLORS["card"])
            line.pack(fill="x", pady=1)
            label(line, code, size=9, bold=True, color="muted").pack(side="left")
            var = tk.DoubleVar(value=50)
            value_label = label(line, "50%", size=9, color="teal", width=5, anchor="e")
            value_label.pack(side="right")
            scale = ttk.Scale(line, from_=0, to=100, orient="horizontal", variable=var,
                              style="Tick.TScale" if self.azure_loaded else "TScale",
                              command=lambda _v: self._update_scenario())
            scale.pack(side="right", fill="x", expand=True, padx=(12, 8))
            self.whatif_vars.append(var)
            self.whatif_value_labels.append(value_label)
        actions = tk.Frame(scenario, background=COLORS["card"])
        actions.pack(fill="x", pady=(12, 0))
        action_button(actions, "Reset", self._reset_scenario).pack(side="left")
        action_button(actions, "Export results", self.export_results).pack(side="right")
        label(scenario, "Hypothetical classifier sensitivity only; not a causal or clinical intervention.",
              size=9, color="dim", wraplength=330).pack(anchor="w", pady=(11, 0))

    def _fit_decision_page(self, _width: int, viewport_height: int):
        """Scale the two charts to the page's usable height, preserving labels.

        Widgets retain a legible minimum size. The existing page scrollbar
        remains available on short displays or with large Windows text scaling.
        """
        contribution_height = max(195, min(255, int(viewport_height * 0.35 - 25)))
        probability_height = max(145, min(215, int(viewport_height * 0.25)))
        if int(self.contribution_chart.cget("height")) != contribution_height:
            self.contribution_chart.configure(height=contribution_height)
        if int(self.scenario_chart.cget("height")) != probability_height:
            self.scenario_chart.configure(height=probability_height)

    def choose_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self, title="Select fundus image",
                                          filetypes=[("Fundus photographs", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                                                     ("All files", "*.*")])
        if not path:
            return
        try:
            data = Path(path).read_bytes()
            if len(data) > 50 * 1024 * 1024:
                raise ValueError("Please choose an image smaller than 50 MB.")
            image = decode_fundus(data)
        except Exception as error:
            messagebox.showerror("Image import failed", str(error), parent=self)
            return
        self.image_bytes = data
        self.image = image
        self.image_name = Path(path).name
        self.image_info.set(f"{self.image_name} · {image.width} × {image.height}")
        self._clear_results()
        self._draw_image(self.overview_canvas)
        self._draw_image(self.evidence_canvas, evidence=True)
        self._refresh_buttons()
        self.status_var.set(f"Image loaded: {self.image_name}")

    def autoload_models(self):
        """Load recognised checkpoints beside app.py without a file dialog."""
        if self.busy:
            return
        for variable in self.model_vars.values():
            variable.set("Scanning models/ …")
        folder = Path(__file__).resolve().parent / "models"
        self._start_worker("autoload", self._load_models_from_folder, folder)

    def choose_model_checkpoint(self, kind: str):
        """Load one model from a file chosen by the user, even without models/."""
        if self.busy:
            return
        title = "black-box" if kind == "blackbox" else "Concept Bottleneck Model"
        path = filedialog.askopenfilename(
            parent=self,
            title=f"Choose {title} checkpoint",
            filetypes=[("PyTorch checkpoints", "*.pt *.pth"), ("All files", "*.*")],
        )
        if path:
            self._start_worker(f"manual_{kind}", self.engine.load, kind, path)

    def download_model_weights(self):
        """Offer an explicit download of missing public research weights."""
        if self.busy:
            return
        accepted = messagebox.askyesno(
            "Download research model weights",
            "Download missing RetinaConcept-Net model checkpoints from Google Drive?\n\n"
            "The total download size is 448 MB. The files will be saved in the local "
            "models folder",
            parent=self,
        )
        if accepted:
            folder = Path(__file__).resolve().parent / "models"
            self._start_worker("download", download_models, folder)

    @staticmethod
    def _load_models_from_folder(folder: Path):
        """Work off the UI thread; errors in one checkpoint don't block another."""
        paths, problems = discover_checkpoints(folder)
        engine = RuntimeModels()
        loaded = {}
        for kind in ("blackbox", "cbm"):
            if kind not in paths:
                continue
            try:
                loaded[kind] = engine.load(kind, str(paths[kind]))
            except Exception as error:
                problems[kind] = f"Load failed: {str(error).splitlines()[0][:100]}"
        return {"engine": engine, "loaded": loaded, "problems": problems}

    def run_analysis(self):
        if self.busy or not self.image_bytes or not (self.engine.cbm or self.engine.blackbox):
            return
        self._start_worker("analyse", self.engine.analyse, self.image_bytes)

    def _start_worker(self, task, fn, *args):
        self.busy = True
        self._refresh_buttons()
        self.progress.pack(side="right", padx=24, pady=10)
        self.progress.start(10)
        self.status_var.set("Loading checkpoints from models/ …" if task == "autoload"
                            else "Downloading missing model weights from Google Drive…" if task == "download"
                            else "Loading selected model checkpoint…" if task.startswith("manual_")
                            else "Running model inference…")

        def worker():
            try:
                result = fn(*args)
                self.pending.put((task, True, result))
            except Exception as error:
                self.pending.put((task, False, str(error)))

        threading.Thread(target=worker, daemon=True, name=f"RetinaConcept-Net-{task}").start()

    def _poll_workers(self):
        try:
            while True:
                task, success, payload = self.pending.get_nowait()
                self.busy = False
                self.progress.stop()
                self.progress.pack_forget()
                if not success:
                    self.status_var.set(f"{task.capitalize()} failed")
                    messagebox.showerror("RetinaConcept-Net", str(payload), parent=self)
                elif task == "autoload":
                    self.engine = payload["engine"]
                    for kind in ("blackbox", "cbm"):
                        if kind in payload["loaded"]:
                            info = payload["loaded"][kind]
                            self.model_vars[kind].set(f"✓ {info['name']} · {info['device'].upper()}")
                        else:
                            self.model_vars[kind].set(payload["problems"].get(kind, "Not loaded"))
                    self._clear_results()
                    count = len(payload["loaded"])
                    self.status_var.set(
                        f"{count}/2 models loaded automatically from models/"
                        if count else "No models found — use Download weights or Load…"
                    )
                    if not self._download_offered:
                        self._download_offered = True
                        if count == 0 and all(
                            "Not found" in payload["problems"].get(kind, "")
                            for kind in ("blackbox", "cbm")
                        ):
                            self.after(250, self.download_model_weights)
                elif task == "download":
                    if payload["errors"]:
                        issues = "\n".join(
                            f"{kind}: {message}" for kind, message in payload["errors"].items()
                        )
                        messagebox.showwarning("Model download", issues, parent=self)
                    if payload["downloaded"]:
                        self.status_var.set(
                            f"Downloaded {len(payload['downloaded'])} checkpoint(s) — loading models…"
                        )
                        self.after_idle(self.autoload_models)
                    elif not payload["errors"]:
                        self.status_var.set("Models already exist — no download needed")
                elif task.startswith("manual_"):
                    kind = task.split("_", 1)[1]
                    self.model_vars[kind].set(f"✓ {payload['name']} · {payload['device'].upper()}")
                    self._clear_results()
                    self.status_var.set(f"{payload['name']} loaded — ready for retinal analysis")
                elif task == "analyse":
                    self.results = payload
                    self._refresh_results()
                    self.status_var.set("Inference complete · results retained locally in memory")
                self._refresh_buttons()
        except queue.Empty:
            pass
        self.after(80, self._poll_workers)

    def _refresh_buttons(self):
        for widget in (self.open_button, self.reload_button, self.download_button,
                       *self.model_load_buttons.values()):
            widget.configure(state="disabled" if self.busy else "normal")
        can_run = not self.busy and self.image_bytes is not None and (
            self.engine.blackbox is not None or self.engine.cbm is not None)
        self.run_button.configure(
            state="normal" if can_run else "disabled",
            style="Accent.TButton" if can_run else "TButton",
        )

    def _clear_results(self):
        self.results = {}
        self.overview_chart.show(None)
        self.contribution_chart.show(None)
        self.scenario_chart.show(None)
        self.grade_summary.set("Awaiting model inference")
        self.scenario_title.set("Awaiting CBM inference")
        self.contribution_caption.set("Awaiting CBM inference")
        self.evidence_info.set("Run the Concept Bottleneck Model to inspect evidence.")
        for score in self.concept_value_labels:
            score.configure(text="—")
        self._set_explanation("Run CBM inference to generate an explanation based on the trained grading weights.")
        self._draw_image(self.evidence_canvas, evidence=True)

    def _refresh_results(self):
        chosen = "cbm" if "cbm" in self.results else "blackbox"
        probabilities = self.results[chosen]["grade_probs"]
        grade_index = int(np.argmax(probabilities))
        if "cbm" in self.results and "blackbox" in self.results:
            blackbox = self.results["blackbox"]["grade_probs"]
            other_index = int(np.argmax(blackbox))
            self.grade_summary.set(f"CBM: {GRADE_NAMES[grade_index]}   |   Black-box: {GRADE_NAMES[other_index]}")
            self.chart_title.set("CBM: cyan/blue  ·  Black-box: purple")
            self.overview_chart.show(probabilities, comparison=blackbox)
        else:
            self.grade_summary.set(f"{GRADE_NAMES[grade_index]}  ·  {probabilities[grade_index] * 100:.1f}%")
            self.chart_title.set("Concept Bottleneck Model" if chosen == "cbm" else "Black-box model")
            self.overview_chart.show(probabilities)
        self._refresh_evidence()
        self._reset_scenario()

    def _refresh_evidence(self):
        if "cbm" in self.results:
            index = self.selected_concept.get()
            scores = self.results["cbm"]["concept_probs"]
            for row, score in enumerate(self.concept_value_labels):
                score.configure(text=f"{float(scores[row]) * 100:.1f}%")
            self.evidence_info.set(
                f"{CONCEPT_NAMES[index]}: model concept score {float(scores[index]):.3f}. "
                "The heatmap is a coarse model evidence map, not a validated lesion outline."
            )
        self._draw_image(self.evidence_canvas, evidence=True)

    def _on_overlay_control_change(self, _value=None):
        self.opacity_text.set(f"{self.opacity.get():.0f}%")
        self.threshold_text.set(f"{self.threshold.get():.0f}%")
        self._draw_image(self.evidence_canvas, evidence=True)

    def _draw_image(self, canvas, *, evidence=False):
        canvas.delete("all")
        width = max(160, canvas.winfo_width() - 26)
        height = max(150, canvas.winfo_height() - 26)
        if self.image is None:
            canvas.create_text((width + 26) / 2, (height + 26) / 2,
                               text="○\n\nYour fundus photograph will appear here",
                               fill=COLORS["dim"], font=("Segoe UI", 12), justify="center")
            return

        # Work at display resolution to avoid allocating large arrays for
        # high-resolution fundus photographs during slider updates.
        rendered = self.image.copy()
        rendered.thumbnail((width, height), Image.Resampling.LANCZOS)
        if evidence and "cbm" in self.results and self.opacity.get() > 0:
            raw_map = self.results["cbm"]["concept_maps"][self.selected_concept.get()]
            # The model returns a coarse [H/8,W/8] sigmoid activation map.
            map_image = Image.fromarray(np.uint8(np.clip(raw_map, 0, 1) * 255), mode="L")
            map_image = map_image.resize(rendered.size, Image.Resampling.BILINEAR)
            strength = np.asarray(map_image, dtype=np.float32) / 255.0
            threshold = self.threshold.get() / 100.0
            strength = np.clip((strength - threshold) / max(1e-6, 1 - threshold), 0, 1)
            # Cyan → amber → red colour scale, blended per pixel with the original.
            r = np.clip(2.2 * strength, 0, 1)
            g = np.clip(1.2 - abs(strength - 0.45) * 2.0, 0, 1)
            b = np.clip(1.3 - 2.1 * strength, 0, 1)
            heat = np.uint8(np.stack((r, g, b), axis=-1) * 255)
            opacity = np.clip(strength * (self.opacity.get() / 100.0), 0, 1)[..., None]
            original = np.asarray(rendered, dtype=np.float32)
            merged = np.uint8(np.clip(original * (1 - opacity) + heat * opacity, 0, 255))
            rendered = Image.fromarray(merged, mode="RGB")

        photo = ImageTk.PhotoImage(rendered)
        self._photo_references[canvas] = photo
        x, y = canvas.winfo_width() / 2, canvas.winfo_height() / 2
        canvas.create_image(x, y, image=photo, anchor="center")
        if evidence and "cbm" in self.results:
            canvas.create_text(16, 17, text=f"{CONCEPT_CODES[self.selected_concept.get()]} · EVIDENCE",
                               fill=COLORS["teal"], anchor="w", font=("Segoe UI", 10, "bold"))

    def _reset_scenario(self):
        if "cbm" not in self.results:
            return
        concept_probs = self.results["cbm"]["concept_probs"]
        self._updating_sliders = True
        for var, value in zip(self.whatif_vars, concept_probs):
            var.set(float(value) * 100.0)
        self._updating_sliders = False
        self._update_scenario()

    def _update_scenario(self):
        if self._updating_sliders:
            return
        for value, variable in zip(self.whatif_value_labels, self.whatif_vars):
            value.configure(text=f"{variable.get():.0f}%")
        if "cbm" not in self.results:
            return
        cbm = self.results["cbm"]
        altered = np.array([var.get() / 100.0 for var in self.whatif_vars], dtype=float)
        probabilities = scenario_probabilities(cbm["weights"], cbm["bias"], altered)
        grade = int(np.argmax(probabilities))
        self.scenario_chart.show(probabilities)
        self.scenario_title.set(f"Hypothetical: {GRADE_NAMES[grade]} · {probabilities[grade] * 100:.1f}%")

        original = cbm["grade_probs"]
        original_grade = int(np.argmax(original))
        runner = int(np.argsort(original)[-2])
        contrib, baseline = margin_contributions(cbm["weights"], cbm["bias"],
                                                cbm["concept_probs"], original_grade, runner)
        self.contribution_chart.show(contrib)
        self.contribution_caption.set(
            f"{GRADE_NAMES[original_grade]} versus {GRADE_NAMES[runner]} · "
            f"baseline bias {baseline:+.3f}"
        )
        margin = baseline + float(np.sum(contrib))
        supports = np.argsort(-contrib)
        opposes = np.argsort(contrib)
        positive = [i for i in supports if contrib[i] > 0][:3]
        negative = [i for i in opposes if contrib[i] < 0][:3]
        lines = [
            f"The original CBM prediction was {GRADE_NAMES[original_grade]} "
            f"({original[original_grade] * 100:.1f}% model score).",
            f"Its logit margin over {GRADE_NAMES[runner]} was {margin:+.3f}; "
            f"this equals a bias difference of {baseline:+.3f} plus seven concept contributions.",
        ]
        if positive:
            lines.append("Strongest supporting concepts: " + ", ".join(
                f"{CONCEPT_CODES[i]} ({contrib[i]:+.3f})" for i in positive) + ".")
        if negative:
            lines.append("Concepts opposing this contrast: " + ", ".join(
                f"{CONCEPT_CODES[i]} ({contrib[i]:+.3f})" for i in negative) + ".")
        lines.append("\nThe sliders change only the trained grading head's inputs. "
                     "These computations describe model sensitivity; they do not establish causality or clinical effects.")
        self._set_explanation("\n\n".join(lines))

    def _set_explanation(self, text):
        self.explanation.configure(state="normal")
        self.explanation.delete("1.0", "end")
        self.explanation.insert("end", text)
        self.explanation.configure(state="disabled")

    def export_results(self):
        if not self.results:
            messagebox.showinfo("Export results", "Run inference before exporting results.", parent=self)
            return
        destination = filedialog.asksaveasfilename(parent=self, title="Export model outputs",
                                                    defaultextension=".json",
                                                    filetypes=[("JSON", "*.json")])
        if not destination:
            return
        payload = {"application": "RetinaConcept-Net", "research_use_only": True,
                   "image_filename": self.image_name, "grade_labels": list(GRADE_NAMES),
                   "model_outputs": {}}
        for kind, result in self.results.items():
            item = {"grade_probabilities": result["grade_probs"].tolist()}
            if kind == "cbm":
                item["concept_labels"] = list(CONCEPT_NAMES)
                item["concept_probabilities"] = result["concept_probs"].tolist()
                item["classifier_weights"] = result["weights"].tolist()
                item["classifier_bias"] = result["bias"].tolist()
            payload["model_outputs"][kind] = item
        try:
            Path(destination).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as error:
            messagebox.showerror("Export failed", str(error), parent=self)
            return
        self.status_var.set(f"Results exported to {Path(destination).name}")


if __name__ == "__main__":
    RetinaXAI().mainloop()
