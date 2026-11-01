"""Shared dark theme for allocator and flight-replay matplotlib views."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# Pure black canvas; chrome sits just above black so controls read as panels.
BG = "#000000"
PANEL = "#121212"
GRID = "#2a2a2a"
TEXT = "#e6e6e6"
MUTED = "#8a8a8a"
SPINE = "#333333"
ACCENT = "#5b9fd4"
ACCENT_DIM = "#2a455c"
HANDLE = "#e6e6e6"
BUTTON = "#1a1a1a"
BUTTON_HOVER = "#2a2a2a"
BUTTON_ACTIVE = "#1f6feb"
BUTTON_ACTIVE_HOVER = "#388bfd"
INACTIVE_LABEL = "#666666"
INACTIVE_TRACK = "#1a1a1a"
INACTIVE_FILL = "#2a2a2a"
INACTIVE_HANDLE = "#555555"

# Soft RGB panes so mplot3d doesn't wash the figure gray.
PANE_RGBA = (0.0, 0.0, 0.0, 1.0)
PANE_EDGE_RGBA = (0.18, 0.18, 0.18, 0.55)
GRID_RGBA = (0.22, 0.22, 0.22, 0.35)


def apply_figure_theme(fig) -> None:
    plt.style.use("dark_background")
    fig.patch.set_facecolor(BG)


def style_3d_axes(ax, *, title: str | None = None) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for label in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        label.set_color(MUTED)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANE_RGBA)
        axis.pane.set_edgecolor(PANE_EDGE_RGBA)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = GRID_RGBA
        axis._axinfo["grid"]["linewidth"] = 0.6
        axis._axinfo["grid"]["linestyle"] = "-"
    if title is not None:
        ax.set_title(title, color=TEXT, fontsize=11, pad=8)


def style_widget_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(SPINE)


def slider_kwargs(*, active: bool = True) -> dict:
    if active:
        return {
            "initcolor": ACCENT,
            "track_color": SPINE,
            "facecolor": ACCENT,
            "handle_style": {
                "facecolor": HANDLE,
                "edgecolor": SPINE,
            },
        }
    return {
        "initcolor": INACTIVE_FILL,
        "track_color": INACTIVE_TRACK,
        "facecolor": INACTIVE_FILL,
        "handle_style": {
            "facecolor": INACTIVE_HANDLE,
            "edgecolor": INACTIVE_TRACK,
        },
    }


def style_slider(slider: Slider, *, active: bool = True) -> None:
    label = TEXT if active else INACTIVE_LABEL
    track = SPINE if active else INACTIVE_TRACK
    fill = ACCENT if active else INACTIVE_FILL
    handle = HANDLE if active else INACTIVE_HANDLE
    slider.label.set_color(label)
    slider.valtext.set_color(label)
    slider.ax.set_facecolor(PANEL)
    for spine in slider.ax.spines.values():
        spine.set_color(SPINE)
    if hasattr(slider, "track"):
        slider.track.set_facecolor(track)
        if hasattr(slider.track, "set_color"):
            slider.track.set_color(track)
    if hasattr(slider, "poly"):
        slider.poly.set_facecolor(fill)
        if hasattr(slider.poly, "set_color"):
            slider.poly.set_color(fill)
    if hasattr(slider, "hline"):
        slider.hline.set_color(track)
    if hasattr(slider, "_handle"):
        slider._handle.set_color(handle)
        if hasattr(slider._handle, "set_markeredgecolor"):
            slider._handle.set_markeredgecolor(track)


def style_button(button: Button, *, active: bool = False) -> None:
    face = BUTTON_ACTIVE if active else BUTTON
    hover = BUTTON_ACTIVE_HOVER if active else BUTTON_HOVER
    button.color = face
    button.hovercolor = hover
    button.ax.set_facecolor(face)
    button.label.set_color(TEXT)
    button.label.set_fontweight("bold")
    for spine in button.ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(1.0)
