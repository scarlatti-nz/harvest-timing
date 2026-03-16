"""
Shared publication styling for paper figures.
"""

from __future__ import annotations

import os

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


TEXT_GREY = "#4D4D4D"
LIGHT_GREY = "#BDBDBD"
TEXT_BLACK = "#000000"

ACTION_COLORS = {
    "Hold": "#8491B4",
    "Harvest": "#E64B35",
    "Switch": "#4DBBD5",
}
ACTION_COLOR_LIST = [
    ACTION_COLORS["Hold"],
    ACTION_COLORS["Harvest"],
    ACTION_COLORS["Switch"],
]

REGIME_COLORS = {
    "averaging": "#4DBBD5",
    "permanent": "#E64B35",
    "stock_change": "#8491B4",
}
CARBON_PRICE_COLOR = "#3C5488"
TIMBER_PRICE_COLOR = "#00A087"
REFERENCE_COLOR = "#7E6148"


def register_arial_fonts() -> None:
    arial_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/ariali.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbi.ttf",
    ]
    for font_path in arial_candidates:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)


def configure_paper_style() -> None:
    register_arial_fonts()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            # Keep Arial as the house font while rendering CO$_2$-style subscripts
            # via mathtext instead of relying on Arial's missing U+2082 glyph.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.default": "regular",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.edgecolor": TEXT_BLACK,
            "axes.labelcolor": TEXT_BLACK,
            "axes.titlecolor": TEXT_BLACK,
            "xtick.color": TEXT_BLACK,
            "ytick.color": TEXT_BLACK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=7, width=0.6, length=2.5, colors=TEXT_BLACK)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(TEXT_BLACK)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=TEXT_BLACK,
    )


def add_panel_condition(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=7,
        ha="left",
        va="top",
        color=TEXT_GREY,
    )


def add_action_legend(
    fig: plt.Figure,
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> None:
    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in ACTION_COLORS.items()
    ]
    legend_ax = fig.add_axes([left, bottom, width, height])
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        labelcolor=TEXT_BLACK,
        fontsize=8,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.6,
        borderaxespad=0.0,
    )


def save_figure(fig: plt.Figure, save_path: str) -> None:
    stem, ext = os.path.splitext(save_path)
    stem_path = stem if ext else save_path
    fig.savefig(f"{stem_path}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem_path}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
