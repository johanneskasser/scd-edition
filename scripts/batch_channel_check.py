"""
batch_channel_check.py
──────────────────────
Interactive batch channel inspection tool.

Opens a GUI (identical to the main app's channel rejection view) for each EMG
file in turn.  You toggle bad channels, then click "Save & Next File" to
advance.  Results are written to a JSON file after every file so no progress
is lost if you close mid-session.

The saved JSON can then be passed to batch_decompose.py via --rejections-file.

Usage
─────
    python scripts/batch_channel_check.py \\
        --config  path/to/session.yaml \\
        --layout  path/to/data_layout.yaml \\
        --files   data/*.otb+ \\
        --output  results/channel_rejections.json

Re-running the command lets you resume: already-inspected files are skipped
(or re-inspected if you answer 'y' at the prompt).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


# ── colours (mirror the main app) ─────────────────────────────────────────────

COLORS = {
    'background':       '#1e1e1e',
    'background_light': '#2d2d2d',
    'background_hover': '#3d3d3d',
    'foreground':       '#d4d4d4',
    'text_muted':       '#858585',
    'info':             '#4ec9b0',
    'info_light':       '#9cdcfe',
    'error':            '#f44747',
    'success':          '#4caf50',
    'border':           '#404040',
}


# ── GUI ────────────────────────────────────────────────────────────────────────

def _run_channel_check_gui(
    file_paths: List[Path],
    layout: dict,
    grid_configs: dict,
    output_path: Path,
    existing_rejections: dict,
) -> dict:
    """
    Show the channel rejection GUI for each file in sequence.
    Saves progress to *output_path* after every file.
    Returns the complete {filename: {grid: mask}} dict.
    """
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
    from PyQt5.QtCore import QEventLoop, QTimer
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.widgets import Button
    from scd_app.io.data_loader import load_field

    app = QApplication.instance() or QApplication(sys.argv)

    rejections: dict = dict(existing_rejections)

    # ── main window ───────────────────────────────────────────────────────────
    class ChannelCheckWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Batch Channel Inspection")
            self.resize(1600, 1000)
            self.setStyleSheet(f"background-color: {COLORS['background']};")

            central = QWidget()
            self.setCentralWidget(central)
            vbox = QVBoxLayout(central)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(0)

            self.figure = Figure(facecolor=COLORS['background'])
            self.canvas = FigureCanvas(self.figure)
            vbox.addWidget(self.canvas, 1)

            self.show()

    win = ChannelCheckWindow()

    # ── per-file loop ─────────────────────────────────────────────────────────
    for file_idx, file_path in enumerate(file_paths):
        fname = file_path.name
        n_files = len(file_paths)
        app.processEvents()

        try:
            emg = load_field(file_path, layout, "emg")   # (samples, channels)
        except Exception as exc:
            print(f"  ERROR loading {fname}: {exc} — skipping.")
            continue

        grid_list = list(grid_configs.items())

        # Initialise masks from existing JSON or zeros
        masks: List[np.ndarray] = []
        saved_file = rejections.get(fname, {})
        for port_name, cfg in grid_list:
            n = len(cfg["channels"])
            prev = saved_file.get(port_name)
            masks.append(
                np.array(prev, dtype=int) if prev is not None
                else np.zeros(n, dtype=int)
            )

        nav = {'current': 0, 'cids': [], 'buttons': []}
        event_loop = QEventLoop()

        # ── draw one grid ─────────────────────────────────────────────────────
        def draw_grid(grid_idx,
                      _emg=emg, _grid_list=grid_list, _masks=masks,
                      _nav=nav, _win=win, _loop=event_loop,
                      _file_idx=file_idx, _fname=fname, _n_files=n_files):

            _win.figure.clf()
            port_name, cfg = _grid_list[grid_idx]
            channels   = cfg['channels']
            n_channels = len(channels)
            mask       = _masks[grid_idx]
            is_first   = grid_idx == 0
            is_last    = grid_idx == len(_grid_list) - 1

            ax = _win.figure.add_axes([0.03, 0.08, 0.94, 0.90])
            ax.set_facecolor(COLORS['background'])

            _win.figure.text(
                0.5, 0.985,
                f"[{_file_idx + 1}/{_n_files}]  {_fname}   —   "
                f"{port_name}  ({grid_idx + 1}/{len(_grid_list)})",
                ha="center", va="center",
                fontsize=12, weight='bold', color=COLORS['foreground'],
            )
            _win.figure.text(
                0.5, 0.055,
                "Click = Toggle channel  |  Scroll = Zoom  |  Right-drag = Pan  |  R = Reset",
                ha="center", va="center",
                fontsize=9, color=COLORS['info'],
            )

            # Down-sample for display speed
            grid_data = _emg[:, channels].numpy()
            step      = max(1, grid_data.shape[0] // 4000)
            disp      = grid_data[::step, :]
            max_len   = disp.shape[0]

            # Normalise using active channels only
            active_idx = np.where(mask == 0)[0]
            ref        = disp[:, active_idx] if len(active_idx) > 0 else disp
            active_std = float(np.std(ref))
            separation = active_std * 15 if active_std > 0 else 1.0

            for ch in range(n_channels):
                y = ch * separation
                if mask[ch]:
                    ax.plot([0, max_len], [y, y],
                            color=COLORS['error'], alpha=0.25,
                            linewidth=0.8, linestyle='--')
                else:
                    ax.plot(disp[:, ch] + y,
                            color='#4a9eff', alpha=0.8, linewidth=1.0)

            for ch in range(n_channels):
                ax.text(
                    -max_len * 0.01, ch * separation, str(ch),
                    color=COLORS['error'] if mask[ch] else COLORS['text_muted'],
                    fontsize=7, ha='right', va='center',
                )

            ax.set_xlim(0, max_len)
            ax.set_ylim(-separation, n_channels * separation)
            ax.axis('off')

            n_rej = int(np.sum(mask))
            _win.figure.text(
                0.5, 0.032,
                f"{n_rej} channel{'s' if n_rej != 1 else ''} rejected" if n_rej else "",
                ha="center", va="center",
                fontsize=9, color=COLORS['error'],
            )

            # ── buttons ───────────────────────────────────────────────────────
            prev_ax    = _win.figure.add_axes([0.05, 0.005, 0.12, 0.040])
            next_ax    = _win.figure.add_axes([0.79, 0.005, 0.16, 0.040])
            confirm_ax = _win.figure.add_axes([0.40, 0.005, 0.18, 0.040])

            prev_btn = Button(
                prev_ax, "← Previous",
                color=COLORS['background_light'],
                hovercolor=COLORS['background_hover'],
            )
            prev_btn.label.set_color(
                COLORS['foreground'] if not is_first else COLORS['text_muted'])
            prev_btn.label.set_fontsize(10)

            if not is_last:
                next_label = "Next →"
                next_color = COLORS['info']
            elif _file_idx < _n_files - 1:
                next_label = "Save & Next File →"
                next_color = COLORS['success']
            else:
                next_label = "Save & Done ✓"
                next_color = COLORS['success']

            next_btn = Button(next_ax, next_label,
                              color=next_color,
                              hovercolor=COLORS['background_hover'])
            next_btn.label.set_color('white')
            next_btn.label.set_fontsize(10)
            next_btn.label.set_weight('bold')

            confirm_btn = Button(confirm_ax, "CONFIRM ALL",
                                 color='#555555',
                                 hovercolor=COLORS['success'])
            confirm_btn.label.set_color(COLORS['foreground'])
            confirm_btn.label.set_fontsize(9)

            # ── interaction state ──────────────────────────────────────────────
            state = {
                'press_event':      None,
                'last_scroll_time': 0.0,
                'panning':          False,
                'pan_start':        None,
                'pan_xlim':         None,
                'pan_ylim':         None,
            }

            def on_press(ev,
                         _ax=ax, _state=state):
                if ev.inaxes != _ax:
                    return
                if ev.button == 1:
                    _state['press_event'] = ev
                elif ev.button == 3:
                    _state['panning']   = True
                    _state['pan_start'] = (ev.x, ev.y)
                    _state['pan_xlim']  = _ax.get_xlim()
                    _state['pan_ylim']  = _ax.get_ylim()

            def on_motion(ev,
                          _ax=ax, _state=state, _canvas=_win.canvas):
                if not _state['panning'] or ev.inaxes != _ax:
                    return
                dx   = ev.x - _state['pan_start'][0]
                dy   = ev.y - _state['pan_start'][1]
                xlim = _state['pan_xlim']
                ylim = _state['pan_ylim']
                bbox = _ax.get_window_extent()
                data_dx = -(dx / bbox.width)  * (xlim[1] - xlim[0])
                data_dy = -(dy / bbox.height) * (ylim[1] - ylim[0])
                _ax.set_xlim(xlim[0] + data_dx, xlim[1] + data_dx)
                _ax.set_ylim(ylim[0] + data_dy, ylim[1] + data_dy)
                _canvas.draw_idle()

            def on_release(ev,
                           _ax=ax, _state=state, _mask=mask,
                           _n_ch=n_channels, _sep=separation,
                           _nav=_nav, _grid_idx=grid_idx):
                if ev.button == 3:
                    _state['panning'] = False
                    return
                if ev.button != 1 or _state['press_event'] is None:
                    return
                press = _state['press_event']
                _state['press_event'] = None
                elapsed_ms = (time.time() - _state['last_scroll_time']) * 1000
                if elapsed_ms < 300:
                    return
                if abs(ev.x - press.x) > 5 or abs(ev.y - press.y) > 5:
                    return
                if ev.inaxes != _ax or ev.ydata is None:
                    return
                closest, min_d = None, float('inf')
                for ch in range(_n_ch):
                    d = abs(ev.ydata - ch * _sep)
                    if d < _sep * 0.6 and d < min_d:
                        min_d, closest = d, ch
                if closest is None:
                    return
                _mask[closest] = 1 - _mask[closest]
                disconnect()
                draw_grid(_grid_idx)

            def on_scroll(ev,
                          _ax=ax, _state=state, _canvas=_win.canvas):
                if ev.inaxes != _ax:
                    return
                _state['last_scroll_time'] = time.time()
                scale = 0.85 if ev.button == 'up' else 1.18
                xlim, ylim = _ax.get_xlim(), _ax.get_ylim()
                _ax.set_xlim(ev.xdata - (ev.xdata - xlim[0]) * scale,
                             ev.xdata + (xlim[1] - ev.xdata) * scale)
                _ax.set_ylim(ev.ydata - (ev.ydata - ylim[0]) * scale,
                             ev.ydata + (ylim[1] - ev.ydata) * scale)
                _canvas.draw_idle()

            def on_key(ev,
                       _ax=ax, _canvas=_win.canvas,
                       _max_len=max_len, _sep=separation,
                       _total_h=n_channels * separation):
                if ev.key in ('r', 'R'):
                    _ax.set_xlim(0, _max_len)
                    _ax.set_ylim(-_sep * 0.5, _total_h + _sep * 0.5)
                    _canvas.draw_idle()

            def go_prev(ev, _nav=_nav, _is_first=is_first):
                if not _is_first:
                    disconnect()
                    _nav['current'] -= 1
                    draw_grid(_nav['current'])

            def go_next(ev, _nav=_nav, _is_last=is_last, _loop=_loop):
                disconnect()
                if not _is_last:
                    _nav['current'] += 1
                    draw_grid(_nav['current'])
                else:
                    QTimer.singleShot(50, _loop.quit)

            def on_confirm(ev, _loop=_loop):
                disconnect()
                QTimer.singleShot(50, _loop.quit)

            cids = [
                _win.canvas.mpl_connect('button_press_event',   on_press),
                _win.canvas.mpl_connect('button_release_event', on_release),
                _win.canvas.mpl_connect('motion_notify_event',  on_motion),
                _win.canvas.mpl_connect('scroll_event',         on_scroll),
                _win.canvas.mpl_connect('key_press_event',      on_key),
            ]
            prev_btn.on_clicked(go_prev)
            next_btn.on_clicked(go_next)
            confirm_btn.on_clicked(on_confirm)

            _nav['cids']    = cids
            _nav['buttons'] = [prev_btn, next_btn, confirm_btn]
            _win.canvas.draw()

        def disconnect(_nav=nav, _win=win):
            for cid in _nav.get('cids', []):
                _win.canvas.mpl_disconnect(cid)
            for btn in _nav.get('buttons', []):
                btn.disconnect_events()
                try:
                    btn.ax.remove()
                except Exception:
                    pass
            _nav['buttons'] = []

        # Launch event loop for this file
        draw_grid(0)
        event_loop.exec_()

        # ── save masks for this file ──────────────────────────────────────────
        rejections[fname] = {
            port_name: masks[i].tolist()
            for i, (port_name, _) in enumerate(grid_list)
        }
        _save_json(output_path, rejections)

        n_rej = sum(int(np.sum(m)) for m in masks)
        print(f"  [{fname}] saved — {n_rej} channel(s) rejected "
              f"across {len(grid_list)} grid(s)")

    win.close()
    return rejections


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _setup_from_channel_config(json_path: Path):
    """
    Parse the app's native channel_config.json into (grid_configs, layout, sampling_rate).
    This is an alternative to providing a session YAML + layout YAML.
    """
    with open(json_path) as f:
        data = json.load(f)

    sampling_rate = data.get("sampling_rate", 10240)

    grid_configs: Dict[str, dict] = {}
    for g in data.get("grids", []):
        channels = list(range(g["start_chan"], g["end_chan"]))
        grid_configs[g["name"]] = {
            "channels":       channels,
            "num_channels":   len(channels),
            "electrode_type": g.get("config", ""),
            "electrode_class": "surface_grid" if g.get("type", "").lower() == "surface"
                               else "intramuscular",
        }

    # Derive the layout from the loader field (e.g. ".otb+")
    from scd_app.io.data_loader import load_layout
    loader_map = {
        ".otb+": "loader_otb+.yaml",
        ".h5":   "loader_h5.yaml",
        ".mat":  "loader_mat.yaml",
    }
    loader_key = data.get("loader", ".otb+")
    loader_file = loader_map.get(loader_key, "loader_otb+.yaml")
    # Find the built-in loader YAML from the package resources
    _here = Path(__file__).resolve().parent.parent
    loader_path = _here / "src" / "scd_app" / "resources" / "loaders_configs" / loader_file
    if not loader_path.exists():
        # Fallback: installed package resources
        import importlib.resources as pkg_res
        loader_path = Path(str(pkg_res.files("scd_app") / "resources" / "loaders_configs" / loader_file))
    layout = load_layout(loader_path)

    return grid_configs, layout, sampling_rate


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Interactive batch channel inspection. "
                    "Saves per-file rejection masks to a JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Option A: app's native channel_config.json (simplest)
    ap.add_argument("--channel-config", default=None, metavar="JSON",
                    help="Path to the app's channel_config.json. "
                         "When provided, --config and --layout are not needed.")
    # Option B: session YAML + layout YAML
    ap.add_argument("--config",  default=None,
                    help="Session YAML path (not needed if --channel-config is given).")
    ap.add_argument("--layout",  default=None,
                    help="Data-layout YAML path (not needed if --channel-config is given).")
    ap.add_argument("--files",   required=True, nargs="+",
                    help="EMG file path(s), directory, or glob pattern.")
    ap.add_argument("--output",  default="channel_rejections.json",
                    help="Output JSON path (default: channel_rejections.json).")
    ap.add_argument("--ext",     default=None,
                    help="File extension when --files is a directory "
                         "(e.g. .h5, .mat, .otb+).")
    args = ap.parse_args()

    # ── load config ───────────────────────────────────────────────────────────
    if args.channel_config:
        grid_configs, layout, sampling_rate = _setup_from_channel_config(
            Path(args.channel_config))
        print(f"Channel config : {args.channel_config}")
        print(f"Fs             : {sampling_rate} Hz")
        print(f"Grids          : {list(grid_configs.keys())}")
    elif args.config and args.layout:
        from scd_app.core.config import ConfigManager
        from scd_app.io.data_loader import load_layout
        mgr    = ConfigManager()
        config = mgr.load_session(Path(args.config))
        layout = load_layout(Path(args.layout))
        sampling_rate = config.sampling_frequency
        print(f"Session : {config.name}  |  Fs: {sampling_rate} Hz")
        grid_configs: Dict[str, dict] = {}
        for port in config.ports:
            if not port.enabled:
                continue
            grid_configs[port.name] = {
                "channels":       port.electrode.channels,
                "num_channels":   len(port.electrode.channels),
                "electrode_type": port.electrode.name,
            }
    else:
        ap.error("Provide either --channel-config OR both --config and --layout.")

    # ── resolve file list ─────────────────────────────────────────────────────
    _fmt_ext = {
        "h5": ".h5", "mat": ".mat", "npy": ".npy",
        "otb": ".otb+", 
    }
    default_ext = _fmt_ext.get(layout.get("format", ""), ".h5")
    search_ext  = args.ext if args.ext else default_ext

    file_paths: List[Path] = []
    for pat in args.files:
        p = Path(pat)
        if p.is_dir():
            found = sorted(p.glob(f"*{search_ext}"))
            if not found:
                print(f"  Warning: no *{search_ext} files in {p}")
            file_paths.extend(found)
        elif "*" in pat or "?" in pat:
            file_paths.extend(sorted(Path(".").glob(pat)))
        else:
            file_paths.append(p)
    file_paths = sorted(set(file_paths))

    if not file_paths:
        print("No files matched — exiting.")
        sys.exit(1)

    print(f"\nFiles to inspect ({len(file_paths)}):")
    for p in file_paths:
        print(f"  {p}")

    output_path = Path(args.output)

    # ── resume support ────────────────────────────────────────────────────────
    existing: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"\nLoaded existing rejections from {output_path} "
              f"({len(existing)} file(s) already done).")

    already_done = [p for p in file_paths if p.name in existing]
    if already_done:
        print(f"\n{len(already_done)} file(s) already inspected:")
        for p in already_done:
            print(f"  {p.name}")
        ans = input("Re-inspect these files? [y/N]: ").strip().lower()
        if ans != 'y':
            file_paths = [p for p in file_paths if p.name not in existing]
            if not file_paths:
                print("All files already inspected. Done.")
                sys.exit(0)
            print(f"Skipping {len(already_done)} file(s).")

    # ── run GUI ───────────────────────────────────────────────────────────────
    print(f"\nStarting inspection for {len(file_paths)} file(s)...\n")
    rejections = _run_channel_check_gui(
        file_paths         = file_paths,
        layout             = layout,
        grid_configs       = grid_configs,
        output_path        = output_path,
        existing_rejections= existing,
    )

    print(f"\nAll done. Rejections saved to: {output_path}")
    print("\nSummary:")
    for fname, grids in rejections.items():
        total = sum(sum(m) for m in grids.values())
        print(f"  {fname}: {total} channel(s) rejected")


if __name__ == "__main__":
    _src = Path(__file__).resolve().parent.parent / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    main()
