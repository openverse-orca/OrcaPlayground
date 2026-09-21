"""Read-only charts for the dexterous-hand provider example.

Matplotlib is imported only when a dashboard is constructed. Headless mode uses
an Agg canvas directly, without importing pyplot or requiring a display server.
"""

from collections import deque
import math
from pathlib import Path


TOUCH_IDS = tuple(f"touch_f{finger}" for finger in range(1, 6))
RANGE_IDS = tuple(f"range_f{finger}" for finger in range(1, 6))
FINGER_LABELS = tuple(f"Finger {finger}" for finger in range(1, 6))
FINGER_COLORS = ("#2563eb", "#0d9488", "#e09a19", "#c25281", "#8660c4")


class _DashboardWindow:
    """Shared figure ownership and event handling; no simulation dependencies."""

    def __init__(self, *, interactive: bool):
        try:
            import numpy as np
            from matplotlib.figure import Figure
        except ImportError as exc:
            raise RuntimeError("Sensor charts require matplotlib and numpy") from exc

        self._np = np
        self._interactive = bool(interactive)
        self._closed = False
        self._pyplot = None
        if self._interactive:
            import matplotlib.pyplot as plt

            self._pyplot = plt
            self.figure = plt.figure(figsize=(13.8, 7.4), facecolor="#f5f7fb")
            if self.figure.canvas.required_interactive_framework is None:
                plt.close(self.figure)
                raise RuntimeError(
                    "Interactive charts require a desktop Matplotlib backend "
                    "and a display; use headless PNG output on a server"
                )
            self.figure.canvas.manager.set_window_title("OrcaGym | Sensor Provider Dashboard")
            self.figure.canvas.mpl_connect("close_event", self._on_close)
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            self.figure = Figure(figsize=(13.8, 7.4), facecolor="#f5f7fb")
            FigureCanvasAgg(self.figure)

    @staticmethod
    def _style_axis(axis):
        axis.set_facecolor("#ffffff")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#d5dfeb")
        axis.tick_params(colors="#60718b", labelsize=8)
        axis.xaxis.label.set_color("#60718b")
        axis.yaxis.label.set_color("#60718b")
        axis.title.set_color("#182b45")

    def _on_close(self, _event):
        self._closed = True

    def is_open(self) -> bool:
        """Whether this chart is available (closing a GUI window stops its run)."""
        return not self._closed and (
            not self._interactive or self._pyplot.fignum_exists(self.figure.number)
        )

    def poll(self, interval: float = 0.01) -> None:
        """Service GUI events; a noninteractive chart never blocks or sleeps."""
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("poll interval must be finite and nonnegative")
        if self._interactive and self.is_open():
            self.figure.canvas.flush_events()
            if interval > 0:
                self.figure.canvas.start_event_loop(interval)

    def save(self, path) -> None:
        """Write a PNG without overwriting an existing file."""
        destination = Path(path)
        if destination.suffix.lower() != ".png":
            raise ValueError("Chart output must have a .png extension")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            self.figure.savefig(stream, format="png", dpi=140, facecolor=self.figure.get_facecolor())

    def close(self) -> None:
        """Close the window; safe to call more than once."""
        if self._interactive and self._pyplot is not None:
            self._pyplot.close(self.figure)
        self._closed = True


class SensorDashboard(_DashboardWindow):
    """Display copied provider observations; never read or advance a simulator.

    Heatmaps show the DLL's per-cell responses. Their sum is a grid response
    sum, not the norm of a resultant force. Distances are displayed in mm;
    the provider's -1 m sentinel is explicitly shown as NO HIT.
    """

    def __init__(self, *, interactive: bool = True, history_size: int = 300):
        if type(history_size) is not int or history_size < 2:
            raise ValueError("history_size must be an integer of at least 2")
        super().__init__(interactive=interactive)
        from matplotlib.colors import Normalize

        np = self._np
        self._times = deque(maxlen=history_size)
        self._totals = tuple(deque(maxlen=history_size) for _ in TOUCH_IDS)

        self.figure.suptitle(
            "DEXTEROUS HAND  /  LIVE SENSOR OUTPUTS", x=0.06, y=0.962,
            ha="left", color="#182b45", fontsize=17, fontweight="bold",
        )
        self._status = self.figure.text(
            0.06, 0.904, "WAITING FOR FIRST PROVIDER SAMPLE", fontsize=10,
            color="#60718b", va="top", linespacing=1.55,
        )
        layout = self.figure.add_gridspec(
            2, 6, width_ratios=[1, 1, 1, 1, 1, 0.065],
            height_ratios=[1, 1.16], left=0.06, right=0.94, top=0.79,
            bottom=0.13, hspace=0.61, wspace=0.42,
        )
        self._normalization = Normalize(vmin=0.0, vmax=0.001)
        self._heatmaps = []
        self._heat_status = []
        for index, label in enumerate(FINGER_LABELS):
            axis = self.figure.add_subplot(layout[0, index])
            self._style_axis(axis)
            axis.set_title(label, color=FINGER_COLORS[index], fontweight="bold", pad=10)
            axis.set_xticks([0, 1, 2, 3])
            axis.set_yticks([0, 1, 2, 3])
            axis.set_xlabel("column", fontsize=8)
            axis.set_ylabel("row" if index == 0 else "", fontsize=8)
            axis.set_xticks(np.arange(-0.5, 4, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=1.2)
            axis.tick_params(which="minor", bottom=False, left=False)
            heatmap = axis.imshow(
                np.full((4, 4), np.nan), origin="lower", cmap="YlOrRd",
                norm=self._normalization, interpolation="nearest",
            )
            self._heatmaps.append(heatmap)
            self._heat_status.append(axis.text(
                0.5, 0.5, "NOT READY", transform=axis.transAxes,
                ha="center", va="center", fontsize=9, color="#60718b",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            ))
        color_axis = self.figure.add_subplot(layout[0, 5])
        self.figure.colorbar(self._heatmaps[0], cax=color_axis, label="Cell response (N)")
        color_axis.tick_params(labelsize=8)

        self._history_axis = self.figure.add_subplot(layout[1, :3])
        self._style_axis(self._history_axis)
        self._history_axis.set_title("Touch response history", loc="left", fontweight="bold", pad=12)
        self._history_axis.set_xlabel("Source sample time (s)")
        self._history_axis.set_ylabel("Grid response sum (N)")
        self._history_axis.grid(axis="y", color="#e5ebf2", linewidth=0.8)
        self._history_axis.set_xlim(0, 1)
        self._history_axis.set_ylim(0, 0.001)
        self._lines = tuple(
            self._history_axis.plot([], [], color=color, label=f"F{index + 1}", linewidth=1.8)[0]
            for index, color in enumerate(FINGER_COLORS)
        )
        self._history_axis.legend(
            loc="upper left", ncol=5, frameon=False, fontsize=8,
            handlelength=1.1, columnspacing=1.2,
        )

        self._range_axis = self.figure.add_subplot(layout[1, 3:5])
        self._style_axis(self._range_axis)
        self._range_axis.set_title("Rangefinder output", loc="left", fontweight="bold", pad=12)
        self._range_axis.set_ylabel("Distance (mm)")
        self._range_axis.set_ylim(0, 110)
        self._range_axis.set_axisbelow(True)
        self._range_axis.grid(axis="y", color="#e5ebf2", linewidth=0.8)
        self._bars = self._range_axis.bar(np.arange(5), np.full(5, np.nan), color=FINGER_COLORS, width=0.58)
        self._range_axis.set_xticks(np.arange(5), [f"F{i}" for i in range(1, 6)])
        self._range_axis.set_xlim(-0.6, 4.6)
        self._range_labels = tuple(
            self._range_axis.text(
                index, 0.93, "NOT\nREADY", transform=self._range_axis.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5, color="#60718b",
            ) for index in range(5)
        )
        self.figure.text(
            0.06, 0.036,
            "OrcaGym env.step()  >  Orca input assembly  >  vendor library  >  observation  >  charts\n"
            "Read-only display. Grid sums are not resultant-force norms. NO HIT is not zero distance.",
            color="#60718b", fontsize=8.5, linespacing=1.6,
        )
        if self._interactive:
            self._pyplot.show(block=False)

    def update(self, observation, *, simulation_time: float, source_time: float | None) -> None:
        """Copy a complete task observation into the charts, without retaining it."""
        if not self.is_open():
            return
        if not math.isfinite(simulation_time):
            raise ValueError("simulation_time must be finite")
        if source_time is not None and not math.isfinite(source_time):
            raise ValueError("source_time must be finite or None")
        np = self._np
        validity = np.asarray(observation["provider_valid"])
        if validity.size != 1 or validity.reshape(-1)[0] not in (0, 1):
            raise ValueError("provider_valid must contain exactly one 0 or 1")
        ready = bool(validity.reshape(-1)[0])
        if not ready:
            self._clear_history()
            self._normalization.vmax = 0.001
            for heatmap, label in zip(self._heatmaps, self._heat_status):
                heatmap.set_data(np.full((4, 4), np.nan))
                label.set_text("NOT READY")
                label.set_visible(True)
            for bar, label in zip(self._bars, self._range_labels):
                bar.set_height(np.nan)
                label.set_text("NOT\nREADY")
            self._status.set_text(
                f"SIMULATION  {simulation_time:.4f} s     |     PROVIDERS NOT READY\n"
                "Reset placeholders are hidden; no measurements have been appended."
            )
        else:
            values = observation["provider_sensors"]
            grids = []
            distances = []
            for touch_name, range_name in zip(TOUCH_IDS, RANGE_IDS):
                grid = np.asarray(values[touch_name], dtype=np.float64)
                distance = np.asarray(values[range_name], dtype=np.float64)
                if grid.shape != (4, 4) or not np.isfinite(grid).all() or (grid < 0).any():
                    raise ValueError(f"{touch_name} must be a finite nonnegative 4x4 array")
                if distance.shape != (1,) or not np.isfinite(distance).all():
                    raise ValueError(f"{range_name} must be a finite shape-(1,) array")
                value = float(distance[0])
                if value < 0 and value != -1.0:
                    raise ValueError(f"{range_name} must be nonnegative or -1 for no hit")
                grids.append(grid.copy())
                distances.append(np.nan if value == -1.0 else value * 1000.0)
            timestamp = float(simulation_time if source_time is None else source_time)
            if self._times and timestamp < self._times[-1]:
                self._clear_history()
            totals = [float(grid.sum()) for grid in grids]
            if self._times and timestamp == self._times[-1]:
                for history, total in zip(self._totals, totals):
                    history[-1] = total
            else:
                self._times.append(timestamp)
                for history, total in zip(self._totals, totals):
                    history.append(total)
            maximum = max(float(grid.max()) for grid in grids)
            self._normalization.vmax = max(self._normalization.vmax, maximum * 1.08, 0.001)
            for heatmap, label, grid in zip(self._heatmaps, self._heat_status, grids):
                heatmap.set_data(grid)
                label.set_visible(False)
            for line, history in zip(self._lines, self._totals):
                line.set_data(list(self._times), list(history))
                line.set_marker("o" if len(self._times) == 1 else "")
            first, last = self._times[0], self._times[-1]
            self._history_axis.set_xlim(first, last if last > first else first + 0.01)
            self._history_axis.set_ylim(0, max(0.001, max(max(history) for history in self._totals) * 1.25))
            self._history_axis.set_xlabel(
                "Source sample time (s)" if source_time is not None else "Simulation time (s)"
            )
            for bar, label, distance in zip(self._bars, self._range_labels, distances):
                bar.set_height(distance)
                label.set_text("NO\nHIT" if math.isnan(distance) else f"{distance:.1f}")
            finite_distances = [distance for distance in distances if math.isfinite(distance)]
            self._range_axis.set_ylim(0, max(110.0, max(finite_distances, default=0.0) * 1.28))
            source_label = "unavailable" if source_time is None else f"{source_time:.4f} s"
            self._status.set_text(
                f"SIMULATION  {simulation_time:.4f} s     |     SOURCE SAMPLE  {source_label}"
                f"     |     {len(self._times)} samples shown\n"
                "Five touch grids + five rangefinders  /  shared force color scale  /  latest DLL outputs"
            )
        if self._interactive:
            self.figure.canvas.draw_idle()

    def _clear_history(self):
        self._times.clear()
        for history, line in zip(self._totals, self._lines):
            history.clear()
            line.set_data([], [])
        self._history_axis.set_xlim(0, 1)
        self._history_axis.set_ylim(0, 0.001)


class HandForceDashboard(_DashboardWindow):
    """Keep every physics sample and display the five SevenPad force outputs.

    ``record`` copies only Fn/Ft (DLL output indices 1/2), without updating any
    artists or drawing. ``refresh`` may therefore run at a lower GUI frequency
    without losing short-lived force peaks. History is bounded by sample count.

    Time is the source timestamp, or simulation time when source time is None.
    An equal timestamp replaces the latest sample without increasing counts;
    backwards time or changing clocks requires an explicit ``reset`` first.
    """

    INSTANCE_IDS = tuple(f"seven_pad_f{finger}" for finger in range(1, 6))

    def __init__(self, *, interactive: bool = True, history_size: int = 10000,
                 noise_scale: float = 0.01):
        if type(history_size) is not int or history_size < 2:
            raise ValueError("history_size must be an integer of at least 2")
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        super().__init__(interactive=interactive)
        self._samples = deque(maxlen=history_size)
        self._recorded_count = 0
        self._clock_is_source = None
        self.figure.suptitle(
            "DEXTEROUS HAND  /  FIVE-FINGER FORCE OUTPUTS", x=0.07, y=0.96,
            ha="left", color="#182b45", fontsize=17, fontweight="bold",
        )
        self._status = self.figure.text(
            0.07, 0.898, "WAITING FOR FIRST PROVIDER SAMPLE\n0 physics samples recorded",
            fontsize=10, color="#60718b", va="top", linespacing=1.5,
        )
        self._force_axis = self.figure.add_axes([0.07, 0.29, 0.88, 0.48])
        self._style_axis(self._force_axis)
        self._force_axis.set_xlabel("Source sample time (s)", fontsize=10)
        self._force_axis.set_ylabel("Force magnitude (N)", fontsize=10)
        self._force_axis.grid(axis="y", color="#e5ebf2", linewidth=0.8)
        self._force_axis.set_xlim(0, 1)
        self._force_axis.set_ylim(0, 0.1)
        self._waiting = self._force_axis.text(
            0.5, 0.5, "WAITING FOR PHYSICS SAMPLES", transform=self._force_axis.transAxes,
            ha="center", va="center", fontsize=12, color="#60718b",
        )
        self._lines = tuple(
            self._force_axis.plot(
                [], [], color=color, linestyle="-" if channel == 0 else "--",
                linewidth=1.6, label=f"F{finger + 1} {'Fn' if channel == 0 else 'Ft'}",
            )[0]
            for finger, color in enumerate(FINGER_COLORS) for channel in range(2)
        )
        self._latest_labels = []
        for finger, color in enumerate(FINGER_COLORS):
            x = 0.07 + finger * 0.18
            self.figure.text(x, 0.198, f"FINGER {finger + 1}", color=color,
                             fontsize=10, fontweight="bold")
            self._latest_labels.append(self.figure.text(
                x, 0.17, "Fn  -- N\nFt  -- N", color=color, fontsize=10,
                va="top", linespacing=1.5,
            ))
        self.figure.text(
            0.07, 0.079,
            "Solid: Fn (normal)   |   Dashed: Ft (tangential)   |   Same color = same finger",
            color="#182b45", fontsize=9,
        )
        self.figure.text(
            0.07, 0.046,
            "Fn / Ft are norms of summed force vectors after DLL noise, not site-axis components.\n"
            f"Illustrative SevenPad DLL  |  noise_scale = {noise_scale:g} N per summed-vector component"
            "  |  Every physics sample is recorded; display refresh does not resample.",
            color="#60718b", fontsize=8.3, va="top", linespacing=1.55,
        )
        if self._interactive:
            self._pyplot.show(block=False)

    @property
    def sample_count(self) -> int:
        """Number of distinct timestamps recorded since the last reset."""
        return self._recorded_count

    @property
    def retained_count(self) -> int:
        """Number of individual physics samples currently retained."""
        return len(self._samples)

    def record(self, outputs, *, simulation_time: float, source_time: float | None) -> None:
        """Atomically copy one five-instance batch; never touch artists or draw."""
        if not math.isfinite(simulation_time) or (source_time is not None and not math.isfinite(source_time)):
            raise ValueError("Force sample timestamps must be finite")
        clock_is_source = source_time is not None
        timestamp = float(source_time if clock_is_source else simulation_time)
        if self._samples:
            if clock_is_source != self._clock_is_source:
                raise ValueError("Cannot change sample clock without reset()")
            if timestamp < self._samples[-1][0] or simulation_time < self._samples[-1][1]:
                raise ValueError("Sample time cannot move backwards; call reset() for a new episode")
        forces = []
        for name in self.INSTANCE_IDS:
            if name not in outputs:
                raise ValueError(f"Missing force output instance: {name}")
            array = self._np.asarray(outputs[name])
            if array.shape != (11,):
                raise ValueError(f"{name} must have shape (11,)")
            try:
                normal, tangent = float(array[1]), float(array[2])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} Fn/Ft must be finite nonnegative numbers") from exc
            if not math.isfinite(normal) or not math.isfinite(tangent) or normal < 0 or tangent < 0:
                raise ValueError(f"{name} Fn/Ft must be finite and nonnegative")
            forces.extend((normal, tangent))
        # Immutable Python numbers are independent of DLL/NumPy buffer reuse.
        sample = (timestamp, float(simulation_time),
                  None if source_time is None else float(source_time), tuple(forces))
        if self._samples and timestamp == self._samples[-1][0]:
            self._samples[-1] = sample
        else:
            self._samples.append(sample)
            self._recorded_count += 1
        self._clock_is_source = clock_is_source

    def refresh(self) -> None:
        """Display retained samples, without interpolation, aggregation, or thinning."""
        timestamps = [sample[0] for sample in self._samples]
        for index, line in enumerate(self._lines):
            line.set_data(timestamps, [sample[3][index] for sample in self._samples])
            line.set_marker("o" if len(self._samples) == 1 else "")
        self._waiting.set_visible(not self._samples)
        if self._samples:
            first, last = self._samples[0][0], self._samples[-1][0]
            self._force_axis.set_xlim(first, last if last > first else first + 0.01)
            peak = max(max(sample[3]) for sample in self._samples)
            self._force_axis.set_ylim(0, max(0.001, peak * 1.12))
            self._force_axis.set_xlabel("Source sample time (s)" if self._clock_is_source
                                        else "Simulation time (s)")
            _timestamp, simulation_time, source_time, forces = self._samples[-1]
            source = "unavailable" if source_time is None else f"{source_time:.4f} s"
            self._status.set_text(
                f"SIMULATION  {simulation_time:.4f} s     |     SOURCE SAMPLE  {source}\n"
                f"{self.sample_count:,} physics samples recorded     |     "
                f"{self.retained_count:,} / {self._samples.maxlen:,} retained     |     5 instances / 10 force curves"
            )
            for finger, label in enumerate(self._latest_labels):
                label.set_text(f"Fn  {forces[2 * finger]:.4f} N\nFt  {forces[2 * finger + 1]:.4f} N")
        else:
            self._force_axis.set_xlim(0, 1)
            self._force_axis.set_ylim(0, 0.1)
            self._force_axis.set_xlabel("Source sample time (s)")
            self._status.set_text("WAITING FOR FIRST PROVIDER SAMPLE\n0 physics samples recorded")
            for label in self._latest_labels:
                label.set_text("Fn  -- N\nFt  -- N")
        if self._interactive and self.is_open():
            self.figure.canvas.draw_idle()

    def reset(self) -> None:
        """Start an empty episode; the next record may use an earlier timestamp."""
        self._samples.clear()
        self._recorded_count = 0
        self._clock_is_source = None
        self.refresh()

    def save(self, path) -> None:
        """Refresh all recorded samples before writing a non-overwriting PNG."""
        self.refresh()
        super().save(path)


class ProviderDashboard(_DashboardWindow):
    """Charts for any one of the four public provider examples.

    ``values`` maps instance names to already-computed NumPy outputs. Touch
    examples use 4x4 grids; rangefinder uses one distance in metres; seven_pad
    uses [distance, normal norm, tangent norm, angle, seven capacitance values].
    History retains at most ``history_size`` displayed observations per channel.
    """

    def __init__(self, example: str, instance_ids, *, interactive: bool = True,
                 history_size: int = 300):
        if isinstance(instance_ids, (str, bytes)):
            raise ValueError("instance_ids must be a sequence of instance names")
        names = tuple(instance_ids)
        if example not in {"touch_grid", "contact_grid", "rangefinder", "seven_pad"}:
            raise ValueError("Unknown provider chart example")
        if (not 1 <= len(names) <= 5
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("instance_ids must contain one to five unique nonempty names")
        if type(history_size) is not int or history_size < 2:
            raise ValueError("history_size must be an integer of at least 2")
        super().__init__(interactive=interactive)
        from matplotlib.colors import Normalize

        self._example = example
        self._names = names
        self._times = deque(maxlen=history_size)
        channels = 2 if example == "seven_pad" else 1
        self._histories = {
            name: tuple(deque(maxlen=history_size) for _ in range(channels)) for name in names
        }
        self._plots = {}
        self._labels = {}
        self._cap_labels = {}
        self._normalization = Normalize(vmin=0.0, vmax=0.001)
        np = self._np
        self.figure.suptitle(
            f"ORCA SENSOR PROVIDERS  /  {example.replace('_', ' ').upper()}",
            x=0.06, y=0.96, ha="left", fontsize=17, fontweight="bold", color="#182b45",
        )
        self._status = self.figure.text(
            0.06, 0.9, "WAITING FOR FIRST PROVIDER SAMPLE", fontsize=10,
            color="#60718b", va="top", linespacing=1.55,
        )
        if example == "rangefinder":
            layout = self.figure.add_gridspec(
                1, 2, left=0.06, right=0.94, top=0.75, bottom=0.18, wspace=0.28,
            )
            axis = self.figure.add_subplot(layout[0, 0])
            self._style_axis(axis)
            axis.set_title("Latest rangefinder output", loc="left", fontweight="bold", pad=15)
            axis.set_ylabel("Distance (mm)")
            axis.set_ylim(0, 110)
            axis.grid(axis="y", color="#e5ebf2")
            axis.set_axisbelow(True)
            bars = axis.bar(np.arange(len(names)), np.full(len(names), np.nan),
                            color=FINGER_COLORS[:len(names)], width=0.55)
            axis.set_xticks(np.arange(len(names)), names)
            axis.set_xlim(-0.6, len(names) - 0.4)
            for index, (name, bar) in enumerate(zip(names, bars)):
                self._plots[name] = bar
                self._labels[name] = axis.text(
                    index, 0.95, "NOT READY", transform=axis.get_xaxis_transform(),
                    ha="center", va="top", color="#60718b", fontsize=10,
                )
            self._value_axis = axis
            self._history_axis = self.figure.add_subplot(layout[0, 1])
        else:
            layout = self.figure.add_gridspec(
                2, len(names), left=0.06, right=0.94, top=0.74, bottom=0.13,
                hspace=0.76, wspace=0.3, height_ratios=[1, 1.05],
            )
            for index, name in enumerate(names):
                axis = self.figure.add_subplot(layout[0, index])
                self._style_axis(axis)
                axis.set_title(name, color=FINGER_COLORS[index], fontweight="bold", pad=10)
                if example == "seven_pad":
                    axis.set_ylabel("Capacitance (arbitrary units)")
                    axis.set_ylim(0, 1.3)
                    axis.set_axisbelow(True)
                    axis.grid(axis="y", color="#e5ebf2")
                    self._plots[name] = axis.bar(np.arange(7), np.full(7, np.nan),
                                                 color=FINGER_COLORS[index], width=0.65)
                    axis.set_xticks(np.arange(7), [f"P{pad}" for pad in range(1, 8)])
                    axis.set_xlim(-0.6, 6.6)
                    self._cap_labels[name] = tuple(
                        axis.text(pad, 0, "", ha="center", va="bottom", fontsize=7.5, color="#60718b")
                        for pad in range(7)
                    )
                    self._labels[name] = axis
                else:
                    heatmap = axis.imshow(np.full((4, 4), np.nan), origin="lower", cmap="YlOrRd",
                                          norm=self._normalization, interpolation="nearest")
                    axis.set_xticks([0, 1, 2, 3])
                    axis.set_yticks([0, 1, 2, 3])
                    axis.set_xlabel("column", fontsize=8)
                    axis.set_ylabel("row", fontsize=8)
                    axis.set_xticks(np.arange(-0.5, 4, 1), minor=True)
                    axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
                    axis.grid(which="minor", color="white", linewidth=1.2)
                    axis.tick_params(which="minor", bottom=False, left=False)
                    self._plots[name] = heatmap
                    self._labels[name] = axis.text(
                        0.5, 0.5, "NOT READY", transform=axis.transAxes, ha="center", va="center",
                        color="#60718b", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                    )
                    from mpl_toolkits.axes_grid1 import make_axes_locatable

                    divider = make_axes_locatable(axis)
                    color_axis = divider.append_axes("right", size="5%", pad=0.12)
                    self.figure.colorbar(heatmap, cax=color_axis, label="Cell response (N)")
            self._history_axis = self.figure.add_subplot(layout[1, :])

        self._style_axis(self._history_axis)
        self._history_axis.set_title("Output history", loc="left", fontweight="bold", pad=12)
        self._history_axis.set_xlabel("Source sample time (s)")
        self._history_axis.set_ylabel(
            "Distance (mm)" if example == "rangefinder" else
            "Force magnitude (N)" if example == "seven_pad" else "Grid response sum (N)"
        )
        self._history_axis.grid(axis="y", color="#e5ebf2", linewidth=0.8)
        self._history_axis.set_xlim(0, 1)
        self._history_axis.set_ylim(0, 110 if example == "rangefinder" else 0.001)
        self._lines = {}
        for index, name in enumerate(names):
            self._lines[name] = tuple(
                self._history_axis.plot(
                    [], [], color=FINGER_COLORS[index], linewidth=1.8,
                    linestyle="-" if channel == 0 else "--",
                    label=(f"{name}: {'normal' if channel == 0 else 'tangent'}"
                           if channels == 2 else name),
                )[0] for channel in range(channels)
            )
        self._history_axis.legend(loc="upper left", ncol=min(len(names), 3), frameon=False, fontsize=8)
        detail = (
            "Illustrative sensor model: capacitance is not calibrated. Angle is in radians."
            if example == "seven_pad" else "NO HIT is not zero distance."
            if example == "rangefinder" else "Grid response sums are not resultant-force norms."
        )
        self.figure.text(
            0.06, 0.035,
            "Orca input assembly  >  vendor library  >  named instance outputs  >  read-only charts\n" + detail,
            color="#60718b", fontsize=8.5, linespacing=1.6,
        )
        if self._interactive:
            self._pyplot.show(block=False)

    def update(self, values, *, simulation_time: float, source_time: float | None,
               ready: bool = True) -> None:
        """Validate and copy output arrays; invalid/reset placeholders are hidden."""
        if not self.is_open():
            return
        if not math.isfinite(simulation_time) or (source_time is not None and not math.isfinite(source_time)):
            raise ValueError("Chart timestamps must be finite")
        np = self._np
        shape = {"touch_grid": (4, 4), "contact_grid": (4, 4),
                 "rangefinder": (1,), "seven_pad": (11,)}[self._example]
        copied = {}
        if ready:
            for name in self._names:
                array = np.asarray(values[name], dtype=np.float64)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"{name} must be finite with shape {shape}")
                if self._example in {"touch_grid", "contact_grid"} and (array < 0).any():
                    raise ValueError("Grid responses must be nonnegative")
                if self._example in {"rangefinder", "seven_pad"} and array[0] < 0 and array[0] != -1:
                    raise ValueError("Distance must be nonnegative or -1 for no hit")
                if self._example == "seven_pad" and (array[1:3] < 0).any():
                    raise ValueError("Force magnitudes must be nonnegative")
                copied[name] = array.copy()
        timestamp = float(simulation_time if source_time is None else source_time)
        if not ready or (self._times and timestamp < self._times[-1]):
            self._times.clear()
            self._normalization.vmax = 0.001
            for histories in self._histories.values():
                for history in histories:
                    history.clear()
        replace = bool(self._times and timestamp == self._times[-1])
        if ready and not replace:
            self._times.append(timestamp)
        for name in self._names:
            array = copied.get(name)
            label = self._labels[name]
            if self._example in {"touch_grid", "contact_grid"}:
                self._plots[name].set_data(array if ready else np.full(shape, np.nan))
                label.set_visible(not ready)
                if ready:
                    self._normalization.vmax = max(self._normalization.vmax, float(array.max()) * 1.08)
                sample = [float(array.sum())] if ready else []
            elif self._example == "rangefinder":
                distance = float(array[0]) * 1000 if ready and array[0] >= 0 else np.nan
                self._plots[name].set_height(distance)
                label.set_text("NOT READY" if not ready else "NO HIT" if math.isnan(distance)
                               else f"{distance:.1f} mm")
                sample = [distance]
            else:
                for index, bar in enumerate(self._plots[name]):
                    bar.set_height(float(array[4 + index]) if ready else np.nan)
                if ready:
                    distance = "NO HIT" if array[0] < 0 else f"{array[0] * 1000:.1f} mm"
                    label.set_title(f"{name}\nRange: {distance}  |  Angle: {array[3]:.3f} rad", fontsize=10)
                    low, high = min(0.0, float(array[4:].min())), max(0.001, float(array[4:].max()))
                    label.set_ylim(low * 1.2, high * 1.2)
                    for index, caption in enumerate(self._cap_labels[name]):
                        capacitance = float(array[4 + index])
                        caption.set_text(f"{capacitance:.4f}" if abs(capacitance) < 100 else f"{capacitance:.4g}")
                        direction = 1 if capacitance >= 0 else -1
                        caption.set_position((index, capacitance + direction * (high - low) * 0.025))
                        caption.set_verticalalignment("bottom" if capacitance >= 0 else "top")
                        caption.set_visible(True)
                else:
                    label.set_title(f"{name}\nNOT READY", fontsize=10)
                    for caption in self._cap_labels[name]:
                        caption.set_visible(False)
                sample = list(array[1:3]) if ready else []
            for history, value in zip(self._histories[name], sample):
                if ready:
                    if replace:
                        history[-1] = value
                    else:
                        history.append(value)
            for history, line in zip(self._histories[name], self._lines[name]):
                line.set_data(list(self._times), list(history))
                line.set_marker("o" if len(self._times) == 1 else "")
        if self._times:
            first, last = self._times[0], self._times[-1]
            self._history_axis.set_xlim(first, last if last > first else first + 0.01)
        else:
            self._history_axis.set_xlim(0, 1)
        finite_values = [value for histories in self._histories.values() for history in histories
                         for value in history if math.isfinite(value)]
        maximum = max(finite_values, default=0.0)
        margin = 1.4 if self._example == "seven_pad" else 1.25
        self._history_axis.set_ylim(0, max(110.0 if self._example == "rangefinder" else 0.001, maximum * margin))
        self._history_axis.set_xlabel("Source sample time (s)" if source_time is not None else "Simulation time (s)")
        if self._example == "rangefinder":
            self._value_axis.set_ylim(0, max(110.0, maximum * 1.25))
        source = "unavailable" if source_time is None else f"{source_time:.4f} s"
        status = f"{len(self._times)} samples shown" if ready else "NOT READY - placeholders hidden"
        self._status.set_text(
            f"SIMULATION  {simulation_time:.4f} s     |     SOURCE SAMPLE  {source}     |     {status}\n"
            f"{len(self._names)} independently named instances  /  latest DLL outputs"
        )
        if self._interactive:
            self.figure.canvas.draw_idle()
