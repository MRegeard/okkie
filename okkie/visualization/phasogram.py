import matplotlib.pyplot as plt
import numpy as np

__all__ = ["VisuPhasogram"]


class VisuPhasogram:
    """A class that gather useful methods to plot phasograms.

    # TODO(@MRegeard): Use self._was_called to check if the plot method was called
    in place where it is needed.
    # TDOD(@MRegeard): Support weigths in n_bkg.

    Parameters
    ----------
    events: `~gammapy.data.EventList`
        Event list.
    ax: `~matplotlib.axes.Axes`, optional
        Axes to plot the phasogram.
    on_phase: `tuple` or list of `tuple`, optional
        Phase interval(s) to plot on the phasogram.
    off_phase: `tuple` or list of `tuple`, optional
        Phase interval(s) to plot on the phasogram.
    phase_column_name: str, optional
        Name of the column containing the phase in the events table.
    offset: float, optional
        Offset to apply to the phase values.
    """

    def __init__(
        self,
        events,
        ax=None,
        on_phase=None,
        off_phase=None,
        phase_column_name="PHASE",
        offset=0,
    ):
        self.events = events
        self.on_phase = on_phase
        self.off_phase = off_phase
        self.phase_column_name = phase_column_name
        self.offset = offset
        self.ax = ax
        self.phases = (self.events.table[self.phase_column_name] + offset) % 1
        self.nbin = None
        self.nperiod = None
        self.nbin_per_period = None
        self._was_called = False

    @property
    def on_phase(self):
        return self._on_phase

    @on_phase.setter
    def on_phase(self, on_phase):
        if on_phase is None:
            self._on_phase = None
        else:
            self._on_phase = self._split_phase_intervals(on_phase)

    @property
    def off_phase(self):
        return self._off_phase

    @off_phase.setter
    def off_phase(self, off_phase):
        if off_phase is None:
            self._off_phase = None
        else:
            self._off_phase = self._split_phase_intervals(off_phase)

    @property
    def phase_column_name(self):
        return self._phase_column_name

    @phase_column_name.setter
    def phase_column_name(self, phase_column_name):
        self._phase_column_name = self._check_column_name(
            self.events, phase_column_name
        )

    @property
    def ax(self):
        return self._ax

    @ax.setter
    def ax(self, ax):
        if ax is None:
            self._ax = plt.gca()
        else:
            self._ax = ax

    @property
    def n_bkg(self):
        if self.off_phase is None:
            return None
        else:
            return np.sum(
                [
                    len(self.events.select_parameter(self.phase_column_name, off).time)
                    for off in self._split_phase_intervals(
                        (np.array(self.off_phase) + self.offset).tolist()
                    )
                ]
            )

    @property
    def bkg_level(self):
        if self.n_bkg is None:
            return None
        off_size = 0
        for off in self.off_phase:
            off_size += np.diff(off)
        return self.n_bkg / self.nbin_per_period / off_size

    def plot(self, ax=None, nbin_per_period=20, nperiod=2, **kwargs):
        phases = self.phases
        self.ax = ax
        self.nbin_per_period = nbin_per_period
        self.nperiod = nperiod
        self.nbin = nbin_per_period * nperiod
        for i in range(1, nperiod):
            phases = np.concatenate([phases, phases + i])
        hist_range = (0, nperiod)
        self.bin_value, self.bin_edges, _ = self.ax.hist(
            phases, bins=self.nbin, range=hist_range, **kwargs
        )
        self._was_called = True
        return self.ax

    def plot_bkg_line(self, **kwargs):
        kwargs.setdefault("color", "k")
        kwargs.setdefault("linestyle", "--")

        if self.bkg_level is None:
            raise ValueError("No bakground phase interval (off_phase) defined.")
        self.ax.axhline(self.bkg_level, **kwargs)

    def plot_error_bar(self, **kwargs):
        self.ax.errorbar(
            (self.bin_edges[:-1] + self.bin_edges[1:]) / 2,
            self.bin_value,
            yerr=np.sqrt(self.bin_value),
            fmt="none",
            color="k",
            **kwargs,
        )

    def set_limits(self, buffer=0.1, set_xlim=True, set_ylim=True):
        if set_ylim is False and set_xlim is False:
            pass
        max_value = max(self.bin_value)
        y_min = self.bkg_level - buffer * (max_value - self.bkg_level)
        y_max = max_value + buffer * (max_value - self.bkg_level)
        if set_ylim:
            self.ax.set_ylim(y_min, y_max)
        if set_xlim:
            self.ax.set_xlim(0, self.nperiod)

    def set_off_patch(self, **kwargs):
        if self.off_phase is None:
            raise ValueError("No bakground phase interval (off_phase) defined.")
        offs = np.array(self.off_phase)
        for i in range(1, self.nperiod):
            offs = np.concatenate([offs, offs + i])
        offs = self._merge_phase_intervals(offs)
        for off in offs:
            self.ax.axvspan(
                off[0], off[1], alpha=0.25, color="white", hatch="x", ec="k", **kwargs
            )

    def set_on_patch(self, **kwargs):
        if self.on_phase is None:
            raise ValueError("No signal phase interval (on_phase) defined.")
        ons = np.array(self.on_phase)
        for i in range(1, self.nperiod):
            ons = np.concatenate([ons, ons + i])
        ons = self._merge_phase_intervals(ons)
        for on in ons:
            self.ax.axvspan(on[0], on[1], alpha=0.25, color="gray", **kwargs)

    @staticmethod
    def _merge_phase_intervals(phase_intervals):
        """Merge phase intervals that overlap.

        Parameters
        ----------
        phase_intervals: list of `tuple`
            Phase intervals to merge.

        Returns
        -------
        merged_intervals: list of `tuple`
            Phase intervals merged.
        """
        phase_intervals = sorted(phase_intervals, key=lambda x: x[0])
        merged_intervals = [phase_intervals[0]]
        for interval in phase_intervals[1:]:
            if interval[0] <= merged_intervals[-1][1]:
                merged_intervals[-1] = (
                    merged_intervals[-1][0],
                    max(interval[1], merged_intervals[-1][1]),
                )
            else:
                merged_intervals.append(interval)
        return merged_intervals

    @staticmethod
    def _check_column_name(events, phase_column_name):
        if phase_column_name not in events.table.colnames:
            raise ValueError(f"Column {phase_column_name} not in events table.")
        return phase_column_name

    @staticmethod
    def _split_phase_intervals(intervals):
        """Split phase intervals that go below phase 0 and above phase 1.

        Parameters
        ----------
        intervals: `tuple`or list of `tuple`
            Phase interval or list of phase intervals to check.

        Returns
        -------
        intervals: list of `tuple`
            Phase interval checked.
        """
        if isinstance(intervals, tuple):
            intervals = [intervals]

        for phase_interval in intervals:
            if phase_interval[0] % 1 > phase_interval[1] % 1:
                intervals.remove(phase_interval)
                intervals.append((phase_interval[0] % 1, 1))
                if phase_interval[1] % 1 != 0:
                    intervals.append((0, phase_interval[1] % 1))
        return intervals
