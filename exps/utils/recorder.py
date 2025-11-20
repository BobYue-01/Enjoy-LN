import time
import psutil
import os
from contextlib import contextmanager

process = psutil.Process(os.getpid())


def mem():
    return process.memory_info().rss / (1024**2)


class StatsRecorder:
    """A tiny context-manager based recorder for timed sections and memory deltas.

    Usage:
        recorder = StatsRecorder()
        with recorder.section('name'):
            do_work()
        # later: recorder.stats contains recorded entries
    """

    def __init__(self):
        self.stats: dict[str, list[dict]] = {}

    @contextmanager
    def section(self, field: str, ident: str = None):
        if field not in self.stats:
            self.stats[field] = []
        t0 = time.time()
        m0 = mem()
        try:
            yield
        finally:
            t1 = time.time()
            m1 = mem()
            self.stats[field].append(
                {"time": t1 - t0, "mem": m1 - m0, "id": ident}
            )

    def plot_stats(self, save_path: str = None):
        import matplotlib.pyplot as plt

        for field, records in self.stats.items():
            times = [record['time'] for record in records]
            plt.plot(times, marker='o', label=field)

        plt.xlabel('Iteration')
        plt.ylabel('Time (seconds)')
        plt.title('Section Timing')
        plt.legend()
        if save_path:
            plt.savefig(save_path)

        # find largest time's corresponding identifier
        for field, records in self.stats.items():
            max_time = max(records, key=lambda x: x['time'])
            print(
                f"Max time for section '{field}':",
                f"{max_time['time']:.4f} seconds (ID: {max_time['id']})"
            )
