import time
from collections import defaultdict
from functools import wraps

import time
from collections import defaultdict

class Profiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self._start_times = {}  # Stores start times by name

    def start(self, name):
        if name in self._start_times:
            raise RuntimeError(f"Timer for '{name}' already started.")
        self._start_times[name] = time.perf_counter()

    def stop(self, name):
        if name not in self._start_times:
            raise RuntimeError(f"No timer started for '{name}'.")
        elapsed = time.perf_counter() - self._start_times.pop(name)
        self.timings[name].append(elapsed)

    def summary(self):
        print("=== Profiling Summary ===")
        for key, values in self.timings.items():
            count = len(values)
            total = sum(values)
            avg = total / count
            print(f"{key}: {count} calls | total {total:.6f}s | avg {avg:.6f}s")