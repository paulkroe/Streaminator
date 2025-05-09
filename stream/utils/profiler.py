import time
from collections import defaultdict
from functools import wraps

import time
from collections import defaultdict

class Profiler:
    """
    Profiler class to track the time taken for different operations.
    """
    def __init__(self):
        self.timings = defaultdict(list)
        self._start_times = {}  # Stores start times by name

    def start(self, name):
        """
        Start a timer for a given operation.

        Args:
            name: The name of the operation to start the timer for.
        """
        if name in self._start_times:
            raise RuntimeError(f"Timer for '{name}' already started.")
        self._start_times[name] = time.perf_counter()

    def stop(self, name):
        """
        Stop a timer for a given operation.

        Args:
            name: The name of the operation to stop the timer for.
        """
        if name not in self._start_times:
            raise RuntimeError(f"No timer started for '{name}'.")
        elapsed = time.perf_counter() - self._start_times.pop(name)
        self.timings[name].append(elapsed)

    def summary(self):
        """
        Print a summary of the timings for all operations.
        """
        print("=== Profiling Summary ===")
        for key, values in self.timings.items():
            count = len(values)
            total = sum(values)
            avg = total / count
            print(f"{key}: {count} calls | total {total:.6f}s | avg {avg:.6f}s")