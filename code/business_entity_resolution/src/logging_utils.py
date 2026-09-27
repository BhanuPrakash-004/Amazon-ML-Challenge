"""Stage logging with records/elapsed/throughput/RAM/GPU/candidates."""
import time

from . import hardware as _hw


def _ram():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return "ram=%.1f/%.1fGB" % (vm.used / 1e9, vm.total / 1e9)
    except Exception:
        return "ram=?"


class StageLog:
    def __init__(self, stage, total=0):
        self.stage = stage
        self.total = total
        self.t0 = time.time()
        self.done = 0

    def update(self, n, extra=""):
        self.done = n
        el = time.time() - self.t0
        rate = n / el if el > 0 else 0
        if self.total:
            eta = (self.total - n) / rate if rate > 0 else -1
            eta_s = ("ETA %d:%02d" % (int(eta // 60), int(eta % 60))) if eta >= 0 else "ETA --"
            print("[%s] %d/%d (%.1f%%) | %.0f/s | elapsed %d:%02d | %s | %s %s"
                  % (self.stage, n, self.total, 100.0 * n / self.total, rate,
                     int(el // 60), int(el % 60), eta_s, _ram(), extra), flush=True)
        else:
            print("[%s] %d | %.0f/s | elapsed %ds | %s %s"
                  % (self.stage, n, rate, int(el), _ram(), extra), flush=True)

    def close(self, extra=""):
        el = time.time() - self.t0
        print("[%s] DONE %d in %d:%02d | %s %s"
              % (self.stage, self.done, int(el // 60), int(el % 60), _ram(), extra), flush=True)
