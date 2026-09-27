"""Runtime safeguards (Sec 28): stage timing, rows/sec, RAM, GPU, candidates."""
import time


def ram_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        except Exception:
            return -1.0


def gpu_text():
    try:
        import torch
        if torch.cuda.is_available():
            parts = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                parts.append("gpu%d %.1f/%.1fGB" % (i, (total - free) / 1e9, total / 1e9))
            return " ".join(parts)
    except Exception:
        pass
    return "gpu n/a"


def stage(tag, t0, rows=0, extra=""):
    el = max(time.time() - t0, 1e-6)
    ram = ram_gb()
    print("[%s] %.1fs rows=%d %.0f/s ram=%.1fGB %s %s" %
          (tag, el, rows, rows / el, ram, gpu_text(), extra), flush=True)
