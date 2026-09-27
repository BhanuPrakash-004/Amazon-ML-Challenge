"""Hardware introspection: GPU/CPU/RAM. Never assumes unified 32GB VRAM."""
import os


def get_info():
    info = {"gpu_count": 0, "gpus": [], "cuda": False, "cpu": os.cpu_count()}
    try:
        import torch
        info["cuda"] = bool(torch.cuda.is_available())
        info["gpu_count"] = torch.cuda.device_count() if info["cuda"] else 0
        for i in range(info["gpu_count"]):
            try:
                p = torch.cuda.get_device_properties(i)
                info["gpus"].append({"id": i, "name": p.name, "mem_gb": round(p.total_memory / 1e9, 1)})
            except Exception:
                pass
    except Exception:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_gb"] = round(vm.total / 1e9, 1)
        info["ram_avail_gb"] = round(vm.available / 1e9, 1)
    except Exception:
        info["ram_gb"] = -1
    return info


def print_info():
    i = get_info()
    print("[HARDWARE] cpu=%s ram_total_gb=%s cuda=%s gpu_count=%d"
          % (i.get("cpu"), i.get("ram_gb"), i.get("cuda"), i.get("gpu_count")), flush=True)
    for g in i.get("gpus", []):
        print("[HARDWARE] gpu%d: %s %.1fGB" % (g["id"], g["name"], g["mem_gb"]), flush=True)
    return i


def pick_device(prefer=0):
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            d = prefer if 0 <= prefer < n else 0
            return "cuda:%d" % d
    except Exception:
        pass
    return "cpu"
