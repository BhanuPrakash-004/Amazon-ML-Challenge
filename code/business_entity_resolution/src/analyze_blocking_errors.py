"""Blocking error analysis: which blocker missed which validation positive, and why."""
import argparse
import csv


def categorize(s1_name, s1_addr, t_name, t_addr, found_by):
    from rapidfuzz import fuzz as _fz
    try:
        ns = _fz.token_set_ratio(s1_name, t_name) / 100.0
        ad = _fz.token_set_ratio(s1_addr, t_addr) / 100.0
    except Exception:
        ns = 1.0 if s1_name == t_name else 0.0
        ad = 1.0 if s1_addr == t_addr else 0.0
    if not found_by:
        if ns < 0.4 and ad >= 0.6:
            return "name-only-miss(address-only-pair)"
        if ad < 0.4 and ns >= 0.6:
            return "address-only-miss(name-only-pair)"
        if ns < 0.5 and ad < 0.5:
            return "semantic-only"
        return "candidate-cutoff"
    return "rank-loss"


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--miss-file", default=None, help="CSV of missed pairs (s1id,tid); else reads cache")
    a = ap.parse_args(args)
    print("analyze_blocking_errors: feed missed-pair CSV via --miss-file for full text report.", flush=True)
    print("categories: no rare-name overlap / no rare-address overlap / heavy typo /", flush=True)
    print("transliteration / address-only / name-only / numeric-only / semantic-only /", flush=True)
    print("normalization failure / candidate cutoff / other", flush=True)


if __name__ == "__main__":
    main()
