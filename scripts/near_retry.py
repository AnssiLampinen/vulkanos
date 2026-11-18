"""Retry the stage 2 (nearest-site) reruns that are not done, on a few Aalto machines with a free GPU.
Normally run through `bash scripts/launch_near.sh retry`, from ~/vulkanos-team:

    python scripts/near_retry.py --dry-run       # what is left, which machines are free, change nothing
    python scripts/near_retry.py --gpus 5        # write data/near/retry/tNN.csv + hosts.csv

A site is done when its map <work>/data/out/<site>_unet-site-near.html exists in the work folder of any
nNN / tNN batch (from ~/vulkanos-results/logs/<batch>.log), or that log says its detect step was ok.
Sites of batches that are still running are left to them.

Machines (the Liftarit classroom, RTX 5070 12 GB) are probed over ssh in parallel. A machine is free
when it answers, its GPU has less than --max-mem MiB in use (someone else's job) and no gpu_run.py of
yours runs there. The --gpus free machines with the fewest logged in users get one batch each.
Galapagos sites (donor > 1000 km away) go last in their batch, or are left out with --skip-galapagos.
"""
import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
# https://www.aalto.fi/en/services/linux-computer-names-in-it-classrooms, Liftarit (Konemiehentie 2, T8 C106.2)
LIFTARIT = ("agda beeblebrox effrafax fenchurch gallumbits gargravarr hactar kapelsen lallafa loonquawl lunkwill "
            "nullify phouchg poodoo prefect quordlepleen random thrashbarg tragula vantrashell varntvar versenwald "
            "vranx vroomfondel wonko wowbagger zarniwoop zarquon zem").split()
SKIP = {"agda", "gallumbits"}  # agda hangs, gallumbits asks a password
PROBE = ("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1; "
         "echo MINE $(pgrep -u $USER -f 'scripts/gpu_run[.]py' | wc -l); "
         "echo USERS $(who | awk '{print $1}' | sort -u | wc -l)")


def batch_log(results, b):
    # (host, work folder, finished?, log text) from run_batch.sh's log, the last start wins
    log = results / "logs" / f"{b}.log"
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    host = work = None
    finished = False
    for line in text.splitlines():
        m = re.match(rf"=== batch {b} on (\S+), .*, work folder (\S+)", line)
        if m:
            host, work, finished = m.group(1), Path(m.group(2)), False
        elif line.startswith(f"=== batch {b} finished"):
            finished = True
    return host, work, finished, text


def probe(host, timeout):
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o",
                            "StrictHostKeyChecking=accept-new", host, PROBE],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return host, None, "ssh timed out"
    if r.returncode != 0:
        return host, None, (r.stderr.strip().splitlines() or ["ssh failed"])[-1][:70]
    info = {"mine": 0, "users": 0, "mem": None, "total": None}
    for line in r.stdout.splitlines():
        if line.startswith("MINE "):
            info["mine"] = int(line.split()[1])
        elif line.startswith("USERS "):
            info["users"] = int(line.split()[1])
        elif re.match(r"^\s*\d+\s*,\s*\d+\s*$", line):
            info["mem"], info["total"] = (int(x) for x in line.split(","))
    if info["mem"] is None:
        return host, None, "no nvidia-smi"
    return host, info, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, default=5, help="how many machines to use")
    p.add_argument("--hosts", help="comma list to probe instead of the whole Liftarit classroom")
    p.add_argument("--exclude", default="", help="comma list of machines not to use")
    p.add_argument("--max-mem", type=int, default=1500, help="MiB of GPU memory in use above which a machine is busy")
    p.add_argument("--skip-galapagos", action="store_true", help="leave the Galapagos sites out")
    p.add_argument("--sites", help="comma list: retry only these")
    p.add_argument("--results", default="~/vulkanos-results")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    results = Path(args.results).expanduser()
    near = ROOT / "data" / "near"
    plan = pd.read_csv(near / "rerun" / "plan.csv")

    # what is done, what is still running
    done, running, busy_hosts = set(), set(), set()
    batches = sorted({f.stem for f in (results / "logs").glob("*.log") if re.fullmatch(r"[nt]\d+", f.stem)})
    for b in batches:
        info = batch_log(results, b)
        if info is None:
            continue
        host, work, finished, text = info
        done |= set(re.findall(r"^=== (\S+) detect with nearest-site weights ok", text, re.M))
        if work is not None and (work / "data" / "out").is_dir():
            done |= {f.name[:-len("_unet-site-near.html")] for f in (work / "data" / "out").glob("*_unet-site-near.html")}
        if not finished:
            csv = (near / ("rerun" if b.startswith("n") else "retry")) / f"{b}.csv"
            if csv.exists():
                running |= set(pd.read_csv(csv)["name"].map(slug))
            if host:
                busy_hosts.add(host)
    todo = plan[~plan["site"].isin(done) & ~plan["site"].isin(running)].copy()
    if args.sites:
        todo = todo[todo["site"].isin(args.sites.split(","))]
    todo["galapagos"] = (todo["lon"] < -88) & (todo["lon"] > -93) & (todo["lat"].abs() < 2)
    print(f"{len(plan)} sites in the plan: {len(done & set(plan['site']))} done, "
          f"{len((running & set(plan['site'])) - done)} in running batches ({' '.join(sorted(busy_hosts)) or 'none'}), "
          f"{len(todo)} to retry")
    if args.skip_galapagos and todo["galapagos"].any():
        print(f"  leaving out Galapagos: {' '.join(todo.loc[todo['galapagos'], 'site'])}")
        todo = todo[~todo["galapagos"]]
    if todo.empty:
        sys.exit("nothing to retry")
    print("  " + " ".join(todo["site"]))

    # free machines
    hosts = args.hosts.split(",") if args.hosts else LIFTARIT
    skip = SKIP | busy_hosts | set(filter(None, args.exclude.split(",")))
    cand = [h for h in hosts if h not in skip]
    print(f"\nprobing {len(cand)} machines (skipping {' '.join(sorted(skip & set(hosts)))})")
    with ThreadPoolExecutor(16) as pool:
        res = list(pool.map(lambda h: probe(h, 30), cand))
    free = []
    for h, info, err in res:
        if info is None:
            print(f"  {h:13} -        {err}")
            continue
        state = "free"
        if info["mine"]:
            state = "busy (your gpu_run.py runs here)"
        elif info["mem"] >= args.max_mem:
            state = "busy (someone else's job)"
        print(f"  {h:13} {info['mem']:5}/{info['total']} MiB, {info['users']} users  {state}")
        if state == "free":
            free.append((info["users"], info["mem"], h))
    use = [h for _, _, h in sorted(free)[:args.gpus]]
    if not use:
        sys.exit("no free machine found")
    print(f"\nusing {len(use)}: {' '.join(use)}")

    # batches, continuing the tNN numbers of earlier retries so their logs stay apart
    out = near / "retry"
    used = [int(m.group(1)) for f in list(out.glob("t*.csv")) + list((results / "logs").glob("t*.log"))
            if (m := re.fullmatch(r"t(\d+)", f.stem))]
    start = max(used, default=0) + 1
    # biggest jobs spread first, Galapagos last in each batch
    todo = todo.sort_values(["galapagos", "osm"], ascending=[True, False]).reset_index(drop=True)
    lines, plan_rows = ["person,host"], []
    for k, h in enumerate(use):
        part = todo.iloc[k::len(use)]
        if part.empty:
            continue
        name = f"t{start + k:02d}"
        lines.append(f"{name},{h}")
        plan_rows.append((name, h, part))
        print(f"  {name} on {h:13} {len(part)} sites: {' '.join(part['site'])}")
    if args.dry_run:
        print("\ndry run, nothing written")
        return
    out.mkdir(parents=True, exist_ok=True)  # earlier tNN.csv stay, a running batch still reads its own
    for name, h, part in plan_rows:
        part[["name", "lat", "lon", "volcano_number"]].to_csv(out / f"{name}.csv", index=False)
    (out / "hosts.csv").write_text("\n".join(lines) + "\n")
    print(f"\nbatches and hosts.csv in {out}")


def slug(name):
    import unicodedata

    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


if __name__ == "__main__":
    main()
