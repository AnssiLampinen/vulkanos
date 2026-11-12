"""Run the whole pipeline for several sites on a GPU machine, one log per site.

    python scripts/gpu_run.py                       # default sites with models/ercolano-r34/best.pt
    python scripts/gpu_run.py --train               # train a new unet on this machine first, then run
    python scripts/gpu_run.py --site merapi --site taal --radius 5
    python scripts/gpu_run.py --per-site-train      # retrain the weights for each site, then run with them

--per-site-train, for every site: tiles + osm, r34 detection, a training set from that site's osm
(every cell, background ignored where r34 sees a building osm lacks), fine-tune r34 on it into
models/<site>/site-osm, then detection, comparison and map with those weights. r34 itself is only read.

The checkpoint is downloaded from --model-url when it isn't on disk. A failing site is logged and
skipped. At the end data/out/summary_<detector>.csv has one row per site.
"""
import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = [sys.executable, "-u", str(ROOT / "src" / "main.py")]
SITES = ["ercolano", "vesuvius", "merapi", "nyiragongo", "goma", "fuego", "taal", "sakurajima", "mayon", "agung"]
MODEL_URL = "https://github.com/AnssiLampinen/vulkanos/raw/models/ercolano-r34/best.pt"


def check_gpu(device):
    import torch

    print(f"torch {torch.__version__}, cuda build {torch.version.cuda}, cuda available {torch.cuda.is_available()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} {p.total_memory / 2**30:.0f} GB")
    if device not in ("cpu", "mps") and not torch.cuda.is_available():
        print("WARNING: no cuda, this runs on the cpu and will be slow. Reinstall torch with setup.sh / setup.bat")


def step(name, cmd, log):
    # output goes to the screen and to the log file
    print(f"\n=== {name}: {' '.join(cmd[2:])}", flush=True)
    t0 = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        code = p.wait()
    print(f"=== {name} {'ok' if code == 0 else f'FAILED ({code})'} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return code == 0


def fetch_model(path, url):
    print(f"downloading {url} to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", action="append", help=f"default {' '.join(SITES)}")
    p.add_argument("--sites-csv", help="name,lat,lon csv of extra sites; ids are the names in lower_snake_case")
    p.add_argument("--list-sites", action="store_true", help="print the site ids (with --sites-csv) and stop")
    p.add_argument("--checkpoint", default="models/ercolano-r34/best.pt")
    p.add_argument("--model-url", default=MODEL_URL, help="where to get the checkpoint when it isn't on disk")
    p.add_argument("--train", action="store_true", help="train a new unet here first and use it")
    p.add_argument("--train-site", action="append", help="sites for the training set, default ercolano")
    p.add_argument("--name", default="school-r34", help="run name for --train, saved to models/<name>")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int, help="training batch size, 16-32 fits a 12-24 GB card")
    p.add_argument("--dataset-args", default="", help="extra args for the dataset step, e.g. '--all-cells'")
    p.add_argument("--radius", type=float, help="grid radius in km, default radius_km in config.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=16, help="tile download threads")
    p.add_argument("--detect-batch", type=int, default=32)
    p.add_argument("--per-site-train", action="store_true",
                   help="fine-tune --checkpoint on each site's own osm first and detect with that")
    p.add_argument("--site-epochs", type=int, default=10, help="epochs for --per-site-train")
    p.add_argument("--min-osm", type=int, default=300,
                   help="--per-site-train: with fewer osm buildings there is too little to learn from, "
                        "the site keeps --checkpoint unchanged")
    p.add_argument("--clean", action="store_true",
                   help="--per-site-train: delete the imagery tiles and predictions after each finished site, to "
                        "save disk. The scores, weights and map stay, the map just has no imagery under it")
    p.add_argument("--keep-dataset", action="store_true", help="keep data/<site>/train_osm (about 2 GB a site)")
    p.add_argument("--skip-done", action="store_true", help="skip sites that already have their map in data/out")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    if args.sites_csv:
        from main import csv_sites

        csv_ids = [s["id"] for s in csv_sites(args.sites_csv)]
        if args.list_sites:
            print("\n".join(csv_ids))
            return
        sites = args.site or csv_ids
    else:
        sites = args.site or SITES

    check_gpu(args.device)
    common = ["--config", args.config, "--device", args.device, "--workers", str(args.workers)]
    if args.sites_csv:
        common += ["--sites-csv", str(Path(args.sites_csv).resolve())]
    logs = ROOT / "data" / "logs"
    ckpt = ROOT / args.checkpoint

    if args.train:
        tsites = [a for s in args.train_site or ["ercolano"] for a in ("--site", s)]
        log = logs / f"train_{args.name}.log"
        extra = (["--epochs", str(args.epochs)] if args.epochs else []) + \
                (["--batch-size", str(args.batch_size)] if args.batch_size else [])
        ok = (step("training tiles + osm", MAIN + ["run", "--step", "tiles", "--step", "osm"] + tsites + common, log)
              and step("dataset", MAIN + ["dataset"] + tsites + args.dataset_args.split() + common, log)
              and step("train", MAIN + ["train", "--name", args.name] + extra + common, log))
        if not ok:
            sys.exit(f"training failed, see {log}")
        ckpt = ROOT / "models" / args.name / "best.pt"
    elif not ckpt.exists():
        try:
            fetch_model(ckpt, args.model_url)
        except Exception as e:
            sys.exit(f"no {args.checkpoint} and download failed ({e}). Copy it here, or use --train")

    radius = ["--radius", str(args.radius)] if args.radius else []
    if args.per_site_train:
        per_site(args, sites, ckpt, radius, common, logs)
        return
    failed = []
    tag = "unet-" + ckpt.parent.name
    for s in sites:
        done = ROOT / "data" / "out" / f"{s}_{tag}.html"
        if args.skip_done and done.exists():
            print(f"\n=== {s}: done already ({done.name}), skipping")
            continue
        cmd = MAIN + ["run", "--site", s, "--checkpoint", str(ckpt.relative_to(ROOT)),
                      "--detect-batch", str(args.detect_batch)] + radius + common
        if not step(s, cmd, logs / f"{s}.log"):
            failed.append(s)

    det = ["--checkpoint", str(ckpt.relative_to(ROOT))]
    step("summary", MAIN + ["summary"] + [a for s in sites for a in ("--site", s)] + det + common,
         logs / "summary.log")
    print(f"\nmaps in data/out/, logs in {logs}")
    if failed:
        sys.exit(f"failed sites: {' '.join(failed)} (see their logs)")


SITE_MODEL = "site-osm"  # models/<site>/site-osm, so every site's detector tag is unet-site-osm


def site_checkpoint(site):
    d = ROOT / "models" / site / SITE_MODEL
    return d / "best.pt" if (d / "best.pt").exists() else d / "last.pt"


def train_site(args, s, base, radius, common, log):
    # weights for one site, trained on its own osm. returns False when a step failed
    rel = str(base.relative_to(ROOT))
    base_tag = "unet-" + base.parent.name
    out = ROOT / "models" / s / SITE_MODEL
    if (out / "done").exists():
        print(f"\n=== {s}: weights trained already ({out.relative_to(ROOT)})")
        return True
    if not (step(f"{s} tiles + osm", MAIN + ["run", "--site", s, "--step", "tiles", "--step", "osm"] + radius + common, log)
            and step(f"{s} detect with {base_tag}", MAIN + ["run", "--site", s, "--step", "detect", "--step", "compare",
                     "--checkpoint", rel, "--detect-batch", str(args.detect_batch)] + common, log)):
        return False
    import geopandas as gpd

    n_osm = len(gpd.read_parquet(ROOT / "data" / s / "osm_buildings.parquet"))
    out.mkdir(parents=True, exist_ok=True)
    if n_osm < args.min_osm:
        msg = f"only {n_osm} osm buildings (< --min-osm {args.min_osm}), keeping {rel} unchanged for this site"
        print(f"\n=== {s}: {msg}")
        shutil.copy2(base, out / "best.pt")  # a copy, the original stays as it is
        (out / "done").write_text(msg + "\n")
        return True
    dataset = ROOT / "data" / s / "train_osm"
    ok = (step(f"{s} dataset", MAIN + ["dataset", "--site", s, "--all-cells", "--suspect", base_tag,
               "--dataset", str(dataset.relative_to(ROOT))] + common, log)
          and step(f"{s} train", MAIN + ["train", "--dataset", str(dataset.relative_to(ROOT)), "--name", f"{s}/{SITE_MODEL}",
                   "--init", rel, "--epochs", str(args.site_epochs)]
                   + (["--batch-size", str(args.batch_size)] if args.batch_size else []) + common, log))
    if not ok:
        return False
    if not args.keep_dataset:
        shutil.rmtree(dataset, ignore_errors=True)
    if (out / "best.pt").exists():
        (out / "last.pt").unlink(missing_ok=True)  # about 100 MB each, best.pt is the one used
    (out / "done").write_text(f"fine-tuned from {rel} on {n_osm} osm buildings, {args.site_epochs} epochs\n")
    return True


def per_site(args, sites, base, radius, common, logs):
    failed = []
    for s in sites:
        done = ROOT / "data" / "out" / f"{s}_unet-{SITE_MODEL}.html"
        if args.skip_done and done.exists():
            print(f"\n=== {s}: done already ({done.name}), skipping")
            continue
        log = logs / f"{s}.log"
        if not train_site(args, s, base, radius, common, log):
            failed.append(s)
            continue
        cmd = MAIN + ["run", "--site", s, "--step", "detect", "--step", "compare", "--step", "report",
                      "--checkpoint", str(site_checkpoint(s).relative_to(ROOT)),
                      "--detect-batch", str(args.detect_batch)] + common
        if not step(f"{s} detect with its own weights", cmd, log):
            failed.append(s)
        elif args.clean:
            shutil.rmtree(ROOT / "data" / s / "pred", ignore_errors=True)
            shutil.rmtree(ROOT / "data" / "tiles", ignore_errors=True)

    site_args = [a for s in sites for a in ("--site", s)]
    # the tag is all summary takes from the checkpoint, so any site's path gives unet-site-osm
    step("summary", MAIN + ["summary"] + site_args + ["--checkpoint", f"models/{sites[0]}/{SITE_MODEL}/best.pt"] + common,
         logs / "summary.log")
    step(f"summary {base.parent.name}", MAIN + ["summary"] + site_args + ["--checkpoint", str(base.relative_to(ROOT))]
         + common, logs / "summary.log")
    print(f"\nmaps in data/out/, weights in models/<site>/{SITE_MODEL}, logs in {logs}")
    if failed:
        sys.exit(f"failed sites: {' '.join(failed)} (see their logs)")


if __name__ == "__main__":
    main()
