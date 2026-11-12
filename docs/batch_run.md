# Batch run: 250 volcanoes on 10 Aalto machines, started at once

One person runs everything. The 250 volcanoes are split into 10 batches of 25 (`sites/batches/b01.csv` ... `b10.csv`),
one batch per Liftarit machine (`sites/batches/hosts.csv`). Every volcano gets its own weights: r34 is fine-tuned on that
site's OSM buildings, then the site is mapped with them (see "per-site training" in `docs/team_run.md`).
A batch takes about 10 to 11 hours (about 25 minutes per site).

| Batch | Machine     | Batch | Machine     |
|-------|-------------|-------|-------------|
| b01   | agda        | b06   | wonko       |
| b02   | zarquon     | b07   | zarniwoop   |
| b03   | fenchurch   | b08   | gallumbits  |
| b04   | hactar      | b09   | lunkwill    |
| b05   | prefect     | b10   | wowbagger   |

If a machine is busy or down, change its row in `sites/batches/hosts.csv`, for example to `effrafax`, and start only that batch.
Other Liftarit machines: effrafax, gargravarr, kapelsen, lallafa, loonquawl, nullify, phouchg, poodoo, quordlepleen,
random, thrashbarg, tragula, vantrashell, varntvar, versenwald, vranx, zem. On 2026-11-08 beeblebrox and vroomfondel did not resolve.

## Where things go

The home folder is the same on every Aalto machine, with a 100 GB quota. 250 full sites need far more than that,
so each batch works in a copy of the repo on its machine's local disk when there is one:

1. `/l/$USER/vulkanos` when `/l` is writable (it is not on agda).
2. `/tmp/vulkanos-$USER` when `/tmp` has at least 60 GB free.
3. Otherwise `~/vulkanos-work/<batch>` in the home folder, and the log says NOTE. Each site's tiles and predictions are
   then deleted once the site is done, so about 0.12 GB per site stays (weights, scores, map without imagery).
   All 250 sites then take about 30 GB, plus up to about 2.5 GB per running batch while it works.

Every batch uses the one `.venv` in `~/vulkanos-team` (about 6 GB), which `launch_all.sh start` makes the first time.
Results go to `~/vulkanos-results/` in the home folder, shared by all machines:

- `logs/bNN.log`: the full log of each batch.
- `results_bNN.tar.gz`: summaries, scored cells (own weights and r34), training curves. Written when the batch ends.
- `weights/<site>.pt`: each site's trained weights, about 100 MB each, about 25 GB for all 250.
  In case 3 they stay in `~/vulkanos-work/<batch>/models/<site>/site-osm/` instead. Set `NO_WEIGHTS=1` to skip copying them.

Check the home folder with `du -sh ~` before starting. Keep at least 50 GB free.

## Run it

1. From your laptop, ssh to the first machine with `-A`, so that your key comes along for the next hops:

   ```bash
   ssh -A -J USERNAME@kosh.aalto.fi USERNAME@agda
   ```

2. The first time, get the code:

   ```bash
   git clone -b gpu-run https://github.com/AnssiLampinen/vulkanos.git ~/vulkanos-team
   ```

3. Start all 10:

   ```bash
   cd ~/vulkanos-team && bash scripts/launch_all.sh start
   ```

   For each batch, this ssh's to its machine and starts `scripts/run_batch.sh bNN` in a detached tmux session called
   `vulkanos`. It prints how busy each GPU was and who is logged in. The first time, it makes `.venv` before starting
   (5 to 10 minutes). Then you can log out.

4. Check on them from any Aalto machine:

   ```bash
   cd ~/vulkanos-team && bash scripts/launch_all.sh status
   ```

   This prints one line per batch, for example `b03 fenchurch   7/25 sites, === taal train: ...`.
   To watch one batch live, run `ssh fenchurch` and then `tmux attach -t vulkanos`. To detach, press Ctrl-b then d.

5. When all 10 are finished, `~/vulkanos-results/results_b01.tar.gz` ... `b10` hold everything. Copy them to your laptop:

   ```bash
   scp -J USERNAME@kosh.aalto.fi 'USERNAME@agda:vulkanos-results/results_b*.tar.gz' .
   ```

## If something goes wrong

- **Some sites failed** (Overpass busy, for example): run `bash scripts/launch_all.sh start` again.
  Running batches are left alone. Finished batches start again, but skip the sites that are already done.
- **Start or stop single batches:** `bash scripts/launch_all.sh start b03 b07` or `bash scripts/launch_all.sh stop b03`.
- **`ssh failed`:** you probably logged in without `-A`, or that machine is down. Log in again with `ssh -A`,
  or move the batch to another machine in `hosts.csv`.
- **A machine rebooted:** its tmux session is gone. `start` for that batch picks up where it stopped.
- These are classroom machines. If someone is sitting at one, move the batch to a spare machine.

## The 14 missing volcanoes (b11 ... b17)

14 volcanoes were left out of the 250 (Tanzania, Virunga, Chiles-Cerro Negro and 4 mid-ocean ones). They are
`sites/batches/b11.csv` ... `b17.csv`, 2 sites each, on agda, zarquon, fenchurch, hactar, prefect, wonko and zarniwoop
(rows added to `hosts.csv`). Same pipeline, about an hour in all:

    cd ~/vulkanos-team && git pull && bash scripts/launch_missing.sh start
    bash scripts/launch_missing.sh status

Results land next to the others: `~/vulkanos-results/results_b11.tar.gz` ... `b17`, maps in `~/vulkanos-work/b11/data/out/` ...
The mid-ocean sites (both EPR sites, Romanche Fracture Zone, Ascension) have no people near them and will most likely give
no houses. Don't run plain `launch_all.sh start` with no batches: it would restart b01 ... b10 too.
