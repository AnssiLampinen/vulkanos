# Batch run: 250 volcanoes on 10 Aalto machines, started at once

One person runs everything. The 250 volcanoes are split into 10 batches of 25 (`sites/batches/b01.csv` ... `b10.csv`),
one batch per Liftarit machine (`sites/batches/hosts.csv`). Every volcano gets its own weights: r34 is fine-tuned on that
site's OSM buildings, then the site is mapped with them (see "per-site training" in `docs/team_run.md`).
A batch takes about 10 to 11 hours (about 25 minutes per site).

| Batch | Machine     | Batch | Machine     |
|-------|-------------|-------|-------------|
| b01   | agda        | b06   | wonko       |
| b02   | beeblebrox  | b07   | zarniwoop   |
| b03   | fenchurch   | b08   | gallumbits  |
| b04   | hactar      | b09   | lunkwill    |
| b05   | prefect     | b10   | vroomfondel |

If a machine is busy or down, change its row in `sites/batches/hosts.csv`, for example to `zarquon`, and start only that batch.

## Where things go

The home folder is the same on every Aalto machine, with a 100 GB quota. 250 sites need far more than that,
so each batch works on its machine's local disk:

- `/l/$USER/vulkanos` on each machine: a copy of the repo with its own `.venv`, tiles, predictions, maps and weights.
  It is not backed up. If `/l` is missing, the batch falls back to `~/vulkanos-work/<batch>` and prints a WARNING.
- `~/vulkanos-results/` in the home folder, shared by all machines:
  - `logs/bNN.log`: the full log of each batch.
  - `results_bNN.tar.gz`: summaries, scored cells (own weights and r34), training curves. Written when the batch ends.
  - `weights/<site>.pt`: each site's trained weights, about 100 MB each, about 25 GB for all 250.
    Set `NO_WEIGHTS=1` to skip copying them.

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
   `vulkanos`. It prints how busy each GPU was and who is logged in. On its first run, each machine makes its own `.venv`
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
