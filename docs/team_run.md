# Team run: 250 volcanoes on 7 Aalto machines

Each of us runs about 36 volcanoes on our own Aalto GPU machine. It takes one command, and it keeps
running after you close your laptop.

**Names to use in the commands (lowercase, exact spelling):** `anssi`, `adit`, `miro`, `janna`, `khoi`, `hieu`, `duc`.
The examples below use `anssi`. Put your own name there instead.

## Who runs what, where

| Person | Machine  | Sites file              |
|--------|----------|-------------------------|
| Anssi  | agda       | `sites/parts/anssi.csv` |
| Adit   | beeblebrox | `sites/parts/adit.csv`  |
| Miro   | fenchurch  | `sites/parts/miro.csv`  |
| Janna  | hactar     | `sites/parts/janna.csv` |
| Khoi   | prefect    | `sites/parts/khoi.csv`  |
| Hieu   | wonko      | `sites/parts/hieu.csv`  |
| Duc    | zarniwoop  | `sites/parts/duc.csv`   |

All seven are in the Liftarit classroom (Konemiehentie 2, T8 C106.2) and have an RTX 5070 with 12 GB.
They are listed on Aalto's [Linux computer names in IT classrooms](https://www.aalto.fi/en/services/linux-computer-names-in-it-classrooms) page.

These are shared classroom machines. If yours is busy or down, use a spare from the same room:
`gallumbits`, `lunkwill`, `vroomfondel`, `zarquon`, or any other name from the Liftarit list on that page.
To check whether someone else is using a machine, run `htop` or `nvidia-smi` on it. Tell the group
which spare you took, so that two people don't pick the same one.

The split is made by `scripts/split_sites.py`. Sites are sorted by region and dealt out in turn,
so everyone gets a similar mix and about the same amount of work.

## 1. Set up SSH (once, skip if `ssh kosh.aalto.fi` already works)

From outside the Aalto network you need **both** your Aalto password and an SSH key. Aalto WiFi on your
own laptop counts as outside too. Follow Aalto's instructions:

- [Creating SSH keys for public Linux servers in Aalto](https://www.aalto.fi/en/services/creating-ssh-keys-for-public-linux-servers-in-aalto). This covers making the key and putting it on your Aalto account.
- [Linux shell servers at Aalto](https://www.aalto.fi/en/services/linux-shell-servers-at-aalto) explains the password plus key rule and `kosh.aalto.fi`.
- [Remote access (Aalto SciComp)](https://scicomp.aalto.fi/aalto/remoteaccess/) has more help, for example on using the VPN.

Short version from those pages, with `aaltokey` as the key name:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/aaltokey      # pick a passphrase that is NOT your Aalto password
# then add the contents of ~/.ssh/aaltokey.pub to ~/.ssh/authorized_keys in your Aalto home,
# as the Aalto page explains (e.g. ssh-copy-id while on the Aalto network or VPN)
```

On Windows, use PowerShell and run `ssh-keygen.exe -t ed25519 -f $HOME\.ssh\aaltokey`.

Optional: add this to `~/.ssh/config` on your laptop so that `ssh vulkanos` takes you straight to your machine.
Replace `USERNAME` with your Aalto username and `MACHINE` with your machine from the table:

```
Host kosh
    HostName kosh.aalto.fi
    User USERNAME
    IdentityFile ~/.ssh/aaltokey

Host vulkanos
    HostName MACHINE
    User USERNAME
    IdentityFile ~/.ssh/aaltokey
    ProxyJump kosh
```

Without the config, connect in two hops instead:

```bash
ssh -i ~/.ssh/aaltokey USERNAME@kosh.aalto.fi
ssh MACHINE           # run this on kosh, e.g. ssh agda
```

## 2. Run your part (on your Aalto machine, after you ssh in)

Everything in this step runs on the Aalto machine from the table, not on your own laptop. The first time, get the code:

```bash
git clone -b gpu-run https://github.com/AnssiLampinen/vulkanos.git ~/vulkanos-team
```

Then, every time, run this. Replace `anssi` with your own name, spelled exactly as in the list at the top:

```bash
cd ~/vulkanos-team && bash scripts/run_part.sh anssi
```

That one command does everything:

1. It updates the code with `git pull`.
2. The first time, it installs Python packages into `.venv`, CUDA torch included. This takes about 5 to 10 minutes.
3. It downloads the detector weights (`models/ercolano-r34/best.pt`) on its own.
4. It runs every site in your file: imagery tiles, building detection, OSM buildings, comparison, and map.
5. It packs the results into `results_<name>.tar.gz`.

## 3. Disconnect, then come back later

The run happens inside **tmux**, so it doesn't need your SSH connection to stay open.

- To leave it running, press `Ctrl-b`, let go, then press `d`. Then close the window or log out.
- To check on it later, ssh to the same machine and run the same command again
  (`bash scripts/run_part.sh anssi`). It re-attaches to the running session and does not start a second run.
  `tmux attach -t vulkanos` does the same thing.
- If it stopped (machine rebooted, crash), run the same command again. Finished sites are skipped.
- You must come back to the **same machine**. The tmux session lives on that machine, not on kosh.

## 4. When it's done

The window says `done` and shows where the results are, for example `~/vulkanos-team/results_anssi.tar.gz`.
The file is small: summary, scored cells and logs, no maps. Copy it to your laptop and send it to Anssi
on Telegram:

```bash
scp vulkanos:vulkanos-team/results_anssi.tar.gz .          # with the ssh config above
# without it: scp -J USERNAME@kosh.aalto.fi USERNAME@MACHINE:vulkanos-team/results_anssi.tar.gz .
```

The maps are in `~/vulkanos-team/data/out/*.html` if you want to look at a volcano.

## If something goes wrong

- **Some sites failed.** Overpass (the OSM server) is sometimes overloaded. Run the same command again and only the failed sites rerun. Each site's log is in `data/logs/<site>.log`.
- **`Disk quota exceeded`.** Each site takes roughly 0.5 GB in your home. Check with `df -h ~`, and tell Anssi.
- **`cuda available False`.** You are probably on a machine without a working GPU. Try a spare machine.
- **Anything else.** Send the last 30 lines of `data/logs/part_<name>.log` to the group.
