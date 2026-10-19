"""
run_all.py — Run the full experiment matrix (resumable)
=======================================================
Jobs whose metrics.json already exists are skipped, so the script can be
stopped and restarted at any time.  Each job's stdout/stderr goes to
logs/jobs/<domain>__<run>__seed<k>.txt.

    python run_all.py --workers 4            # everything
    python run_all.py --only tabular --dry   # list pending tabular jobs
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SEEDS = [0, 1, 2, 3, 4]
SWEEP_SEEDS = [0, 1, 2]
SWEEP_P = [0.005, 0.01, 0.02]


def jobs():
    J = []  # (domain, run_dir_name, seed, argv)

    def gan(domain, model, seed, extra=(), name=None):
        from train_gan import get_args, run_name
        argv = ["train_gan.py", "--domain", domain, "--model", model, "--seed", str(seed), *extra]
        rn = name or run_name(get_args(argv[1:]))
        J.append((domain, rn, seed, argv))

    for domain in ("tabular", "stocks", "mnist"):
        for s in SEEDS:
            gan(domain, "qgennet", s)
            gan(domain, "qgennet", s, ["--no_zne"])
            gan(domain, "qgennet", s, ["--no_entangle"])
            gan(domain, "decoder_only", s)
            gan(domain, "mlp", s)
            if domain == "mnist":
                gan(domain, "dcgan", s)
            if domain == "tabular":
                J.append((domain, "ctgan", s, ["run_ctgan.py", "--seed", str(s)]))
                J.append((domain, "ctgan_e1500", s, ["run_ctgan.py", "--seed", str(s), "--epochs", "1500"]))
                gan(domain, "qgennet", s, ["--no_feat_norm"])
            if domain == "stocks":
                J.append((domain, "timegan", s, ["timegan.py", "--seed", str(s)]))
    for p in SWEEP_P:
        for s in SWEEP_SEEDS:
            gan("tabular", "qgennet", s, ["--p", str(p)])
            gan("tabular", "qgennet", s, ["--p", str(p), "--no_zne"])
    return J


def done(domain, rn, seed):
    return os.path.exists(os.path.join(HERE, "runs", domain, rn, f"seed{seed}", "metrics.json"))


NO_NEW_AFTER = None      # epoch seconds; set from --no_new_after
PROCS = []
STOPPING = False


def run(job):
    domain, rn, seed, argv = job
    if STOPPING or (NO_NEW_AFTER is not None and time.time() > NO_NEW_AFTER):
        print(f"{time.strftime('%H:%M:%S')}  not started  {domain}/{rn}/seed{seed} (cut-off)", flush=True)
        return
    os.makedirs(os.path.join(HERE, "logs", "jobs"), exist_ok=True)
    log = os.path.join(HERE, "logs", "jobs", f"{domain}__{rn}__seed{seed}.txt")
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.Popen([PY, *argv], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
        PROCS.append(p)
        rc = p.wait()
    if STOPPING and not done(domain, rn, seed):
        print(f"{time.strftime('%H:%M:%S')}  stopped      {domain}/{rn}/seed{seed} (hard stop; will rerun)", flush=True)
        return
    status = "ok" if rc == 0 and done(domain, rn, seed) else f"FAILED rc={rc}"
    print(f"{time.strftime('%H:%M:%S')}  {status:12s} {domain}/{rn}/seed{seed}  "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no_new_after", default=None, help='"YYYY-mm-dd HH:MM": start no job after this')
    ap.add_argument("--hard_stop", default=None, help='"YYYY-mm-dd HH:MM": kill running jobs and exit')
    a = ap.parse_args()
    parse = lambda s: time.mktime(time.strptime(s, "%Y-%m-%d %H:%M"))
    if a.no_new_after:
        NO_NEW_AFTER = parse(a.no_new_after)
    if a.hard_stop:
        import threading

        def watchdog(t_stop=parse(a.hard_stop)):
            global STOPPING
            while time.time() < t_stop:
                time.sleep(20)
            STOPPING = True
            print(f"{time.strftime('%H:%M:%S')}  HARD STOP reached; terminating running jobs", flush=True)
            for p in PROCS:
                if p.poll() is None:
                    p.kill()
        threading.Thread(target=watchdog, daemon=True).start()
    todo = [j for j in jobs() if not done(*j[:3]) and (a.only is None or j[0] == a.only)]
    # cheap jobs first so tables fill early; quantum MNIST last
    cost = lambda j: (j[0] == "mnist", "qgennet" in j[1] and "noent" not in j[1])
    todo.sort(key=cost)
    print(f"{len(todo)} pending jobs", flush=True)
    if a.dry:
        for j in todo:
            print(" ", j[0], j[1], j[2])
        sys.exit(0)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, todo))
    left = [j for j in jobs() if not done(*j[:3])]
    print("ALL DONE" if not left else f"PAUSED with {len(left)} jobs left (rerun run_all.py to resume)",
          flush=True)
