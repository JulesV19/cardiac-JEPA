"""Driver de la campagne : énumère et lance toute la matrice, idempotent et reprenable.

Matrice (nommage `runs/<backbone>/<arm>/<regime>/s<seed>/result.json`) :
- pretrain  : 1 JEPA par backbone           -> runs/<bb>/pretrain/best.pt
- probe     : {jepa, scratch} × graines      -> runs/<bb>/<arm>/probe/s<seed>
- finetune  : {jepa, scratch} × fracs × graines
- supervised: xresnet BN from-scratch (référence hors grille) -> runs/xresnet_bn/supervised/ft<pct>/s<seed>

Chaque run dont le `result.json` (ou `best.pt` pour le pretrain) existe déjà est SAUTÉ :
on peut relancer après une coupure Colab sans rien recalculer. `--dry-run` affiche le plan.

    python -m jepa.experiment --dry-run
    python -m jepa.experiment --steps pretrain
    python -m jepa.experiment --steps probe,finetune,supervised --backbones vit,cnn,xresnet
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path

# Piper le dry-run vers `head`/`tail` ferme le stdout tôt : rétablir le comportement Unix par
# défaut (terminer silencieusement) plutôt qu'une BrokenPipeError bruyante.
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

RUNS = Path("runs")
CFG = "jepa/configs/{}.yaml"


def _pct(frac: float) -> int:
    return int(round(frac * 100))


def _skip(marker: Path, force: bool) -> bool:
    return marker.exists() and not force


def build_plan(backbones, arms, fracs, ft_seeds, probe_seeds, steps):
    """Retourne la liste des (marker, cmd) à exécuter, dans l'ordre pretrain -> probe -> ft -> sup."""
    plan = []
    py = [sys.executable]

    if "pretrain" in steps:
        for bb in backbones:
            marker = RUNS / bb / "pretrain" / "best.pt"
            cmd = py + ["-m", "jepa.train", "--config", CFG.format(bb),
                        "--out", f"{bb}/pretrain", "--resume", "auto"]
            plan.append((marker, cmd))

    def eval_cmd(bb, mode, arm_flag, out, extra):
        return py + ["-m", "jepa.eval", "--mode", mode, "--config", CFG.format(bb),
                     "--out", str(out)] + arm_flag + extra

    for bb in backbones:
        ckpt = f"runs/{bb}/pretrain/best.pt"
        arm_flag = {"jepa": ["--ckpt", ckpt], "scratch": ["--random-init"]}
        if "probe" in steps:
            for arm in arms:
                for s in probe_seeds:
                    out = RUNS / bb / arm / "probe" / f"s{s}"
                    plan.append((out / "result.json",
                                 eval_cmd(bb, "probe", arm_flag[arm], out, ["--seed", str(s)])))
        if "finetune" in steps:
            for arm in arms:
                for fr in fracs:
                    for s in ft_seeds:
                        out = RUNS / bb / arm / f"ft{_pct(fr)}" / f"s{s}"
                        plan.append((out / "result.json",
                                     eval_cmd(bb, "finetune", arm_flag[arm], out,
                                              ["--seed", str(s), "--train-frac", str(fr)])))

    if "supervised" in steps:
        for fr in fracs:
            for s in ft_seeds:
                out = RUNS / "xresnet_bn" / "supervised" / f"ft{_pct(fr)}" / f"s{s}"
                cmd = py + ["-m", "jepa.eval", "--mode", "supervised",
                            "--config", "jepa/configs/xresnet_supervised.yaml",
                            "--out", str(out), "--seed", str(s), "--train-frac", str(fr)]
                plan.append((out / "result.json", cmd))
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="vit,cnn,xresnet")
    ap.add_argument("--arms", default="jepa,scratch")
    ap.add_argument("--fracs", default="0.01,0.05,0.1,1.0")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="graines du fine-tuning/supervisé")
    ap.add_argument("--probe-seeds", default="0,1,2")
    ap.add_argument("--steps", default="pretrain,probe,finetune,supervised")
    ap.add_argument("--force", action="store_true", help="recalcule même si le résultat existe")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan = build_plan(
        a.backbones.split(","), a.arms.split(","),
        [float(x) for x in a.fracs.split(",")],
        [int(x) for x in a.seeds.split(",")], [int(x) for x in a.probe_seeds.split(",")],
        set(a.steps.split(",")))

    todo = [(m, c) for m, c in plan if not _skip(m, a.force)]
    done = len(plan) - len(todo)
    print(f"Plan : {len(plan)} runs · {done} déjà faits · {len(todo)} à lancer")
    if a.dry_run:
        for m, c in plan:
            flag = "SKIP" if _skip(m, a.force) else "RUN "
            print(f"  [{flag}] {_name(c)}")
        return

    ok = fail = 0
    for i, (marker, cmd) in enumerate(todo, 1):
        print(f"[{i:>3}/{len(todo)}] {_name(cmd)}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode == 0:
            ok += 1
        else:
            fail += 1
            print(f"   ✗ ÉCHEC (code {r.returncode}) — on continue.", flush=True)
    print(f"\nTerminé : {ok} ok · {fail} échec(s). Agréger : python -m jepa.aggregate")


def _name(cmd) -> str:
    """Nom lisible d'un run = sa valeur --out (ou la config pour le pretrain)."""
    if "--out" in cmd:
        return cmd[cmd.index("--out") + 1]
    return " ".join(cmd[2:])


if __name__ == "__main__":
    main()
