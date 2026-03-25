# main.py
"""
Orquestador principal del proyecto ATSC Offline-to-Online.

Uso:
  python main.py --phase all
  python main.py --phase 1
  python main.py --phase 2
  python main.py --phase 3
  python main.py --phase sim     # Solo simulacion sintetica (sin SUMO)
  python main.py --check         # Solo verificar el entorno
"""

from __future__ import annotations

import os
import sys
import platform
import argparse
import yaml
from pathlib import Path
from types import SimpleNamespace


def _setup_sumo_windows():
    if platform.system() != "Windows":
        return
    if os.environ.get("SUMO_HOME"):
        tools = str(Path(os.environ["SUMO_HOME"]) / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        return
    candidates = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
    ]
    for c in candidates:
        if Path(c).exists() and (Path(c) / "bin" / "sumo.exe").exists():
            os.environ["SUMO_HOME"] = c
            tools = str(Path(c) / "tools")
            if tools not in sys.path:
                sys.path.insert(0, tools)
            print(f"  [AUTO] SUMO_HOME detectado: {c}")
            return

_setup_sumo_windows()


def load_config(path="configs/config.yaml"):
    config_path = Path(path)
    if not config_path.exists():
        print(f"[ERROR] No se encontro config: {path}")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    flat = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[section] = values
    cfg = SimpleNamespace(**flat)

    # Aliases sin ambiguedad para claves que colisionan entre secciones
    cfg.dataset_output_dir     = raw.get("dataset",        {}).get("output_dir",    "data/offline_dataset")
    cfg.offline_checkpoint_dir = raw.get("offline_train",  {}).get("checkpoint_dir","checkpoints/offline")
    cfg.offline_lr             = raw.get("offline_train",  {}).get("lr",            1e-4)
    cfg.offline_num_epochs     = raw.get("offline_train",  {}).get("num_epochs",    100)
    cfg.offline_batch_size     = raw.get("offline_train",  {}).get("batch_size",    64)
    cfg.online_checkpoint_dir  = raw.get("online_finetune",{}).get("checkpoint_dir","checkpoints/online")
    cfg.online_lr              = raw.get("online_finetune",{}).get("lr",            5e-5)
    cfg.online_num_episodes    = raw.get("online_finetune",{}).get("num_episodes",  200)
    cfg.online_load_from       = raw.get("online_finetune",{}).get("load_from",     "checkpoints/offline/best_model.pt")
    cfg.metrics_output_dir     = raw.get("metrics",        {}).get("output_dir",    "results")

    for attr in ("net_file", "route_file", "additional_file",
                 "dataset_output_dir", "offline_checkpoint_dir",
                 "online_checkpoint_dir", "online_load_from"):
        val = getattr(cfg, attr, None)
        if val and isinstance(val, str):
            setattr(cfg, attr, val.replace("\\", "/"))
    return cfg


def _autodetect_lemgo(cfg):
    import re
    net_ok   = Path(getattr(cfg, "net_file",   "")).exists()
    route_ok = Path(getattr(cfg, "route_file", "")).exists()
    if net_ok and route_ok:
        return cfg
    root = Path("data/lemgo")
    if not root.exists():
        return cfg
    changes = {}
    if not net_ok:
        found = sorted(root.rglob("*.net.xml"))
        if found:
            cfg.net_file = found[0].as_posix()
            changes["net_file"] = cfg.net_file
            print(f"  [AUTO] net_file detectado: {cfg.net_file}")
    if not route_ok:
        found = sorted(root.rglob("*.rou.xml"))
        if found:
            cfg.route_file = found[0].as_posix()
            changes["route_file"] = cfg.route_file
            print(f"  [AUTO] route_file detectado: {cfg.route_file}")
    if changes:
        config_path = "configs/config.yaml"
        content = Path(config_path).read_text(encoding="utf-8")
        for key, value in changes.items():
            content = re.sub(
                rf"^(\s*{key}:\s*).*$",
                "\\g<1>"" + value + """,
                content, flags=re.MULTILINE
            )
        Path(config_path).write_text(content, encoding="utf-8")
        print("  [AUTO] config.yaml actualizado.")
    return cfg


def run_phase1(cfg):
    print("\n" + "=" * 60)
    print("FASE I: Cimentacion del Marco y Pre-entrenamiento Offline")
    print("=" * 60)
    from utils.preflight import run_preflight
    run_preflight(net_file=cfg.net_file, route_file=cfg.route_file)
    from phase1.env.sumo_env import SUMOTrafficEnv
    from phase1.data.dataset_generator import OfflineDatasetGenerator
    print("\n[1A] Generando dataset offline...")
    env = SUMOTrafficEnv(
        net_file=cfg.net_file,
        route_file=cfg.route_file,
        additional_file=getattr(cfg, "additional_file", None),
        step_length=getattr(cfg, "step_length", 1.0),
        episode_length=getattr(cfg, "episode_length", 3600),
        sumo_binary=getattr(cfg, "sumo_binary", "sumo"),
        min_green=getattr(cfg, "min_green", 5),
        yellow_time=getattr(cfg, "yellow_time", 3),
        max_green=getattr(cfg, "max_green", 60),
    )
    generator = OfflineDatasetGenerator(
        env=env,
        output_dir=cfg.dataset_output_dir,
    )
    generator.generate(
        num_fixed_episodes=getattr(cfg, "num_episodes_fixed", 500),
        num_actuated_episodes=getattr(cfg, "num_episodes_actuated", 500),
        save_every=getattr(cfg, "save_every", 50),
    )
    print("\n[1B] Entrenando Decision Transformer offline...")
    import subprocess, torch
    subprocess.run([
        sys.executable, "phase1/train_offline.py",
        "--dataset_dir",     cfg.dataset_output_dir,
        "--checkpoint_dir",  cfg.offline_checkpoint_dir,
        "--num_tls",         str(getattr(cfg, "num_intersections", 4)),
        "--num_epochs",      str(cfg.offline_num_epochs),
        "--batch_size",      str(cfg.offline_batch_size),
        "--lr",              str(cfg.offline_lr),
        "--steps_per_epoch", str(getattr(cfg, "steps_per_epoch", 500)),
        "--device",          "cuda" if torch.cuda.is_available() else "cpu",
    ], check=True)
    print("\nFase I completada.")


def run_phase2(cfg):
    print("\n" + "=" * 60)
    print("FASE II: Seguridad y Coordinacion Multi-Agente")
    print("=" * 60)
    from utils.preflight import run_preflight
    run_preflight(net_file=cfg.net_file, route_file=cfg.route_file)
    from phase1.env.sumo_env import SUMOTrafficEnv
    from phase2.safety.action_mask import SafeEnvironmentWrapper
    from phase2.multiagent.corridor_coordinator import CorridorGraph, CorridorCoordinator
    base_env = SUMOTrafficEnv(
        net_file=cfg.net_file, route_file=cfg.route_file,
        step_length=getattr(cfg, "step_length", 1.0),
        episode_length=getattr(cfg, "episode_length", 3600),
        sumo_binary=getattr(cfg, "sumo_binary", "sumo"),
    )
    safe_env = SafeEnvironmentWrapper(base_env, num_phases=4,
        min_green=getattr(cfg, "min_green", 5),
        min_intergreen=getattr(cfg, "min_intergreen", 3),
        max_consecutive_red=getattr(cfg, "max_consecutive_red", 120),
    )
    num_tls = getattr(cfg, "num_intersections", 4)
    graph = CorridorGraph(
        tl_ids=[f"TL{i}" for i in range(num_tls)],
        distances=[200.0] * (num_tls - 1),
    )
    coordinator = CorridorCoordinator(graph, obs_dim_per_tl=25)
    print(f"  Offsets onda verde: {coordinator.suggest_green_wave_offsets()}")
    obs, _ = safe_env.reset()
    total_v = 0
    for _ in range(100):
        obs, _, t, tr, info = safe_env.step(base_env.action_space.sample())
        total_v += info.get("safety_violations", 0)
        if t or tr: break
    print(f"  Violaciones interceptadas en 100 pasos aleatorios: {total_v}")
    safe_env.env.close()
    print("\nFase II completada.")


def run_phase3(cfg):
    print("\n" + "=" * 60)
    print("FASE III: Adaptacion Online y Benchmarking")
    print("=" * 60)
    from utils.preflight import run_preflight
    run_preflight(net_file=cfg.net_file, route_file=cfg.route_file)
    import torch
    from phase1.env.sumo_env import SUMOTrafficEnv
    from phase1.models.decision_transformer import DecisionTransformer
    from phase2.safety.action_mask import SafeEnvironmentWrapper
    from phase3.online.online_finetuner import OnlineFineTuner
    from phase3.metrics.metrics_extractor import Benchmarker
    import glob, re
    # Detectar si hay un checkpoint online para reanudar
    online_ckpts = sorted(glob.glob(
        str(Path(cfg.online_checkpoint_dir) / "ep_*.pt")
    ))
    if online_ckpts:
        load_from   = online_ckpts[-1]
        resume_ep   = int(re.search(r"ep_(\d+)", load_from).group(1)) + 1
        resume_mode = True
        print(f"  [RESUME] Reanudando desde: {load_from} (ep {resume_ep})")
    else:
        load_from   = cfg.online_load_from
        resume_ep   = 1
        resume_mode = False
        if not Path(load_from).exists():
            print(f"[ERROR] Checkpoint no encontrado: {load_from}")
            print("        Ejecuta la Fase I primero.")
            sys.exit(1)

    ckpt = torch.load(load_from, map_location="cpu")
    model_cfg = ckpt.get("cfg", {})
    num_tls = getattr(cfg, "num_intersections", 4)
    model = DecisionTransformer(
        obs_dim=25 * num_tls, act_dim=4 * num_tls, num_tls=num_tls,
        context_length=model_cfg.get("context_length", 20),
        d_model=model_cfg.get("d_model", 128),
        n_layer=model_cfg.get("n_layer", 4),
        n_head=model_cfg.get("n_head", 4),
        d_inner=model_cfg.get("d_inner", 512),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    base_env = SUMOTrafficEnv(net_file=cfg.net_file, route_file=cfg.route_file,
        step_length=getattr(cfg, "step_length", 1.0),
        episode_length=getattr(cfg, "episode_length", 3600),
        sumo_binary=getattr(cfg, "sumo_binary", "sumo"),
    )
    safe_env = SafeEnvironmentWrapper(base_env)
    finetuner = OnlineFineTuner(
        model=model, env=safe_env,
        target_return=getattr(cfg, "target_return", -50.0),
        checkpoint_dir=cfg.online_checkpoint_dir,
    )
    if resume_mode:
        finetuner.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    finetuner.run(num_episodes=cfg.online_num_episodes, start_episode=resume_ep)
    benchmarker = Benchmarker(safe_env, output_dir=cfg.metrics_output_dir)
    benchmarker.save_results()
    benchmarker.print_summary_table()
    benchmarker.plot_comparison()
    print("\nFase III completada.")


def run_simulation():
    print("\n" + "=" * 60)
    print("SIMULACION SINTETICA (sin SUMO requerido)")
    print("=" * 60)
    import subprocess
    sim = Path("simulation")
    if not sim.exists():
        print("[ERROR] Carpeta simulation/ no encontrada.")
        sys.exit(1)
    subprocess.run([sys.executable, str(sim / "run_simulation.py")], check=True)
    subprocess.run([sys.executable, str(sim / "plot_results.py")], check=True)
    print("\nSimulacion completada.")
    print("  Datos:   simulation/results/")
    print("  Figuras: simulation/results/figures/")


def main():
    p = argparse.ArgumentParser(description="ATSC Offline-to-Online DRL")
    p.add_argument("--phase", choices=["1","2","3","all","sim"], default="all")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    if args.check:
        from utils.preflight import run_preflight
        cfg = load_config(args.config)
        cfg = _autodetect_lemgo(cfg)
        run_preflight(net_file=getattr(cfg,"net_file",""),
                      route_file=getattr(cfg,"route_file",""),
                      abort_on_error=False)
        return

    if args.phase == "sim":
        run_simulation()
        return

    cfg = load_config(args.config)
    cfg = _autodetect_lemgo(cfg)

    if args.phase in ("1", "all"): run_phase1(cfg)
    if args.phase in ("2", "all"): run_phase2(cfg)
    if args.phase in ("3", "all"): run_phase3(cfg)


if __name__ == "__main__":
    main()
