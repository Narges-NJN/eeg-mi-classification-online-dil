import os

from config import default_config
from utils import set_seed, make_run_dir, save_json, torch_device
from experiments import run_online_domain_incremental, run_offline_domain_incremental, run_loso
from plots import plot_domain_incremental, plot_loso

def main():
    cfg = default_config()

    set_seed(cfg.seed)
    device = torch_device(cfg.device)

    run_dir = make_run_dir(cfg.out_dir, cfg.run_name)

    # 1) Online DI
    online = run_online_domain_incremental(cfg, device)
    online_dir = os.path.join(run_dir, "online_domain_incremental")
    os.makedirs(online_dir, exist_ok=True)
    save_json(online, os.path.join(online_dir, "results.json"))
    plot_domain_incremental(online, online_dir)

    # 2) Offline DI
    offline = run_offline_domain_incremental(cfg, device)
    offline_dir = os.path.join(run_dir, "offline_domain_incremental")
    os.makedirs(offline_dir, exist_ok=True)
    save_json(offline, os.path.join(offline_dir, "results.json"))
    plot_domain_incremental(offline, offline_dir)

    # 3) LOSO
    loso = run_loso(cfg, device)
    loso_dir = os.path.join(run_dir, "loso")
    os.makedirs(loso_dir, exist_ok=True)
    save_json(loso, os.path.join(loso_dir, "results.json"))
    plot_loso(loso, loso_dir)

    print(f"\nDone. Outputs saved to: {run_dir}")

if __name__ == "__main__":
    main()
