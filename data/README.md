# Data Directory

## Directory Structure

```
data/
├── README.md
├── scripts/               # Download scripts for external datasets
│   ├── download_kienzle_50k.py
│   ├── download_kienzle_120k.py
│   ├── download_aimy.py
│   ├── download_roboflow_tt.py
│   └── download_blurball.py
├── generated/             # Auto-generated outputs (git-ignored except .gitkeep)
│   ├── trajectories/      # Simulated trajectory batches
│   ├── eval_runs/         # Evaluation run outputs
│   ├── rl_episodes/       # RL training episodes
│   ├── optimized_swings/  # Swing optimization results
│   └── laser_sim/         # Laser simulation data
├── kienzle_50k/           # Kienzle 50k spin+trajectory dataset
├── kienzle_120k/          # Kienzle 120k trajectory dataset
├── aimy/                  # AIMY real launcher trajectories (HDF5)
├── roboflow_tt/           # Roboflow TT ball detection (YOLO format)
└── blurball/              # BlurBall motion blur annotations
```

## External Datasets

Download each dataset with its corresponding script:

```bash
python data/scripts/download_kienzle_50k.py
python data/scripts/download_kienzle_120k.py
python data/scripts/download_aimy.py
python data/scripts/download_roboflow_tt.py   # requires ROBOFLOW_API_KEY
python data/scripts/download_blurball.py
```

All scripts are **idempotent** — they skip downloading if the output directory already contains files.
Pass `--help` for details on each script.

| Dataset | Size | Format | Source |
|---------|------|--------|--------|
| `kienzle_50k` | ~50k trajectories | CSV/NumPy | [SpinAndTrajectoryTableTennis](https://github.com/KieDani/SpinAndTrajectoryTableTennis) |
| `kienzle_120k` | ~120k trajectories | CSV/NumPy | [UpliftingTableTennis](https://github.com/KieDani/UpliftingTableTennis) |
| `aimy` | Real launcher data | HDF5 | [MPI WebDAV](https://webdav.tuebingen.mpg.de/aimy/) |
| `roboflow_tt` | 1,110 images | YOLO | [Roboflow](https://universe.roboflow.com/computer-vision-project-mjsdu/table-tennis-ball-um5tc) |
| `blurball` | Motion blur annotations | Images + labels | [Uni Tübingen Cloud](https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7) |

## Dataset → Subsystem Mapping

| Dataset | Used By |
|---------|---------|
| `kienzle_50k` | Physics model fitting, spin estimation validation |
| `kienzle_120k` | Trajectory prediction training, aerodynamics calibration |
| `aimy` | Real-world trajectory validation, launcher modeling |
| `roboflow_tt` | Ball detection (YOLO training/eval), vision pipeline |
| `blurball` | Motion blur–aware detection, vision preprocessing |
| `generated/trajectories` | Simulation outputs from physics engine |
| `generated/eval_runs` | End-to-end evaluation pipeline |
| `generated/rl_episodes` | RL policy training (swing optimization) |
| `generated/optimized_swings` | Swing optimization results |
| `generated/laser_sim` | Laser tracker simulation data |

## Generated Data

The `generated/` directory is populated by running simulation and training pipelines.
Contents are git-ignored (only `.gitkeep` placeholders are tracked).
See the main project README for commands to generate this data.
