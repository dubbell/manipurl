# manipurl

Robotic manipulation training algorithm implementations for sparse-reward environments. Evaluated with simple Metaworld tasks, implemented with PyTorch.

Recommended setup: Linux (Ubuntu 24.04), Anaconda, Python 3.10.

MLflow logging:
- PostgreSQL backend: `postgresql+psycopg2://manipurl_user:123@localhost:5432/manipurl`
  - DB name: `manipurl`
  - DB user with perms for `manipurl`: `manipurl_user`
  - Password for `manipurl_user`: `123`
  - (If you want to change these settings you'll have to change the startup script described below.)
- Start MLflow server with `bash ./scripts/start_mlflow_server.sh`

Run experiments:
- Create Anaconda environment, e.g. `conda create -n manipurl python=3.10`
- Install required packages with `pip install -r requirements.txt` (could be somewhat bloated).
- Install `manipurl` with `pip install -e .`
- (Optional) Define configuration file in `configs`, following `configs/default.yaml` structure.
- (Optional) Enable/disable logging and profiling with environment variables `MANIPURL_ENABLE_LOGGING` and `MANIPURL_PROFILING`.
  - Default for both is `False`.
  - Logs are sent to MLflow, tracked at `http://0.0.0.0:5000`.
  - Profiling is performed with `cProfile` and results are stored in `data/profiling` as `.prof` files. Can be analyzed with `snakeviz`.
- Start experiments with `manipurl train <ALGORITHM> [CONFIGURATION] [OPTIONS]`.
  - `ALGORITHM` is the RL training algorithm, e.g. `sac`. Currently available training algorithms:
    - `sac`
  - (Optional) `CONFIGURATION` is the name of a configuration file in `configs`, excluding `.yaml` file extension. Defaults to `default` configuration.
  - `OPTIONS` can overwrite options defined in configuration file with corresponding flag:
    - `--seeds`
      - Randomization seeds to run. Integers separated by commas or `..` in string. E.g. `"1..5"`, `"1,2,3,4,5"`, `"1,2..5"`, etc. Default `"0..9"`.
    - `--tasks`
      - Metaworld tasks, comma-separated list. Default `"button-press-v3"`. Available tasks:
        - `button-press-v3`
        - `door-open-v3`
        - `drawer-close-v3`
        - `drawer-open-v3`
        - `peg-insert-side-v3`
        - `pick-place-v3`
        - `push-v3`
        - `reach-v3`
        - `window-open-v3`
        - `window-close-v3`
    - `--n_episodes`
      - Number of training episodes. Default 500.
    - `--max_episode_step`
      - Maximum number of environment steps per episode. Default 500.
    - `--start_training`
      - Environment step at which training will start. Default 2500.
    - `--eval_freq`
      - Frequency of evaluation, in number of episodes. Default 50.
    - `--eval_eps`
      - Number of episodes per evaluation. Default 25.
    - `--runs_per_iteration`
      - Number of runs to run in parallel. If total number of runs (num seeds * num tasks) exceeds `runs_per_iteration`, then multiple iterations are executed.