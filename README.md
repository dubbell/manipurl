# manipurl

Robotic manipulation training algorithm implementations. Evaluation with simple Metaworld tasks, implementation with PyTorch.

Recommended setup: Linux, Python 3.9, Anaconda.

MLflow logging:
- PostgreSQL backend: `postgresql+psycopg2://manipurl_user:123@localhost:5432/manipurl`
- Start MLflow server with `bash ./scripts/start_mlflow_server.sh`

Run experiments:
- Install requirements with `pip install -r requirements.txt`
- Install `manipurl` with `pip install -e .`
- (Optional) Define configuration file in `configs`, following `configs/default.yaml` structure.
- (Optional) Enable/disable logging and profiling with environment variables `MANIPURL_ENABLE_LOGGING` and `MANIPURL_PROFILING`.
  - Default for both is `False`.
  - Logs are sent to MLflow, tracked at `http://0.0.0.0:5000`.
- Start experiments with `manipurl train <ALGORITHM> [CONFIGURATION] [OPTIONS]`.
  - `ALGORITHM` is training algorithm, e.g. `sac`.
  - (Optional) `CONFIGURATION` is name of configuration file in `configs`, excluding `.yaml` file extension. Defaults to `default` configuration.
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