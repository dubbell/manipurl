import click
import torch.multiprocessing as mp
from manipurl.experiments.run_config import config_dict_to_configs


def sweep(algorithm, config_dict):
    mp.set_start_method('spawn', force=True)
    
    processes = []
    np = config_dict['runs_per_iteration']  # number of processes to run in parallel per iteration

    all_run_configs = config_dict_to_configs(config_dict)
    offset = 0 if len(all_run_configs) % np == 0 else 1
    iterations = [all_run_configs[i * np : (i+1) * np] for i in range(len(all_run_configs) // np + offset)]

    click.echo(f'Number of experiments: {len(all_run_configs)}')
    click.echo(f'Experiments per iteration: {np}')
    click.echo(f'Number of iterations: {len(iterations)}\n')

    for iter, run_configs in enumerate(iterations):
        click.echo(f'Starting iteration {iter+1}...')

        for run_config in run_configs:
            process = mp.Process(
                target=algorithm.start_training,
                args=(run_config,))
            
            processes.append(process)
            process.start()
        
        click.echo(f"{len(run_configs)} experiments started.")
        
        for process in processes:
            process.join()
    
        click.echo(f'Iteration {iter+1} finished.\n')