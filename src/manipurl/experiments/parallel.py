import click
import torch.multiprocessing as mp
from manipurl.experiments.run_config import RunConfig, config_dict_to_configs


def sweep(algorithm, config_dict):
    mp.set_start_method('spawn', force=True)
    
    click.echo("Starting experiments...")
    processes = []
    run_configs = config_dict_to_configs(config_dict)
    for run_config in run_configs:
        process = mp.Process(
            target=algorithm.start_training,
            args=(run_config,))
        
        processes.append(process)
        process.start()
    
    click.echo(f"{len(run_configs)} experiments started.")
    
    for process in processes:
        process.join()
    
    click.echo(f'Finished.')