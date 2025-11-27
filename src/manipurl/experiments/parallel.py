import click
import torch.multiprocessing as mp



def sweep(algorithm, tasks, seeds, n_episodes, max_episode_step):
    mp.set_start_method('spawn', force=True)
    
    click.echo("Starting experiments...")
    processes = []
    for task_name in tasks:
        for seed in seeds:
            process = mp.Process(
                target=algorithm.start_training,
                args=(task_name, seed, n_episodes, max_episode_step, False))
            
            processes.append(process)
            process.start()
    
    click.echo(f"{len(tasks) * len(seeds)} experiments started.")
    
    for process in processes:
        process.join()

    