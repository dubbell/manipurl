import torch.multiprocessing as mp


def seed_sweep(experiment, task_name, num_episodes, seeds = list(range(10))):
    mp.set_start_method('spawn', force=True)

    processes = []
    for seed in seeds:
        process = mp.Process(
            target=experiment.start_training,
            args=(task_name, num_episodes, seed, False))
        
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
