from collections import namedtuple


RunConfig = namedtuple(
    'RunConfig', 
    [
        'task', 
        'seed', 
        'n_episodes', 
        'max_episode_step', 
        'start_training',
        'eval_freq',
        'eval_eps'
    ])

def config_dict_to_configs(config_dict):
    return [
        RunConfig(
            task, 
            seed, 
            config_dict['n_episodes'], 
            config_dict['max_episode_step'], 
            config_dict['start_training'], 
            config_dict['eval_freq'], 
            config_dict['eval_eps'])
        
        for task in config_dict['tasks']
        for seed in config_dict['seeds']
    ]