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
        'eval_eps',
        'runs_per_iteration',
        'task_description'
    ])

TASK_TO_TASK_DESCRIPTION = {
    "button-press-v3": "press the red button",
    "door-open-v3": "open the door",
    "drawer-close-v3": "close the drawer",
    "drawer-open-v3": "open the drawer",
    "peg-insert-side-v3": "insert the peg into the wall",
    "pick-place-v3": "pick up the puck and place it on the goal",
    "push-v3": "push the puck to the goal",
    "reach-v3": "reach towards the goal",
    "window-open-v3": "open the window",
    "window-close-v3": "close the window"
}

def config_dict_to_configs(config_dict):
    return [
        RunConfig(
            task, 
            seed, 
            config_dict['n_episodes'], 
            config_dict['max_episode_step'], 
            config_dict['start_training'], 
            config_dict['eval_freq'], 
            config_dict['eval_eps'],
            config_dict['runs_per_iteration'],
            TASK_TO_TASK_DESCRIPTION[task])
        
        for task in config_dict['tasks']
        for seed in config_dict['seeds']]