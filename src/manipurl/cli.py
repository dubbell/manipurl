import click
from manipurl.experiments import train_sac
from dotenv import load_dotenv
from manipurl.experiments.parallel import sweep
import os
import yaml


load_dotenv()


TEST_SUITE = [
    "button-press-v3",
    "door-open-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "peg-insert-side-v3",
    "pick-place-v3",
    "push-v3",
    "reach-v3",
    "window-open-v3",
    "window-close-v3"]


def available_configs():
    return [
        filename.split('.yaml')[0]
        for filename in os.listdir('configs')
        if '.yaml' in filename]

def get_config(config_name):
    filepath = f'configs/{config_name}.yaml'
    if not os.path.exists(filepath):
        click.echo(f'Configuration file {config_name}.yaml not found.')
    else:
        with open(filepath, 'r') as file:
            try:
                data_dict = yaml.safe_load(file)
                return data_dict
            except yaml.YAMLError as exc:
                click.echo(f'Error parsing YAML file: {exc}')
    
    return {}

def unfold_seeds(seed_str):
    try:
        sections = [[int(seed) for seed in section.split(',')] for section in seed_str.split('..')]
    except:
        click.echo("Seeds can only be integers.")
        return
    for s_i in range(len(sections) - 1):
        sections[s_i].extend(list(range(sections[s_i][-1] + 1, sections[s_i+1][0])))

    return [seed for section in sections for seed in section]

def unfold_tasks(task_str):
    if task_str == 'all':
        return TEST_SUITE
    else:
        tasks = task_str.split(',')
        for task in tasks:
            if task not in TEST_SUITE:
                click.echo(f"Task {task} not available.")
                return
        return tasks


@click.group()
def cli():
    """manipurl: Command line interface for running RL experiments in RLBench."""
    pass

@cli.command()
@click.argument('algorithm', type=click.Choice(['sac']))
@click.argument('config', type=click.Choice(available_configs()), required=False, default='default')
@click.option('--seeds', help="Comma separated list of seeds to sweep.")
@click.option('--tasks', help="Comma separated list of tasks to sweep. `all` for all tasks.")
@click.option('--n_episodes', help="Number of training episodes.", type=int)
@click.option('--max_episode_step', help="Max number of environment steps per episode.", type=int)
@click.option('--start_training', help="Environment step for starting training.", type=int)
@click.option('--eval_freq', help="Evaluation frequency (number of episodes).", type=int)
@click.option('--eval_eps', help="Number of episodes per evaluation.", type=int)
@click.option('--runs_per_iteration', help="Number of experiments to run in parallel.", type=int)
def train(algorithm, config, seeds, tasks, n_episodes, max_episode_step, start_training, eval_freq, eval_eps, runs_per_iteration):
    """Start experiments for specified algorithm."""
    options = { key : value for key, value in {
            "algorithm" : algorithm, 
            "config" : config, 
            "seeds" : seeds, 
            "tasks" : tasks, 
            "n_episodes" : n_episodes, 
            "max_episode_step" : max_episode_step, 
            "start_training" : start_training, 
            "eval_freq" : eval_freq, 
            "eval_eps" : eval_eps, 
            "runs_per_iteration" : runs_per_iteration}.items()
        if value is not None }

    config_dict = get_config('default')
    config_dict.update(get_config(config))
    config_dict.update(options)

    config_dict['seeds'] = unfold_seeds(config_dict['seeds'])
    if config_dict['seeds'] is None:
        return

    config_dict['tasks'] = unfold_tasks(config_dict['tasks'])
    if config_dict['tasks'] is None:
        return

    # execute sweep
    if algorithm.lower() == 'sac':
        sweep(train_sac, config_dict)
    else:
        click.echo(f"Algorithm {algorithm} not available.")

