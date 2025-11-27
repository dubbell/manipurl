import click
from manipurl.experiments import train_sac
from dotenv import load_dotenv
from manipurl.experiments.parallel import sweep


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


@click.group()
def cli():
    """manipurl: Command line interface for running RL experiments in RLBench."""
    pass

@cli.command()
@click.argument('algorithm', type=click.Choice(['sac']))
@click.option('--seeds', default="0,1,2,3,4,5,6,7,8,9", help="Comma separated list of seeds to sweep.")
@click.option('--tasks', default="button-press-v3", help="Comma separated list of tasks to sweep.")
@click.option('--n_episodes', default=500, help="Number of training episodes.")
@click.option('--max_episode_step', default=500, help="Max number of environment steps per episode.")
def train(algorithm, seeds, tasks, n_episodes, max_episode_step):
    """Start experiments for specified algorithm."""
    try:
        seeds = [int(seed) for seed in seeds.split(',')]
    except:
        click.echo("Seeds can only be integers.")
        return
    
    for task in tasks.split(','):
        if task not in TEST_SUITE:
            click.echo(f"Task {task} not available.")
            return

    # execute sweep
    if algorithm.lower() == 'sac':
        sweep(train_sac, tasks.split(','), seeds, n_episodes, max_episode_step)
    else:
        click.echo(f"Algorithm {algorithm} not available.")

