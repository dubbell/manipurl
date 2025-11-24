import click
from manipurl.experiments import train_sac
from dotenv import load_dotenv
from manipurl.experiments.parallel import seed_sweep


load_dotenv()


TEST_SUITE = [
    "reach-v3",

]


@click.group()
def cli():
    """manipurl: Command line interface for running RL experiments in RLBench."""
    pass

@cli.command()
@click.argument('algorithm', type=click.Choice(['sac']))
def train(algorithm):
    """Start experiment for specified algorithm."""

    if algorithm.lower() == 'sac':
        seed_sweep(train_sac, 'drawer-close-v3', 500)
    else:
        click.echo(f"Algorithm {algorithm} not available.")

