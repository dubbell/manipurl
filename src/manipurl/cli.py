import click
from manipurl.experiments import train_sac
from dotenv import load_dotenv


load_dotenv()


@click.group()
def cli():
    """manipurl: Command line interface for running RL experiments in RLBench."""
    pass

@cli.command()
@click.argument('algorithm', type=click.Choice(['sac']))
def train(algorithm):
    """Start experiment for specified algorithm."""

    if algorithm.lower() == 'sac':
        train_sac.start_training("REACH_TARGET", 1000)
    else:
        click.echo(f"Algorithm {algorithm} not available.")