import click

from lhotse.bin.modes import prepare
from lhotse.recipes.uniphore_dev import prepare_uniphore_dev
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--normalize-text/--no-normalize-text", default=False, help="Normalize the text."
)
def uniphore_dev(corpus_dir: Pathlike, output_dir: Pathlike, normalize_text: bool):
    """Uniphore data preparation."""
    prepare_uniphore_dev(
        corpus_dir, output_dir=output_dir, normalize_text=normalize_text
    )
