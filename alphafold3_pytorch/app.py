from pathlib import Path

import click
import gradio as gr

from alphafold3_pytorch.models.components.alphafold3 import Alphafold3
from alphafold3_pytorch.models.components.inputs import Alphafold3Input

# constants

model = None

# main fold function


def fold(protein):
    """Fold a biomolecule using AlphaFold 3."""
    alphafold3_input = Alphafold3Input(proteins=[protein])

    model.eval()
    (atom_pos,) = model.forward_with_alphafold3_inputs(alphafold3_input)

    return str(atom_pos.tolist())


# gradio

gradio_app = gr.Interface(
    fn=fold,
    inputs=["text"],
    outputs=["text"],
)

# cli


@click.command()
@click.option(
    "-ckpt", "--checkpoint", type=str, help="Path to AlphaFold 3 checkpoint", required=True
)
def app(checkpoint: str):
    """A Gradio app for folding biomolecules using AlphaFold 3."""
    path = Path(checkpoint)
    assert path.exists(), "checkpoint does not exist at path"

    global model
    model = Alphafold3.init_and_load(str(path))

    gradio_app.launch()
