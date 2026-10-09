from dataclasses import replace
import pytest

from brain import Connectome
from config import Config


@pytest.fixture
def config(tmp_path):
    return replace(Config(), results_path=tmp_path / "episodes.csv")


@pytest.fixture
def demo(config):
    return Connectome.load_from_json(config.connectome_path)
