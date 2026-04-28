import pathlib

import pytest


syside = pytest.importorskip("syside")

from src.sysml.Syside_AST_Parser import parse_sysml_to_model
from src.sysml.model import SysMLModel


def test_parse_sysml_to_model_with_example_file() -> None:
    example_path = pathlib.Path(
        __file__
    ).resolve().parents[1] / "src" / "rag" / "SysML-v2-release-src" / "examples" / "Import Tests" / "AliasImport.sysml"

    if not example_path.exists():
        pytest.skip(f"SysML example file not found: {example_path}")

    sysml_text = example_path.read_text(encoding="utf-8")
    model = parse_sysml_to_model(sysml_text, model_name="AliasImportTest", strict=False)

    assert isinstance(model, SysMLModel)
    assert model.name == "AliasImportTest"
    assert hasattr(model, "diagnostics")

