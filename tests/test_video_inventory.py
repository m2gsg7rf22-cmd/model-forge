from pathlib import Path
from modelforge.video_inventory import inspect_fastmetal


def test_missing_bundle_never_claims_generation_ready(tmp_path: Path) -> None:
    result = inspect_fastmetal(tmp_path / 'missing')
    assert not result['known_files_present']
    assert not result['model_integrity_verified']
    assert not result['generation_verified']
    assert 'mlx_dit.safetensors' in result['missing_or_empty_files']


def test_known_files_are_only_inventory_not_execution_proof(tmp_path: Path) -> None:
    from modelforge.video_inventory import KNOWN_FILES
    for name in KNOWN_FILES:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'x')
    result = inspect_fastmetal(tmp_path)
    assert result['known_files_present']
    assert not result['model_integrity_verified']
    assert not result['generation_verified']
    (tmp_path / KNOWN_FILES[0]).write_bytes(b'')
    assert not inspect_fastmetal(tmp_path)['known_files_present']
