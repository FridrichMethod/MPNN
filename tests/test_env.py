from __future__ import annotations

import contextlib
import shutil

from mpnn import env


def test_detect_package_root_dir_preserves_extracted_path(tmp_path, monkeypatch):
    """Simulate a zip-safe install and ensure the extracted path stays alive."""

    fake_root = object()
    extracted_dir = tmp_path / "pkg"
    extracted_dir.mkdir()
    (extracted_dir / "sentinel.txt").write_text("ok", encoding="utf-8")

    def fake_files(package: str) -> object:
        assert package == env.__package__
        return fake_root

    def fake_as_file(resource: object):
        assert resource is fake_root

        @contextlib.contextmanager
        def manager():
            yield extracted_dir
            shutil.rmtree(extracted_dir, ignore_errors=True)

        return manager()

    stack = contextlib.ExitStack()

    monkeypatch.setattr(env, "files", fake_files)
    monkeypatch.setattr(env, "as_file", fake_as_file)
    monkeypatch.setattr(env, "_PACKAGE_ROOT_DIR_EXIT_STACK", stack, raising=False)

    path = env._detect_package_root_dir()
    assert path.exists()
    stack.close()
    assert not path.exists()
