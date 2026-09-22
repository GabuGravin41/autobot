"""
Offline tests for the kernel methods added to autobot/computer/kaggle_tool.py
(pull_kernel, kernel_status, kernel_output, push_kernel).

`Kaggle._get_api()` is mocked directly rather than patching the `kaggle`
package's internals — this sandbox doesn't have the `kaggle` pip package
installed (it's a real dependency of the user's actual machine, not of
these tests), and mocking at `_get_api()` means these tests don't care
either way. They verify: the right underlying KaggleApi method is called
with the right arguments, results are formatted as plain strings (what
computer_call's dispatch.py expects back), and push_kernel's
kernel-metadata.json precondition is enforced before ever touching the API.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autobot.computer.kaggle_tool import Kaggle


def _kaggle_with_mock_api() -> tuple[Kaggle, MagicMock]:
    k = Kaggle()
    mock_api = MagicMock()
    k._api = mock_api   # bypass _get_api()'s real KaggleApi() + authenticate()
    return k, mock_api


class TestPullKernel:
    def test_calls_kernels_pull_with_metadata(self, tmp_path):
        k, api = _kaggle_with_mock_api()
        result = k.pull_kernel("user/my-kernel", str(tmp_path / "work"))
        api.kernels_pull.assert_called_once_with("user/my-kernel", str(tmp_path / "work"), metadata=True)
        assert "user/my-kernel" in result
        assert (tmp_path / "work").exists()

    def test_missing_kernel_raises(self):
        k, api = _kaggle_with_mock_api()
        with pytest.raises(ValueError, match="kernel slug"):
            k.pull_kernel("", "./somewhere")
        api.kernels_pull.assert_not_called()


class TestKernelStatus:
    def test_returns_status_as_string(self):
        k, api = _kaggle_with_mock_api()
        api.kernels_status.return_value = "complete"
        result = k.kernel_status("user/my-kernel")
        assert result == "complete"
        api.kernels_status.assert_called_once_with("user/my-kernel")

    def test_missing_kernel_raises(self):
        k, api = _kaggle_with_mock_api()
        with pytest.raises(ValueError):
            k.kernel_status("")


class TestKernelOutput:
    def test_calls_kernels_output(self, tmp_path):
        k, api = _kaggle_with_mock_api()
        out_dir = tmp_path / "out"
        result = k.kernel_output("user/my-kernel", str(out_dir))
        api.kernels_output.assert_called_once_with("user/my-kernel", str(out_dir))
        assert out_dir.exists()
        assert "downloaded" in result


class TestPushKernel:
    def test_blocks_without_metadata_file(self, tmp_path):
        k, api = _kaggle_with_mock_api()
        with pytest.raises(FileNotFoundError, match="kernel-metadata.json"):
            k.push_kernel(str(tmp_path))
        api.kernels_push.assert_not_called()   # never touches the API

    def test_pushes_when_metadata_present(self, tmp_path):
        k, api = _kaggle_with_mock_api()
        (tmp_path / "kernel-metadata.json").write_text("{}")
        api.kernels_push.return_value = "Kernel version 3 successfully pushed"
        result = k.push_kernel(str(tmp_path))
        api.kernels_push.assert_called_once_with(str(tmp_path))
        assert "pushed" in result

    def test_empty_path_raises(self):
        k, api = _kaggle_with_mock_api()
        with pytest.raises(ValueError):
            k.push_kernel("")
