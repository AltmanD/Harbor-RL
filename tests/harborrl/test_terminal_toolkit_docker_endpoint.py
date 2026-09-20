from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from harborrl.environments.terminal.runtime import _align_toolkit_docker_client


def test_exec_uses_same_api_as_container_lookup():
    api = Mock()
    api.exec_create.return_value = {"Id": "exec-1"}
    api.exec_inspect.return_value = {"ExitCode": 0}
    previous = Mock()
    toolkit = SimpleNamespace(
        docker_client=SimpleNamespace(api=api),
        docker_api_client=previous,
        container=SimpleNamespace(id="private-container"),
        docker_workdir="/workspace with spaces",
    )
    _align_toolkit_docker_client(toolkit)
    assert toolkit.docker_api_client is api
    previous.close.assert_called_once_with()
    api.exec_create.assert_called_once_with(
        "private-container", ["mkdir", "-p", "--", "/workspace with spaces"]
    )
    previous.exec_create.assert_not_called()


def test_already_aligned_client_is_not_closed():
    api = Mock()
    toolkit = SimpleNamespace(
        docker_client=SimpleNamespace(api=api), docker_api_client=api
    )
    _align_toolkit_docker_client(toolkit)
    api.close.assert_not_called()


def test_workdir_failure_is_not_hidden():
    api = Mock()
    api.exec_create.return_value = {"Id": "exec-1"}
    api.exec_inspect.return_value = {"ExitCode": 1}
    toolkit = SimpleNamespace(
        docker_client=SimpleNamespace(api=api),
        docker_api_client=Mock(),
        container=SimpleNamespace(id="private-container"),
        docker_workdir="/workspace",
    )
    with pytest.raises(RuntimeError, match="workdir"):
        _align_toolkit_docker_client(toolkit)
