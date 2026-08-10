from unittest.mock import patch

from hearthia.service import (
    DAEMON_LABEL,
    GATEWAY_LABEL,
    UPDATE_LABEL,
    ServiceSpec,
    _plist_xml,
    install_plists,
    render_plists,
    restart_service,
    service_status,
)


def test_plist_xml_contains_required_keys():
    spec = ServiceSpec(
        label="com.hearthia.test",
        program_arguments=["/bin/echo", "hello"],
        log_path=None,
    )
    xml = _plist_xml(spec)
    assert "com.hearthia.test" in xml
    assert "/bin/echo" in xml
    assert "hello" in xml
    assert "<key>RunAtLoad</key><true/>" in xml
    assert "<key>KeepAlive</key><true/>" in xml
    assert "<?xml" in xml
    assert "</plist>" in xml


def test_plist_xml_with_calendar_interval():
    spec = ServiceSpec(
        label="com.hearthia.update",
        program_arguments=["/bin/zsh", "-c", "echo update"],
        calendar_interval={"Weekday": 0, "Hour": 11, "Minute": 0},
        keep_alive=False,
    )
    xml = _plist_xml(spec)
    assert "StartCalendarInterval" in xml
    assert "<key>Weekday</key><integer>0</integer>" in xml
    assert "<key>KeepAlive</key><true/>" not in xml


def test_plist_xml_with_env_and_working_dir():
    spec = ServiceSpec(
        label="com.hearthia.test",
        program_arguments=["python3", "-m", "hearthia.cli"],
        working_directory="/tmp/work",
        environment={"PATH": "/usr/bin", "FOO": "bar"},
    )
    xml = _plist_xml(spec)
    assert "WorkingDirectory" in xml
    assert "/tmp/work" in xml
    assert "EnvironmentVariables" in xml
    assert "FOO" in xml


def test_render_plists_returns_three_services(tmp_path):
    from hearthia.settings import PathsSettings, Settings

    paths = PathsSettings(
        stack_dir=tmp_path,
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    settings = Settings(paths=paths)
    plists = render_plists(settings)
    assert GATEWAY_LABEL in plists
    assert DAEMON_LABEL in plists
    assert UPDATE_LABEL in plists
    assert "llama-swap" in plists[GATEWAY_LABEL]
    assert "hearthia.cli" in plists[DAEMON_LABEL]
    assert "brew upgrade" in plists[UPDATE_LABEL]


def test_render_plists_creates_logs_dir(tmp_path):
    from hearthia.settings import PathsSettings, Settings

    logs_dir = tmp_path / "logs"
    paths = PathsSettings(
        stack_dir=tmp_path,
        models_dir=tmp_path / "models",
        logs_dir=logs_dir,
    )
    settings = Settings(paths=paths)
    render_plists(settings)
    assert logs_dir.exists()


def test_install_plists_reports_bootstrap_failure(tmp_path):
    from hearthia.settings import PathsSettings, Settings

    paths = PathsSettings(
        stack_dir=tmp_path,
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    settings = Settings(paths=paths)
    failed = type("R", (), {"returncode": 5, "stderr": "bootstrap failed"})()

    with (
        patch("hearthia.service.Path.home", return_value=tmp_path),
        patch("hearthia.service.subprocess.run", return_value=failed),
    ):
        try:
            install_plists(settings)
        except RuntimeError as error:
            assert "bootstrap failed" in str(error)
        else:
            raise AssertionError("bootstrap failure was silently ignored")


def test_restart_service_success():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {"returncode": 0, "stderr": ""})()
        assert restart_service("com.hearthia.gateway") is True


def test_restart_service_failure():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {"returncode": 1, "stderr": "fail"})()
        assert restart_service("com.hearthia.gateway") is False


def test_service_status_loaded():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {"returncode": 0, "stdout": ""})()
        assert service_status("com.hearthia.gateway") == "loaded"


def test_service_status_not_loaded():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {"returncode": 1, "stderr": ""})()
        assert service_status("com.hearthia.gateway") == "not loaded"
