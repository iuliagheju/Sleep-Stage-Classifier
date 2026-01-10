from __future__ import annotations

from sleep_stager.utils.system import collect_hardware_summary, collect_package_versions


def test_collect_hardware_summary_keys():
    summary = collect_hardware_summary()
    assert "platform" in summary
    assert "python_version" in summary
    assert "cpu_count" in summary


def test_collect_package_versions():
    versions = collect_package_versions(["numpy", "torch"])
    assert "numpy" in versions
    assert "torch" in versions
