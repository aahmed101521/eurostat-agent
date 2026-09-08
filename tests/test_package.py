from importlib.metadata import version

import eurostat_agent


def test_package_version() -> None:
    assert eurostat_agent.__version__ == version("eurostat-agent")
