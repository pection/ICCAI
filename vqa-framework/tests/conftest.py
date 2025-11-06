# Taken from docs: https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runhuman",
        action="store_true",
        default=False,
        help="run tests requiring a human",
    )
    parser.addoption(
        "--runglobaldataset",
        action="store_true",
        default=False,
        help="run tests that use global dataset storage location -- dangerous & slow!",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "human: mark test as requiring a human")
    config.addinivalue_line(
        "markers",
        "globaldataset: mark test as using the global dataset location -- therefore risk corrupting it!",
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_human = pytest.mark.skip(reason="need --runhuman option to run")
    skip_globaldataset = pytest.mark.skip(
        reason="need --runglobaldataset option to run"
    )
    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "human" in item.keywords and not config.getoption("--runhuman"):
            item.add_marker(skip_human)
        if "globaldataset" in item.keywords and not config.getoption(
            "--runglobaldataset"
        ):
            item.add_marker(skip_globaldataset)
