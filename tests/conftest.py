from _pytest.config import Config

markers = [
    "slow: marks tests as slow (deselect with -m 'not slow')",
]

def pytest_configure(config: Config) -> None:
    for m in markers:
        config.addinivalue_line("markers", m)
