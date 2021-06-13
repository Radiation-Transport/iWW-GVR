import pytest

from iww_gvr.utils.resource import Path, filename_resolver, path_resolver

THIS_FILENAME = Path(__file__).name


@pytest.mark.parametrize(
    "package, resource, expected",
    [
        (None, THIS_FILENAME, THIS_FILENAME),
        ("tests", "utils/data/empty.txt", "/tests/utils/data/empty.txt"),
        ("tests.utils", "data/empty.txt", "/tests/utils/data/empty.txt"),
    ],
)
def test_filename_resolver(package, resource, expected):
    resolver = filename_resolver(package)
    actual = resolver(resource)
    if "\\" in actual:
        actual = actual.replace("\\", "/")
    assert actual.endswith(expected), "Failed to compute resource file name"
    assert Path(actual).exists(), f"The resource '{resource}' is not available"


@pytest.mark.parametrize(
    "package, resource, expected",
    [
        (None, "not_existing.py", "not_existing.py"),
        ("tests", "utils/data/not_existing", "tests/utils/data/not_existing"),
        ("tests.utils", "data/not_existing", "mckit/data/not_existing"),
    ],
)
def test_filename_resolver_when_resource_doesnt_exist(package, resource, expected):
    resolver = filename_resolver(package)
    actual = resolver(resource)
    assert not Path(
        actual
    ).exists(), f"The resource '{resource}' should not be available"


def test_path_resolver_in_own_package_with_separate_file():
    resolver = path_resolver()
    assert resolver(
        "__init__.py"
    ).exists(), "Should find __init__.py in the current package"
