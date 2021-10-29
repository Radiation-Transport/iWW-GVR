from pathlib import Path
from typing import Callable

import pkg_resources as pkg


def filename_resolver(package: str) -> Callable[[str], str]:

    resource_manager = pkg.ResourceManager()

    def func(resource):
        return resource_manager.resource_filename(package, resource)

    func.__doc__ = f"Computes file names for resources located in {package}"

    return func


def path_resolver(package: str) -> Callable[[str], Path]:

    resolver = filename_resolver(package)

    def func(resource):
        filename = resolver(resource)
        return Path(filename)

    func.__doc__ = f"Computes Path for resources located in {package}"

    return func
