from __future__ import annotations

import logging

from scs.environment import Environment
from scs.owls import Owl


def logger_setup(log_file_name: str = "test") -> None:
    """soon"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s]::%(levelname)s::[%(name)s.%(funcName)s]::%(lineno)d::%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(f"{log_file_name}.log", mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def main() -> None:
    forest = Environment()
    forest.add_owl_scout(("os_1", (10, 10), 0))
    forest.add_owl_scout(("os_2", (-10, -10), 0))

    test_owl = Owl("test_owl", 50)
    forest.add_owl(test_owl)
    forest.simulate(for_seconds=120)

    # forest.plot_owl_path("test_owl")


# TODO: Improve method of calculating position accuracy error

if __name__ == "__main__":
    logger_setup()
    main()
