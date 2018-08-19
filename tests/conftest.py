def pytest_addoption(parser):
    parser.addoption(
        "--nll",
        action="store_true",
        dest="nll",
        default=False,
        help=(
            "Run Negative Log Likelihood gradient tests. "
            "Note that they are non-deterministic and may fail."
        ),
    )
