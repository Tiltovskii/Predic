from __future__ import annotations

import unittest

from predic_v2.cli import _parser
from predic_v2.light_argus_data import DEFAULT_MAX_HISTORY


class CliDefaultsTest(unittest.TestCase):
    def test_light_argus_history_default_is_shared(self) -> None:
        args = _parser().parse_args(
            [
                "build-light-argus-dataset",
                "--state-db",
                "state.sqlite3",
                "--matches-csv",
                "matches.csv",
                "--map-features-csv",
                "maps.csv",
                "--output-dir",
                "dataset",
            ]
        )

        self.assertEqual(args.max_history, DEFAULT_MAX_HISTORY)
        self.assertEqual(args.max_history, 256)


if __name__ == "__main__":
    unittest.main()
