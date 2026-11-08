"""Settings for the test run.

install.sh writes OFIQ_CONFIG_DIR (and SIGLIP_ENABLED) to the repo's .env. The validator reads .env itself,
but the skip markers in test_checks.py look at os.environ, so the file is loaded here before the tests
are collected. Variables already set in the environment win.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')
