"""Tests to ensure configuration consistency."""
import inspect

import pytest

from edutap.image_api.core.config import ValidatorConfig
from edutap.image_api.core.checks import quality, geometry, lighting, features, background


def test_defaults_are_in_sync():
    """
    TODO: Implement a test that uses reflection to ensure that the default
    values in the check function signatures match the default values
    defined in the ValidatorConfig.

    This test will prevent the configuration and the function defaults
    from drifting out of sync.

    Example logic:
    - Get the default config instance.
    - Iterate through check modules (quality, geometry, etc.).
    - For each check function, inspect its signature.
    - For each parameter with a default value, find the corresponding
      attribute in the config.
    - Assert that the function's default value matches the config's default value.
    """
    pytest.skip("TODO: Implement reflection-based config sync test.")


