#!/usr/bin/env python3
"""
Silent test runner that completely suppresses all warnings before running tests.

This script monkey-patches the warnings module at the system level before
any test code runs, ensuring even the most stubborn warnings are silenced.
"""

import os
import sys
import warnings
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Nuclear option: completely disable warnings before anything else loads
os.environ['PYTHONWARNINGS'] = 'ignore'

# Override warnings module at the system level
original_warn = warnings.warn
original_showwarning = warnings.showwarning


def null_warn(*args, **kwargs):
    """No-op replacement for warnings.warn"""
    pass


def null_showwarning(*args, **kwargs):
    """No-op replacement for warnings.showwarning"""
    pass


# Replace all warning functions with no-ops
warnings.warn = null_warn
warnings.showwarning = null_showwarning
warnings.simplefilter('ignore')

# Monkey-patch the warnings module in sys.modules
import types  # noqa: E402

warnings_module = types.ModuleType('warnings')
warnings_module.warn = null_warn
warnings_module.showwarning = null_showwarning
warnings_module.simplefilter = lambda *args, **kwargs: None
warnings_module.filterwarnings = lambda *args, **kwargs: None
sys.modules['warnings'] = warnings_module

# Now import and run the regular test runner
if __name__ == '__main__':
    from run_tests import main

    sys.exit(main())
