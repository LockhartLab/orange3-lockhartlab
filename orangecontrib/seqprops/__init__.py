"""Old name of `orangecontrib.lockhartlab` (before 0.12).

Kept so that models and tables saved with older versions, which refer to
`orangecontrib.seqprops.<module>`, still load. Widgets are not listed
here; each one names its old path in `replaces`, so old workflows open.
"""

import importlib
import sys

for _name in ("balance", "descriptors", "featuresets", "kmers", "properties"):
    _module = importlib.import_module(f"orangecontrib.lockhartlab.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
