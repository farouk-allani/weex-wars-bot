"""WEEX AI Wars II trading bot."""

import sys

# Windows consoles default to cp1252, and rich raises UnicodeEncodeError on any
# character outside it (arrows, box drawing) the moment stdout is a pipe rather
# than a terminal — which is how Docker and CI always run us. That exception is
# raised from inside whatever handler was printing, so an unlucky one can abort a
# cycle halfway through booking a fill. Degrade the glyph instead of the trade.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
