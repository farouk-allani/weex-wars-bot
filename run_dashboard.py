"""Launch the WEEX AI Wars monitoring dashboard.

  python run_dashboard.py
  python run_dashboard.py --port 8788
  → http://127.0.0.1:8787  (auto-picks next free port if busy)
"""

from dashboard.app import main

if __name__ == "__main__":
    main()
