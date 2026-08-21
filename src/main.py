"""WEEX competition-rehearsal trading bot entry point."""

from .core.engine import TradingEngine


def main():
    """Run the trading bot."""
    engine = TradingEngine()
    engine.run()


if __name__ == "__main__":
    main()
