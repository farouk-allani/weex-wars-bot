"""WEEX AI Wars II — Trading Bot Entry Point"""

from .core.engine import TradingEngine


def main():
    """Run the trading bot."""
    engine = TradingEngine()
    engine.run()


if __name__ == "__main__":
    main()
