# Scripts

## `download_data.py`

Download real option chain data from Yahoo Finance.

### Installation

```bash
pip install yfinance

Basic Usage


# Download SPY options (default)
python scripts/download_data.py

# Download AAPL options with more expiries
python scripts/download_data.py --ticker AAPL --num-expiries 6

# Custom output location
python scripts/download_data.py --output data/my_data.csv

# Filter for higher volume
python scripts/download_data.py --min-volume 50