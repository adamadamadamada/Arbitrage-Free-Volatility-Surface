"""
Download real SPY option chain data from Yahoo Finance.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --ticker SPY --num-expiries 4 --output data/spy_options.csv
"""

import argparse
import sys
from datetime import datetime

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Install with:")
    print("  pip install yfinance")
    sys.exit(1)


def download_option_chain(ticker_symbol, num_expiries=4, min_volume=10):
    """
    Download option chain data from Yahoo Finance.
    
    Parameters
    ----------
    ticker_symbol : str
        Ticker symbol (e.g., 'SPY', 'AAPL')
    num_expiries : int
        Number of expiration dates to download
    min_volume : int
        Minimum volume filter (removes illiquid options)
    
    Returns
    -------
    pd.DataFrame
        Cleaned option data with columns:
        ['strike', 'expiry', 'option_type', 'bid', 'ask', 'price', 'volume', 'open_interest']
    """
    print(f"Fetching {ticker_symbol} option chain...")
    
    # Get ticker object
    ticker = yf.Ticker(ticker_symbol)
    
    # Get available expiration dates
    try:
        expirations = ticker.options
    except Exception as e:
        print(f"ERROR: Could not fetch expirations for {ticker_symbol}")
        print(f"  {e}")
        return None
    
    if len(expirations) == 0:
        print(f"ERROR: No options found for {ticker_symbol}")
        return None
    
    # Limit to first N expiries
    expirations = expirations[:num_expiries]
    print(f"Found {len(expirations)} expiration dates: {expirations}")
    
    # Get current spot price
    try:
        spot = ticker.history(period='1d')['Close'].iloc[-1]
        print(f"Current {ticker_symbol} price: ${spot:.2f}")
    except:
        print("WARNING: Could not fetch current price, using placeholder")
        spot = 100.0
    
    all_data = []
    
    for exp_date in expirations:
        print(f"\nProcessing expiry: {exp_date}")
        
        try:
            # Get option chain for this expiry
            chain = ticker.option_chain(exp_date)
            calls = chain.calls
            puts = chain.puts
            
            # Calculate time to expiry (in years)
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            today = datetime.now()
            days_to_expiry = (exp_datetime - today).days
            time_to_expiry = days_to_expiry / 365.25
            
            if time_to_expiry <= 0:
                print(f"  Skipping (already expired)")
                continue
            
            print(f"  Time to expiry: {time_to_expiry:.3f} years ({days_to_expiry} days)")
            
            # Process calls
            for _, row in calls.iterrows():
                if row.get('volume', 0) < min_volume:
                    continue
                
                all_data.append({
                    'strike': row['strike'],
                    'expiry': time_to_expiry,
                    'option_type': 'call',
                    'bid': row.get('bid', 0),
                    'ask': row.get('ask', 0),
                    'price': row.get('lastPrice', (row.get('bid', 0) + row.get('ask', 0)) / 2),
                    'volume': row.get('volume', 0),
                    'open_interest': row.get('openInterest', 0),
                    'implied_volatility': row.get('impliedVolatility', None),
                })
            
            # Process puts
            for _, row in puts.iterrows():
                if row.get('volume', 0) < min_volume:
                    continue
                
                all_data.append({
                    'strike': row['strike'],
                    'expiry': time_to_expiry,
                    'option_type': 'put',
                    'bid': row.get('bid', 0),
                    'ask': row.get('ask', 0),
                    'price': row.get('lastPrice', (row.get('bid', 0) + row.get('ask', 0)) / 2),
                    'volume': row.get('volume', 0),
                    'open_interest': row.get('openInterest', 0),
                    'implied_volatility': row.get('impliedVolatility', None),
                })
            
            print(f"  Collected {len([d for d in all_data if abs(d['expiry'] - time_to_expiry) < 0.001])} options")
        
        except Exception as e:
            print(f"  ERROR processing {exp_date}: {e}")
            continue
    
    if len(all_data) == 0:
        print("\nERROR: No option data collected")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Clean data
    print("\nCleaning data...")
    
    # Remove zero prices
    df = df[df['price'] > 0]
    
    # Remove NaN values
    df = df.dropna(subset=['strike', 'expiry', 'price'])
    
    # Sort by expiry then strike
    df = df.sort_values(['expiry', 'strike']).reset_index(drop=True)
    
    print(f"\nFinal dataset:")
    print(f"  Total options: {len(df)}")
    print(f"  Calls: {len(df[df['option_type'] == 'call'])}")
    print(f"  Puts: {len(df[df['option_type'] == 'put'])}")
    print(f"  Expiries: {df['expiry'].nunique()}")
    print(f"  Strike range: ${df['strike'].min():.2f} - ${df['strike'].max():.2f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Download option chain data from Yahoo Finance')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--num-expiries', type=int, default=4, help='Number of expiries to download (default: 4)')
    parser.add_argument('--min-volume', type=int, default=10, help='Minimum volume filter (default: 10)')
    parser.add_argument('--output', type=str, default='data/spy_options.csv', help='Output CSV path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OPTION CHAIN DATA DOWNLOADER")
    print("=" * 60)
    print(f"Ticker:       {args.ticker}")
    print(f"Expiries:     {args.num_expiries}")
    print(f"Min Volume:   {args.min_volume}")
    print(f"Output:       {args.output}")
    print("=" * 60)
    print()
    
    # Download data
    df = download_option_chain(args.ticker, args.num_expiries, args.min_volume)
    
    if df is None:
        print("\nFailed to download data. Exiting.")
        sys.exit(1)
    
    # Save to CSV
    print(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=False)
    
    print(f"✓ Saved {len(df)} options to {args.output}")
    print("\nSample data:")
    print(df.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nYou can now load this data:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_csv('{args.output}')")
    print()


if __name__ == "__main__":
    main()