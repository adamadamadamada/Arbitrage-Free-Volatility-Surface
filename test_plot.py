import pandas as pd
from vol_surface import VolatilitySurface

df = pd.read_csv('data/spy_options.csv')

surface = VolatilitySurface(S=590, r=0.045)
surface.load_data(df)
surface.compute_ivs()

surface.plot_smile(expiry=df['expiry'].iloc[0])

surface.fit_svi()
surface.plot_surface_3d(model='svi')


# Plot raw market data (no fitting needed)
surface.plot_surface_3d(model='market')

surface.summary()

