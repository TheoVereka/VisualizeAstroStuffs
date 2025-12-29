# plot_horizons_plotly.py
# English comments only (per user request)
# Purpose: fetch spacecraft vectors from JPL Horizons (via astroquery)
# and plot interactive 3D tracks with Plotly.

import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u
import time
from requests.exceptions import HTTPError

# astroquery to talk to JPL Horizons
from astroquery.jplhorizons import Horizons

# optional: coordinate transforms if you want ecliptic frame
from astropy.coordinates import SkyCoord, CartesianRepresentation, ICRS, HeliocentricTrueEcliptic

import plotly.graph_objects as go

def _call_horizons_with_retries(name, epochs_arg, max_retries=4, pause_base=1.0):
    """
    Helper: call Horizons(..., epochs=epochs_arg) with retries and exponential backoff.
    epochs_arg may be either:
      - a numpy array/list of JDs (short lists are OK), or
      - a dict {'start': 'YYYY-MM-DD', 'stop': 'YYYY-MM-DD', 'step': 'Nd'} which avoids long TLIST.
    Returns pandas DataFrame (or raises if all retries fail).
    """
    attempt = 0
    while attempt < max_retries:
        try:
            obj = Horizons(id=name, location='@sun', epochs=epochs_arg)
            table = obj.vectors()
            return table.to_pandas()
        except HTTPError as e:
            # If server returned 502/5xx, retry with backoff
            attempt += 1
            wait = pause_base * (2 ** (attempt - 1))
            print(f"HTTPError on attempt {attempt}/{max_retries}: {e}. Retrying after {wait:.1f}s...")
            time.sleep(wait)
        except Exception as e:
            # Other unexpected exceptions: raise immediately
            print("Unexpected error when querying Horizons:", type(e), e)
            raise
    # if we exit loop, all retries failed
    raise RuntimeError(f"Horizons query failed after {max_retries} attempts.")

def fetch_vectors_horizons(name, start_iso, stop_iso, n_points=200, prefer_range_threshold=250):
    """
    Robust fetch from JPL Horizons:
    - If n_points is small (< prefer_range_threshold), request explicit JD list.
    - If n_points is large, request a range (start/stop/step) to avoid long TLIST URIs.
    - Returns concatenated pandas DataFrame with x,y,z columns (AU).
    """
    t0 = Time(start_iso, format='iso', scale='utc')
    t1 = Time(stop_iso, format='iso', scale='utc')

    total_days = (t1 - t0).to(u.day).value
    if n_points <= 1:
        jd_grid = np.array([t0.jd])
    else:
        jd_grid = np.linspace(t0.jd, t1.jd, n_points)

    # If many points, prefer asking Horizons using a step-range (smaller URL)
    if n_points > prefer_range_threshold:
        # compute integer day step (at least 1 day)
        step_days = max(1, int(round(total_days / max(1, n_points - 1))))
        epochs_arg = {'start': start_iso, 'stop': stop_iso, 'step': f'{step_days}d'}
        print(f"Using range request to Horizons with step = {step_days} day(s) to avoid long URL.")
        df = _call_horizons_with_retries(name, epochs_arg)
        # Note: number of returned samples may differ slightly from n_points
        return df
    else:
        # For modest sized requests, use chunking to be safe (avoid giant single TLIST)
        max_chunk = 120  # keep each TLIST under ~120 entries
        dfs = []
        for i in range(0, len(jd_grid), max_chunk):
            chunk = jd_grid[i:i+max_chunk]
            print(f"Querying Horizons for {name}: chunk {i//max_chunk + 1}, {len(chunk)} epochs...")
            df_chunk = _call_horizons_with_retries(name, chunk)
            dfs.append(df_chunk)
        # concatenate and drop possible duplicate header rows
        df_all = pd.concat(dfs, ignore_index=True)
        # Some Horizons responses include overlapping rows; optionally drop exact-duplicate epochs
        if 'datetime_jd' in df_all.columns:
            df_all = df_all.drop_duplicates(subset=['datetime_jd'])
        return df_all

def to_ecliptic(df):
    """
    Convert a DataFrame with 'x','y','z' (ICRS-like heliocentric cartesian in AU)
    to Heliocentric True Ecliptic coordinates. Returns (x_ecl, y_ecl, z_ecl) in AU arrays.
    """
    coords = SkyCoord(
        x=df['x'].values * u.AU,
        y=df['y'].values * u.AU,
        z=df['z'].values * u.AU,
        frame=ICRS,
        representation_type=CartesianRepresentation
    )
    ecl = coords.transform_to(HeliocentricTrueEcliptic())
    return ecl.x.to(u.AU).value, ecl.y.to(u.AU).value, ecl.z.to(u.AU).value

def plot_3d_trajectories(name_list, df_list, use_ecliptic=False):
    """
    Make an interactive Plotly 3D plot from lists of dataframes.
    """
    fig = go.Figure()
    for name, df in zip(name_list, df_list):
        if use_ecliptic:
            x, y, z = to_ecliptic(df)
        else:
            x, y, z = df['x'].values, df['y'].values, df['z'].values

        # add trace: line + gradient markers (color by epoch index)
        color_vals = np.linspace(0, 1, len(x))
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(size=3, opacity=0.8, color=color_vals, colorscale='Viridis', colorbar=dict(title='time')),
            line=dict(width=2),
            name=name
        ))

    # draw Sun at origin
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=6), name='Sun'))

    fig.update_layout(
        title='Spacecraft outbound trajectories (heliocentric coordinates, units: AU)',
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)',
            aspectmode='auto'  # change to 'cube' if you want equal axes
        ),
        legend=dict(x=0.9, y=0.9)
    )
    fig.show()

if __name__ == "__main__":
    # configuration - copy-paste change these lines as you like
    names = ['Voyager 1', 'Voyager 2', 'New Horizons']
    start = '2012-01-01'   # ISO string
    stop  = '2025-12-01'   # ISO string
    samples = 300

    dfs = []
    for nm in names:
        print(f"Fetching {nm} from JPL Horizons...")
        df = fetch_vectors_horizons(nm, start, stop, n_points=samples)
        # inspect columns if you want
        print(df.columns.tolist())
        dfs.append(df)

    # Plot in heliocentric ICRS-like frame (use_ecliptic=True to convert to ecliptic)
    fig = plot_3d_trajectories(names, dfs, use_ecliptic=False)
    
    
    fig.show()
    # Prevent the script from exiting immediately so you can inspect the plot (press Enter to exit)
    input("Press ENTER to exit and close the script (the browser window stays).")