import os, glob
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews import opts
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.path as mpath
from holoviews import Overlay
from panel.template import FastListTemplate
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os, requests, zipfile, io

import warnings
warnings.filterwarnings("ignore")

import panel as pn

# ─── Extensions ─────────────────────────────────────────────────────────────
hv.extension('bokeh')
pn.extension()

import boto3
import s3fs
# AWS credentials & bucket info via env vars

aws_access_key = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
aws_region     = os.environ.get('AWS_REGION', 'us-east-1')
bucket         = os.environ['ICEBRIDGE_BUCKET']
# prefix         = os.environ.get('ICEBRIDGE_PREFIX', '')  # e.g. 'icebridge/'

session = boto3.Session(region_name=aws_region)
s3      = session.client('s3')
paginator = s3.get_paginator('list_objects_v2')


all_keys = []
for page in paginator.paginate(Bucket=bucket):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.lower().endswith('.csv'):
            all_keys.append(key)

# 3) Map just the basename → full key
flight_to_key = {os.path.basename(key): key for key in all_keys}

# Initialize s3fs filesystem (streaming reads)
# fs = s3fs.S3FileSystem(
#     key=aws_access_key,
#     secret=aws_secret_key,
#     client_kwargs={'region_name': aws_region}
# )

# data_files = [f"s3://{bucket}/{key}" for key in fs.glob(f"{bucket}/*.csv")]
# data_dir = None  # not used

# instrument flags → friendly names
instrument_flags = {
    'has_POSAV':      'POSAV',
    'has_snow_radar': 'Snow Radar',
    'has_full_ATM':   'Full-swath ATM',
    'has_narrow_ATM': 'Narrow-swath ATM',
    'has_KT19':       'KT-19'
}

# desired vertical order
instrument_order = [
    'POSAV',
    'Snow Radar',
    'Full-swath ATM',
    'Narrow-swath ATM',
    'KT-19'
]

# custom palette for instruments
color_map = {
    'POSAV':         'black',
    'Snow Radar':    'cornflowerblue',
    'Full-swath ATM':'green',
    'Narrow-swath ATM':'lightgreen',
    'KT-19':         'darksalmon'
}

# ─── 1. Build the segments dataframe ─────────────────────────────────────────
records = []

# for uri in data_files:
for fn, key in flight_to_key.items():
    # Build a s3:// URI for pandas
    uri = f"s3://{bucket}/{key}"
    
    # Stream in only the columns we care about
    df = pd.read_csv(
        uri,
        storage_options={
            "key": aws_access_key,
            "secret": aws_secret_key
        },
        usecols=lambda c: c in {
            'cumulative_distance',
            'has_POSAV','has_snow_radar',
            'has_full_ATM','has_narrow_ATM','has_KT19',
            'mission','radar_name_code',
            'full_atm_instrument_id','narrow_atm_instrument_id'
        }
    )
    
    df['cumulative_distance'] = df['cumulative_distance']/1000

    # extract metadata safely, defaulting to None
    mission         = df['mission'].loc[df['mission'].notna()].iloc[0]         if 'mission' in df.columns         else None
    radar_name_code = df['radar_name_code'].loc[df['radar_name_code'].notna()].iloc[0] if 'radar_name_code' in df.columns else None
    full_atm_instrument_id   = df['full_atm_instrument_id'].loc[df['full_atm_instrument_id'].notna()].iloc[0]   if 'full_atm_instrument_id' in df.columns   else None
    narrow_atm_instrument_id   = df['narrow_atm_instrument_id'].loc[df['narrow_atm_instrument_id'].notna()].iloc[0]   if 'narrow_atm_instrument_id' in df.columns   else None

    for flag, name in instrument_flags.items():
        sub = df[df.get(flag, False) == True].copy()
        if sub.empty:
            records.append({
                'flight': fn,
                'instrument': name,
                'start_dist': None,
                'end_dist': None,
                'row': f"{fn}\n{name}",
                'year': fn.split('_')[1][:4],
                'mission': mission,
                'radar_name_code': radar_name_code,
                'full_atm_instrument_id': full_atm_instrument_id,
                'narrow_atm_instrument_id': narrow_atm_instrument_id
            })
        else:
            sub['_seg'] = (sub.index.to_series().diff() != 1).cumsum()
            for _, grp in sub.groupby('_seg'):
                records.append({
                    'flight': fn,
                    'instrument': name,
                    'start_dist': grp['cumulative_distance'].min(),
                    'end_dist':   grp['cumulative_distance'].max(),
                    'row': f"{fn}\n{name}",
                    'year': fn.split('_')[1][:4],
                    'mission': mission,
                    'radar_name_code': radar_name_code,
                    'full_atm_instrument_id': full_atm_instrument_id,
                    'narrow_atm_instrument_id': narrow_atm_instrument_id
                })


segments_df = pd.DataFrame(records)
segments_df['color'] = segments_df['instrument'].map(color_map)

# ─── 2. Load flight tracks (lon/lat) ─────────────────────────────────────────
tracks_dfs = {}
for fn, key in flight_to_key.items():
    # Build a s3:// URI for pandas
    uri = f"s3://{bucket}/{key}"
    fn = os.path.basename(uri)
    df = pd.read_csv(uri,
        storage_options={'key': aws_access_key, 'secret': aws_secret_key},
        usecols=['x', 'y'])
    tracks_dfs[fn] = df
    
keys = list(tracks_dfs.keys())

# ─── 4. Timeline callback (HoloViews + Bokeh) ────────────────────────────────
def make_timeline(selected_flights):
    # categorical rows in reverse so POSAV at top
    cats = []
    for fn in sorted(selected_flights):
        for instr in instrument_order:
            cats.append(f"{fn}\n{instr}")
        # unique blank spacer (filename + newline + space)
        cats.append(f"{fn}\n\n")
    cats = cats[::-1]                # reverse so POSAV is at top

    # cats = [f"{fn}\n{instr}" for fn in sorted(selected_flights) for instr in instrument_order][::-1]
    row_dim = hv.Dimension('row', label='row', values=cats)

    sub     = segments_df[segments_df['flight'].isin(selected_flights)]
    segs_df = sub.dropna(subset=['start_dist', 'end_dist'])

    segs = hv.Segments(
        segs_df,
        kdims=[('start_dist','Start (dist)'), row_dim, ('end_dist','End (dist)'), row_dim],
        vdims=[('flight','Flight'),('instrument','Instrument'),('year','Year'),('color','Color')]
    )

    # dynamic height: 8px per lane, min 120px
    height = max(len(cats) * 7, 200)


    layers = []

    for instr in instrument_order:
        df_i = segs_df[segs_df['instrument'] == instr]
        if df_i.empty: 
            continue

        # build the list of tooltips you actually want
        hover = [ ]
        
        if instr == 'Full-swath ATM':
            hover.append(('Full ATM ID', '@full_atm_instrument_id'))
            
        if instr == 'Narrow-swath ATM':
            hover.append(('Narrow ATM ID', '@narrow_atm_instrument_id'))
            
        if instr == 'Snow Radar':
            hover.append(('Radar code', '@radar_name_code'))
            hover.append(('Mission', '@mission'))

        seg_i = hv.Segments(
            df_i,
            kdims=[('start_dist','Start (dist)'), row_dim, ('end_dist','End (dist)'), row_dim],
            vdims=[('mission','Mission'),
                ('full_atm_instrument_id','Full ATM ID'),
                ('narrow_atm_instrument_id','Narrow ATM ID'),
                ('radar_name_code','Radar code'),
                ('color','Color')]
        ).opts(
            opts.Segments(
                color=color_map[instr],
                line_width=4,
                # tools=['hover'],
                hover_tooltips=hover
            )
        )
        layers.append(seg_i)


    # overlay all instrument‐specific layers
    segs_overlay = Overlay(layers)


    labels = []
    for fn in sorted(selected_flights):
        raw   = fn.split('_')[1].split('.')[0]
        date  = f"{raw[6:]}-{raw[4:6]}-{raw[:4]}"
        row   = f"{fn}\nFull-swath ATM"
        start = segs_df[segs_df['flight'] == fn]['start_dist'].min()
        labels.append({'start_dist': start, 'row': row, 'Date': date})
    labels_df = pd.DataFrame(labels)

    lbls = hv.Labels(
        labels_df,
        kdims=['start_dist','row'],
        vdims=['Date']
    )
    lbls = lbls.opts(
        opts.Labels(
            text_font_size='10pt',
            text_color='black',
            text_align='right',
            text_baseline='middle',
            xoffset=-.5,
        )
    )
    plot = (segs_overlay * lbls)

    # 7) Final shared options
    return plot.opts(
        opts.Segments(
            fontsize={'xlabel': 10},
            height=height,
            width=1000,
            xlabel='Cumulative distance [km]',
            # title='Instrument availability',
            xaxis='top',
            yaxis=None,
            framewise=True
        )
    )
    

# Bind and then wrap the timeline pane with stretch_width

# after you define flight_options = sorted(tracks_dfs.keys())


# ─── 5. Map callback (Matplotlib + Cartopy) ─────────────────────────────────
def make_map(selected_flights):
    colors = get_flight_colors(selected_flights)
    
    fig = plt.figure(figsize=(5,5), dpi=200)
    ax  = fig.add_subplot(1,1,1, projection=ccrs.NorthPolarStereo())
    ax.coastlines()
    ax.stock_img()
    ax.gridlines(draw_labels=False, linestyle=':')
    for fn in sorted(selected_flights):
        df = tracks_dfs[fn]
        ax.scatter(df['x'][::10], df['y'][::10], transform=ccrs.epsg(3413), color=colors[fn], s=.01, marker='.', zorder=100)
        
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)
    mpl_plane = pn.pane.Matplotlib(
        fig,
        dpi=200,
        tight=True,
        width=375,      # ensure it never exceeds the 400px sidebar
        sizing_mode='fixed'
    )
    plt.close(fig)  # close the figure to free memory
    return mpl_plane




# 1) Define a little Markdown legend with colored swatches

legend_html = pn.pane.HTML("""
    <div style="display:flex; gap:20px; align-items:center; font-size:14px;">
    <div style="display:flex; align-items:center; gap:4px;">
        <span style="background:black; width:12px; height:6px; display:inline-block;"></span>
        <span>POSAV (400 m < Altitude < 600 m, |Pitch| & |Roll| < 3°)</span>
    </div>
    <div style="display:flex; align-items:center; gap:4px;">
        <span style="background:cornflowerblue; width:12px; height:6px; display:inline-block;"></span>
        <span>Snow Radar</span>
    </div>
    <div style="display:flex; align-items:center; gap:4px;">
        <span style="background:green; width:12px; height:6px; display:inline-block;"></span>
        <span>Full-swath ATM</span>
    </div>
    <div style="display:flex; align-items:center; gap:4px;">
        <span style="background:lightgreen; width:12px; height:6px; display:inline-block;"></span>
        <span>Narrow-swath ATM</span>
    </div>
    <div style="display:flex; align-items:center; gap:4px;">
        <span style="background:darksalmon; width:12px; height:6px; display:inline-block;"></span>
        <span>KT-19</span>
    </div>
    </div>
    """,
    sizing_mode="stretch_width"
)



# 1) Compute the list of all years and all flights up-front
flight_options = sorted(tracks_dfs.keys())
all_years      = sorted({fn.split("_")[1][:4] for fn in flight_options})


# 1) Create one Checkbox widget per year
year_checks = [
    pn.widgets.Checkbox(name=yr, value=True)
    for yr in all_years
]

chunk_size = 5
rows = []
for i in range(0, len(year_checks), chunk_size):
    chunk = year_checks[i : i + chunk_size]
    rows.append(pn.Row(*chunk, sizing_mode="fixed"))

# 3) Assemble into a Column with fixed width
year_selector_box = pn.Column(
    *rows,
    width=380,
    sizing_mode="fixed"
)

flight_selector = pn.widgets.MultiSelect(
    options=flight_options,
    value=flight_options,
    size=6,     # number of visible options
    # height=1   # total pixel height (will scroll if needed)
)



def get_flight_colors(flights):
    # pick a colormap with as many entries as flights
    cmap = cm.get_cmap("tab20", len(flights))
    return {fn: mcolors.to_hex(cmap(i)) for i, fn in enumerate(flights)}


# ─── at module top ────────────────────────────────────────────────────────────
# global flag to suppress intermediate callbacks
suppress_update = False

# ─── your existing update function, with a suppression guard ──────────────────
def _update_flights(event=None):
    global suppress_update
    if suppress_update:
        return   # skip while we’re bulk‐setting checkboxes
    sel_years = {cb.name for cb in year_checks if cb.value}
    new_dates = [fn for fn in flight_options if fn.split("_")[1][:4] in sel_years]
    flight_selector.options = new_dates
    flight_selector.value   = list(new_dates)
    

# watch individual boxes as before
for cb in year_checks:
    cb.param.watch(_update_flights, 'value')
    
# seed initial
_update_flights()

# ─── wired to your buttons ────────────────────────────────────────────────────
def _select_all(event):
    global suppress_update
    suppress_update = True
    for cb in year_checks:
        cb.value = True
    suppress_update = False
    _update_flights()

def _deselect_all(event):
    global suppress_update
    suppress_update = True
    for cb in year_checks:
        cb.value = False
    suppress_update = False
    _update_flights()
    
btn_select_all   = pn.widgets.Button(name="Select All", button_type="primary", width=100)
btn_deselect_all = pn.widgets.Button(name="Deselect All", button_type="default", width=100)
btn_select_all.on_click(_select_all)
btn_deselect_all.on_click(_deselect_all)


# 3) Include them in your sidebar

# 4) bind your panes as before:
timeline_pane = pn.bind(make_timeline, flight_selector)
map_pane      = pn.bind(make_map,      flight_selector)
timeline_pane = pn.panel(timeline_pane, sizing_mode="stretch_width")



# ─── Prepare your scrollable panels ────────────────────────────────────────────

# Wrap the timeline in a fixed‐height, scrollable container
timeline_scroll = pn.Column(
    "### Instrument & Data availability",
    legend_html,            # your single‐row legend
    timeline_pane,          # HoloViews Gantt
    # sizing_mode="stretch_width",
    scroll=True             # enables an internal scrollbar
)

# Define the variable‐plot callback (along‐track curves)

def make_variable_plot(selected_flights, variable):
    colors = get_flight_colors(selected_flights)
    
    curves = []
    for fn in sorted(selected_flights):
        date = fn.split('.')[0].split('_')[1]
        date_formatted = date[6:] + '-' + date[4:6] + '-' + date[:4]  # DD-MM-YYYY format
        uri = f"s3://{bucket}/{fn}"
        df = pd.read_csv(uri,
            storage_options={'key': aws_access_key, 'secret': aws_secret_key},
            usecols=lambda c: c in {'cumulative_distance', variable}
        )
            
        df['cumulative_distance'] /= 1000  # to km
        # df['myi_concentration'] *= 100
        curves.append(hv.Scatter(
            (df["cumulative_distance"], df[variable]),
            label=date_formatted
        ).opts(
                color=colors[fn],
                size=.5
            )
        )
    return hv.Overlay(curves).opts(
        opts.Scatter(
            tools=["hover"],
            xlabel="Cumulative distance [km]",
            ylabel=variable,
            show_legend=True,

        ),
        # overlay‐level legend positioning
        opts.Overlay(
            fontsize={'xlabel': 10, 'ylabel': 10},
            legend_position="top",
            legend_cols=8,
            legend_padding=10,
            legend_opts={
                "background_fill_alpha": 0,   # transparent background
                "border_line_alpha":     0    # no border
            }
        )
    )
    
sample_fp = flight_options[53]
uri = f"s3://{bucket}/{sample_fp}"
full_cols = pd.read_csv(uri,
            storage_options={'key': aws_access_key, 'secret': aws_secret_key},
        ).columns.tolist()
var_options = [c for c in full_cols if c != "cumulative_distance"]

variable_selector = pn.widgets.Select(
    options=var_options,
    value="myi_concentration",  # sensible default
    width=200
)


var_pane = pn.bind(make_variable_plot, flight_selector, variable_selector)

# 2) Wrap var_pane so it stretches to fill the column:
var_panel = pn.panel(var_pane, sizing_mode="stretch_width")



var_scroll = pn.Column(
    "### Along-track variable",
    var_panel,
    height=500,               # half‐height box
    scroll=True,
    sizing_mode="stretch_width"
)

# Now rebuild your right_column with:
right_column = pn.Column(
    timeline_scroll,   # already stretch_width
    pn.Spacer(height=5),
    var_scroll,
    sizing_mode="stretch_both"
)



# drop the one you use for x—everything else is a candidate

template = FastListTemplate(
    title="IceBridge Arctic Sea Ice Dashboard",
    sidebar=[
        "## Select years & flights",
        pn.Row(btn_select_all, btn_deselect_all, sizing_mode="fixed", ),
        year_selector_box,
        flight_selector,
        '## Select variable',
        variable_selector,
        pn.Spacer(height=5),
        map_pane,
        ],
    main=[
        right_column
    ],
    sidebar_width=400,
    main_max_width="100%",
)

if __name__.startswith("bokeh"):
    template.servable()
else:
    template.show()