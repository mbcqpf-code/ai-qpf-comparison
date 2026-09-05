import s3fs
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import numpy as np
import requests
import os
from ecmwf.opendata import Client

# ==========================================
# DYNAMIC TIME SYNCING (Match to IFS)
# ==========================================
def get_latest_cycle():
    fs = s3fs.S3FileSystem(anon=True)
    now = datetime.utcnow()
    candidates = [
        (now.replace(hour=12, minute=0, second=0, microsecond=0), '12'),
        (now.replace(hour=0, minute=0, second=0, microsecond=0), '00'),
        ((now - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0), '12')
    ]
    
    for dt, hr in candidates:
        date_str, year_str, md_str = dt.strftime('%Y%m%d'), dt.strftime('%Y'), dt.strftime('%m%d')
        path = f"s3://noaa-oar-mlwp-data/GRAP_v100_IFS/{year_str}/{md_str}/GRAP_v100_IFS_{date_str}{hr}_f000_f240_06.nc"
        if fs.exists(path):
            print(f"Synced to latest available cycle: {date_str} {hr}z")
            return dt, hr, date_str, year_str, md_str
            
    raise FileNotFoundError("Could not find recent GraphCast IFS data.")

DATE, INIT_HOUR, date_str, year_str, month_day_str = get_latest_cycle()

os.makedirs('images', exist_ok=True)

# ==========================================
# COLORMAP CONFIGURATION
# ==========================================
clevs = [0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 10.00, 15.00, 20.00]
colors = ['#7CFF4C', '#26A400', '#006300', '#003E94', '#207CE8', '#42A6FF', '#21F0FF', '#A198FF', '#8A2BE2', '#5200C9', '#8B0000', '#CD0000', '#FF3200', '#FF8C00', '#FFC800', '#FFFF00', '#FFC0CB']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(clevs, cmap.N)
proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)

# Absolute Datetime initialization for accurate slicing
init_dt = datetime.strptime(f"{date_str}{INIT_HOUR}", "%Y%m%d%H")

# ==========================================
# LOOP DAYS 1 TO 7 (12z to 12z)
# ==========================================
for day in range(1, 8):
    print(f"--- Processing Day {day} ---")
    plot_data_dict = {}
    
    if INIT_HOUR == '00':
        start_hr = 12 + (day - 1) * 24
    else:
        start_hr = (day - 1) * 24
        
    target_hours = [start_hr + 6, start_hr + 12, start_hr + 18, start_hr + 24]
    
    # Calculate exact absolute times to avoid index shifting
    target_times = [np.datetime64(init_dt + timedelta(hours=h)) for h in target_hours]
    
    # 1. AWS GRAPHCAST
    fs = s3fs.S3FileSystem(anon=True)
    for init_model in ['GFS', 'IFS']:
        model_dir = f"GRAP_v100" if init_model == 'GFS' else f"GRAP_v100_IFS"
        s3_path = f"s3://noaa-oar-mlwp-data/{model_dir}/{year_str}/{month_day_str}/{model_dir}_{date_str}{INIT_HOUR}_f000_f240_06.nc"
        
        try:
            # Removed 'with' block to prevent premature file closure during lazy loading
            s3_file = fs.open(s3_path, 'rb')
            ds = xr.open_dataset(s3_file, engine='h5netcdf')
            
            qpf_var = [var for var in ds.data_vars if 'precip' in var.lower() or var.lower() in ['tp', 'apcp']][0]
            
            # Sum based on the explicitly calculated absolute times
            qpf_24hr_inches = ds[qpf_var].sel(time=target_times).sum(dim='time') * 39.3701 
            plot_data_dict[init_model] = qpf_24hr_inches.where(qpf_24hr_inches >= 0.01)
        except Exception as e:
            print(f"{init_model} failed: {e}")

    # 2. NCEP AIGFS (Byte Range)
    aigfs_qpf_arrays = []
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.{date_str}/{INIT_HOUR}/model/atmos/grib2/"
    for fhr in target_hours:
        file_name = f"aigfs.t{INIT_HOUR}z.sfc.f{fhr:03d}.grib2"
        r_idx = requests.get(base_url + file_name + ".idx")
        if r_idx.status_code == 200:
            lines = r_idx.text.splitlines()
            start_byte, end_byte = None, None
            for i, line in enumerate(lines):
                if ':APCP:' in line:
                    start_byte = int(line.split(':')[1])
                    end_byte = int(lines[i+1].split(':')[1]) - 1 if i + 1 < len(lines) else ""
                    break
            if start_byte is not None:
                r_grib = requests.get(base_url + file_name, headers={"Range": f"bytes={start_byte}-{end_byte}"})
                with open("tmp.grib2", 'wb') as f: f.write(r_grib.content)
                try:
                    ds_grib = xr.open_dataset("tmp.grib2", engine='cfgrib')
                    if 'tp' in ds_grib.variables: aigfs_qpf_arrays.append(ds_grib['tp'])
                except: pass
    if aigfs_qpf_arrays:
        aigfs_24hr = sum(aigfs_qpf_arrays) * 0.0393701
        plot_data_dict['AIGFS'] = aigfs_24hr.where(aigfs_24hr >= 0.01)

    # 3. ECMWF AIFS
    try:
        s1, s2 = start_hr, start_hr + 24
        client = Client(source="aws", model="aifs-single", resol="0p25")
        steps_to_download = [s2] if s1 == 0 else [s1, s2]
        
        client.download(date=date_str, time=int(INIT_HOUR), step=steps_to_download, param="tp", target="aifs.grib2")
        ds_aifs = xr.open_dataset("aifs.grib2", engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})
        
        if s1 == 0:
            qpf_aifs = ds_aifs['tp'].sel(step=np.timedelta64(s2, 'h'))
        else:
            qpf_aifs = ds_aifs['tp'].sel(step=np.timedelta64(s2, 'h')) - ds_aifs['tp'].sel(step=np.timedelta64(s1, 'h'))
            
        aifs_inches = qpf_aifs * 0.0393701
        plot_data_dict['AIFS'] = aifs_inches.where(aifs_inches >= 0.01)
    except Exception as e:
        print(f"AIFS failed: {e}")

    # ==========================================
    # PLOT GENERATION
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), subplot_kw={'projection': proj})
    axes = axes.flatten()
    mesh = None
    
    plot_mapping = [
        (0, 'GFS', f'GraphCast (GFS) Day {day} 24hr QPF'),
        (1, 'IFS', f'GraphCast (IFS) Day {day} 24hr QPF'),
        (2, 'AIGFS', f'NCEP AIGFS Day {day} 24hr QPF'),
        (3, 'AIFS', f'ECMWF AIFS Day {day} 24hr QPF')
    ]

    for ax_idx, model_key, title in plot_mapping:
        ax = axes[ax_idx]
        if model_key in plot_data_dict:
            ax.set_extent([-125, -67, 24, 50], ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.8)
            ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray')
            ax.add_feature(cfeature.LAKES, alpha=0.5)
            mesh = plot_data_dict[model_key].plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(), x='longitude', y='latitude',
                cmap=cmap, norm=norm, add_colorbar=False, add_labels=False 
            )
            ax.set_title(f"{title}\nInit: {date_str} {INIT_HOUR}z | Valid 12z-12z", fontsize=13, loc='left', pad=6)
        else:
            ax.text(0.5, 0.5, "Data Unavailable", transform=ax.transAxes, ha='center', fontsize=14)
            ax.axis('off')

    plt.subplots_adjust(hspace=0.18, wspace=0.04, bottom=0.12, top=0.95, left=0.05, right=0.95)
    if mesh:
        cbar_ax = fig.add_axes([0.15, 0.05, 0.70, 0.025])
        cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal', ticks=clevs)
        cbar.set_label('24hr QPF (Inches)', fontsize=13, labelpad=6)
        cbar.ax.set_xticklabels([str(c) for c in clevs], fontsize=10)

    plt.savefig(f'images/day{day}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
