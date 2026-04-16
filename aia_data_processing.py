import astropy.units as u
from sunpy.net import Fido, attrs as a
import sunpy.map
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from aiapy.calibrate import update_pointing, register
from aiapy.calibrate.util import get_pointing_table
from reproject import reproject_interp

wavelengths = [94, 131, 171, 193, 211, 335]
results = []

for wl in wavelengths:
    res = Fido.search(
        a.Time("2023-06-01T00:00:00", "2023-06-01T00:00:20"),
        a.Instrument.aia,
        a.Wavelength(wl * u.angstrom),
        a.Sample(12*u.second)
    )
    results.append(res)

files = []
for res in results:
    files += Fido.fetch(res, path="./data", overwrite=False)

maps_1 = sunpy.map.Map(files)
maps_1 = sorted(maps_1, key=lambda m: m.wavelength.value)

pointing_table = get_pointing_table("JSOC", time_range=(maps_1[0].date - 1 * u.h,maps_1[0].date + 1 * u.h))
maps_15 = []

for m in maps_1:
    m = update_pointing(m, pointing_table=pointing_table)
    m = register(m)
    m = m / m.exposure_time
    maps_15.append(m)

ref_map = maps_15[0]
aligned_maps_15 = []

for m in maps_15:
    data, _ = reproject_interp(m, ref_map.wcs, ref_map.data.shape)
    aligned_maps_15.append(sunpy.map.Map(data, m.meta))

for m in aligned_maps_15:
    wavelength = int(m.wavelength.value)
    filename = f"data_15/aia_{wavelength}_15.fits"
    m.save(filename, overwrite=True)