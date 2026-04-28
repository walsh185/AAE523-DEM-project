import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

df = pd.read_csv('wpd_datasets.csv')

df = df.iloc[1:].reset_index(drop=True)

df = df.apply(pd.to_numeric, errors='coerce')

wavelengths = df.columns[::2]

data = {}

for i, wl in enumerate(wavelengths):
    x = df.iloc[:, 2*i].dropna().values
    y = df.iloc[:, 2*i + 1].dropna().values
    
    data[wl] = (x, y)

xmin = max(x.min() for x, y in data.values())
xmax = min(x.max() for x, y in data.values())

x_common = np.linspace(xmin, xmax, 500)

interp_data = {}

for wl, (x, y) in data.items():
    f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
    interp_data[wl] = f(x_common)

interp_df = pd.DataFrame({'x': x_common})

for wl in wavelengths:
    interp_df[wl] = interp_data[wl]

interp_df.to_csv('interpolated_data.csv', index=False)

print("Done. Interpolated data saved.")

for col in interp_df.columns[1:]:
    plt.plot(interp_df['x'], interp_df[col], label=col)
plt.yscale('log')
plt.xlabel('log T')
plt.ylabel('Response Function [DN cm$^{-5}$ s$^{-1}$ px$^{-1}$]')
plt.title('Temperature Response Functions')
plt.xlim(5, 8)
plt.ylim(5e-29, 4e-24)
plt.legend()
plt.grid(True)
plt.show()