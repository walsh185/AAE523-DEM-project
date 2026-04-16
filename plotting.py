import sunpy.map
import matplotlib.pyplot as plt

mapl1 = sunpy.map.Map("data/aia.lev1.211A_2023_06_01T00_00_11.89Z.image_lev1.fits")
mapl15 = sunpy.map.Map("data_15/aia_211_15.fits")

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection=mapl1)
mapl1.plot(axes=ax1)
ax1.set_title("Level 1 Data")

ax2 = fig.add_subplot(1, 2, 2, projection=mapl15)
mapl15.plot(axes=ax2)
ax2.set_title("Level 1.5 Data (Processed)")

plt.tight_layout()
plt.show()