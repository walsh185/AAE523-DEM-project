import sunpy.map
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def remove_hot_pixels(image, threshold=5):
    """
    Replace hot pixels with neighborhood median

    threshold: how many times above median to be considered a hot pixel
    """
    
    # Median filtered version (3x3 neighborhood)
    med = median_filter(image, size=3)

    # Identify hot pixels (outliers)
    diff = image - med
    mask = (image <= 0) | (image > med + threshold*np.std(diff))

    # Replace hot pixels with median
    cleaned = image.copy()
    cleaned[mask] = med[mask]
    return cleaned

def find_lambda(K, g, dg, DEM0, a, f):
    lambdas = np.logspace(-60, -30, 50)
    
    best_lam = None
    best_diff = np.inf
    gt = np.linalg.inv(dg) @ g
    Kt = np.linalg.inv(dg) @ K
    gt = gt.reshape(-1, 1)

    for lam in lambdas:
        xi = solve_tikhonov(K, g, dg, DEM0, lam, f)
        diff = abs(np.linalg.norm(Kt@xi-gt)**2 - len(xi) * a)
        if diff < best_diff:
            best_diff = diff
            best_lam = lam

    xi = solve_tikhonov(K, g, dg, DEM0, best_lam, f)
    g_model = K @ xi
    plt.plot(g, 'o', label='data')
    plt.plot(g_model, 'x', label='model')
    plt.legend()
    plt.yscale('log')
    plt.show()
    print(best_lam)
    return best_lam

def solve_tikhonov(K, g, dg, x0, lam, f):
    g = np.linalg.inv(dg) @ g
    K = np.linalg.inv(dg) @ K
    g = g.reshape(-1, 1)

    KT_K = K.T @ K
    n = KT_K.shape[0]
    L = np.eye(n)
    A = KT_K + lam * L.T @ L
    b = K.T @ g + lam * L.T @ L @ x0

    x = np.linalg.solve(A, b)


    return x * f

maps = []
maps.append(sunpy.map.Map("data_15/aia_94_15.fits"))
maps.append(sunpy.map.Map("data_15/aia_131_15.fits"))
maps.append(sunpy.map.Map("data_15/aia_171_15.fits"))
maps.append(sunpy.map.Map("data_15/aia_193_15.fits"))
maps.append(sunpy.map.Map("data_15/aia_211_15.fits"))
maps.append(sunpy.map.Map("data_15/aia_335_15.fits"))

# Load interpolated response functions
df = pd.read_csv('interpolated_data.csv')

# Extract temperature grid
logT = df['x'].values   # shape: (N,)

# Extract response matrix (each column = one channel)
channels = df.columns[1:]   # skip 'x'
K = df[channels].values     # shape: (N, 6)
K = K.T

print("logT shape:", logT.shape)
print("K shape:", K.shape)

#ny, nx = maps[0].data.shape

#i0 = ny // 2
#j0 = nx // 2
i0 = 1650
j0 = 1000
p = 30

# Extract 10x10 region (±5 pixels)
cutouts = []
for m in maps:
    cutout = m.data[i0-p:i0+p, j0-p:j0+p]
    cutouts.append(cutout)

# Plot
fig, axes = plt.subplots(2, 3, figsize=(10, 6))

for ax, img, m in zip(axes.flatten(), cutouts, maps):
    im = ax.imshow(np.log10(img), origin='lower')
    ax.set_title(f'{m.wavelength.value:.0f} Å')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

clean_cutouts = []

for c in cutouts:
    clean_cutouts.append(remove_hot_pixels(c))
    
# Plot
fig, axes = plt.subplots(2, 3, figsize=(10, 6))

for ax, img, m in zip(axes.flatten(), clean_cutouts, maps):
    im = ax.imshow(np.log10(img), origin='lower')
    ax.set_title(f'{m.wavelength.value:.0f} Å')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()


ny, nx = clean_cutouts[0].shape
EM_map = np.zeros((ny, nx))
for i in range(ny):
    for j in range(nx):
        # شدت vector (6 channels)
        I = np.array([c[i, j] for c in clean_cutouts])
        I = I.reshape(-1, 1)

        # Build dg
        sigma = 0.2 * I.flatten()
        sigma = np.where(sigma == 0, 1e-10, sigma)
        dg = np.diag(sigma)

        # Initial guess
        DEM0 = np.zeros((K.shape[1], 1))

        # Solve DEM
        f = 1
        lam = 1.e-50
        DEM = solve_tikhonov(K, I, dg, DEM0, lam, f)

        f = 1
        lam = 1.e-50
        DEM = solve_tikhonov(K, I, dg, DEM, lam, f)

        DEM = DEM.squeeze()

        # Compute EM
        T = 10**logT
        dlogT = np.gradient(logT)

        EM = np.sum(DEM * np.log(10) * T * dlogT)

        EM_map[i, j] = EM



plt.figure(figsize=(6,5))
plt.imshow(np.log10(EM_map), origin='lower', cmap='inferno')
plt.colorbar(label='log EM [cm$^{-5}$]')
plt.title('Emission Measure Map')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.show()

