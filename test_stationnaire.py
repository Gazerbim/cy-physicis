import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def init():
    line.set_data([], [])
    return line,


def animate(j):
    line.set_data(x_array, final_density[j, :])  # Crée un graphique pour chaque densite sauvegarde
    return line,


dt = 1E-7
dx = 0.001
nx = int(1 / dx) * 2
nt = 200000  # En fonction du potentiel il faut modifier ce parametre car sur certaines animations la particule atteins les bords
n_frames = int(nt / 1000) + 1  # nombre d image dans notre animation
s = dt / (dx ** 2)
v0 = -4000
e = 1  # Valeur du rapport E/V0
E = e * v0
k = math.sqrt(2 * abs(E))

x_array = np.linspace(0, (nx - 1) * dx, nx)
V_potential = np.zeros(nx)
largeur_potentiel = 0.1
largeur_normalisee = largeur_potentiel/2

for i in range(nx):
    if i>= nx/2-nx*largeur_normalisee and i<= nx/2+nx*largeur_normalisee:
        V_potential[i] = v0

# gaussian wave packet (Paquet ondes gaussien)
xc = 0.6
sigma = 0.05
normalisation = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
wp_gauss = normalisation * np.exp(1j * k * x_array - ((x_array - xc) ** 2) / (2 * (sigma ** 2)))
# wave packet Real part
wp_re = np.zeros(nx)
wp_re[:] = np.real(wp_gauss[:])
# wave packet Imaginary part
wp_im = np.zeros(nx)
wp_im[:] = np.imag(wp_gauss[:])

density = np.zeros((nt, nx))
density[0, :] = np.absolute(wp_gauss[:]) ** 2

final_density = np.zeros((n_frames, nx))

# Algo devant retourner la densité de probabilité de présence de la particule à différents instants
for t in range(1, nt):
    # Evolution de la partie réelle
    wp_re[1:-1] += s * (wp_im[2:] + wp_im[:-2] - 2 * wp_im[1:-1]) - dt * V_potential[1:-1] * wp_im[1:-1]
    # Evolution de la partie imaginaire
    wp_im[1:-1] -= s * (wp_re[2:] + wp_re[:-2] - 2 * wp_re[1:-1]) - dt * V_potential[1:-1] * wp_re[1:-1]

    norm = np.sqrt(np.sum(wp_re ** 2 + wp_im ** 2) * dx)
    wp_re /= norm
    wp_im /= norm

    # Densité de probabilité
    density[t, :] = wp_re ** 2 + wp_im ** 2

    # Sauvegarde pour l'animation toutes les 1000 itérations
    if t % 1000 == 0:
        final_density[int(t / 1000), :] = density[t, :]

plot_title = "E/Vo=" + str(e)

fig = plt.figure()  # initialise la figure principale
line, = plt.plot([], [])
plt.ylim(-1, 1)
plt.xlim(0, 2)
if (v0 == 0):
    v0 = 1
plt.plot(x_array, V_potential / abs(v0), 'r--', label="Potentiel (échelle réduite)")
plt.title(plot_title)
plt.xlabel("x")
plt.ylabel("Densité de probabilité de présence")
# plt.legend() #Permet de faire apparaitre la legende

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, blit=False, interval=100, repeat=False)
#file_name = 'paquet_onde_e='+str(e)+'.mp4'
#ani.save(file_name, writer = animation.FFMpegWriter(fps=20, bitrate=5000))
plt.show()

# -------------------------------
# CALCUL DES ÉTATS STATIONNAIRES
# -------------------------------

print("\nCalcul des états stationnaires...")

# Construction de la matrice Hamiltonienne
diag = 2.0 / dx**2 + V_potential
off_diag = -1.0 / dx**2 * np.ones(nx - 1)
H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

# Diagonalisation : énergies et fonctions d'onde
energies, wavefuncs = np.linalg.eigh(H)

# Normalisation des fonctions d'onde
normalized_wavefuncs = np.zeros_like(wavefuncs)
for n in range(wavefuncs.shape[1]):
    psi_n = wavefuncs[:, n]
    psi_n /= np.sqrt(np.sum(psi_n**2) * dx)
    normalized_wavefuncs[:, n] = psi_n

# ---------------------------------------------------
# AFFICHAGE DES DENSITÉS |ψ_n(x)|² UNIQUEMENT
# ---------------------------------------------------
N = 4  # nombre d'états à afficher
plt.figure(figsize=(10,6))
for n in range(N):
    density_n = normalized_wavefuncs[:, n]**2
    # on décale verticalement pour coller à l'énergie
    plt.plot(x_array, density_n + energies[n]/abs(v0),
             label=fr'$|\psi_{{{n}}}(x)|^2 + E_{{{n}}}$')

# Potentiel en fond
plt.plot(x_array, V_potential/abs(v0), 'r--', label='Potentiel (réduit)')
plt.title(r"Densités des états stationnaires $|\psi_n(x)|^2$")
plt.xlabel("x")
plt.ylabel(r"$|\psi_n|^2$ (décalé par énergie)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


