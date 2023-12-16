import numpy as np
import matplotlib.pyplot as plt


def get_sigma_u_s(T, t_b, t_c, q, t, M, z, H, c_v, eta, N, E_oed):
    m_v = 1 / E_oed
    A_0 = 2 * q / T * (t_b - t_c / 2)
    T_v = c_v * t / H ** 2
    sigma_0 = A_0 / 2

    sigma_temp = 0
    u_k = 0
    s_k_temp = 0
    s_k = 0

    for k in range(1, M):
        omega_k = (2 * k * np.pi) / T
        A_k = (q * T) / (2 * np.pi ** 2 * k ** 2 * t_c) * (
                    np.cos(omega_k * t_c) + omega_k * t_c * np.sin(omega_k * t_b) - 1)
        B_k = (q * T) / (2 * np.pi ** 2 * k ** 2 * t_c) * (
                    np.sin(omega_k * t_c) - omega_k * t_c * np.cos(omega_k * t_b))

        sigma_temp += A_k * np.cos(omega_k * t) + B_k * np.sin(omega_k * t)

        u_temp = 0
        for j in range(1, N):
            delta = eta * c_v / (omega_k * H ** 2)
            xi = (2 * j - 1) * np.pi / 2
            Y = ((A_k + B_k * delta * xi ** 2) * (np.cos(omega_k * t) - np.exp(-eta * xi ** 2 * T_v)) -
                 (A_k * delta * xi ** 2 - B_k) * np.sin(omega_k * t))
            u_temp += (-1) ** j / (xi + delta ** 2 * xi ** 5) * Y * np.cos(xi * z / H)
            s_k_temp += Y / (xi ** 2 + delta ** 2 * xi ** 6)

        u_k += u_temp
        s_k += m_v * H * (A_k * np.cos(omega_k * t) + B_k * np.sin(omega_k * t) - 2 * eta * s_k_temp)

    sigma = A_0 / 2 + sigma_temp
    u = -2 * eta * u_k
    s = m_v * H * sigma_0 + s_k
    return sigma, u, s


E_oed = 2400
z = 0.2
H = 1
c_v = (1 * 10**-7) * 60**2 * 24
t_b = 2.6 * H**2 / c_v
t_c = 20
T = 1.1 * t_b
t = np.arange(0, 400, 1)
q = 10
M = 100
N = 10
eta = 1


sigma, u, s = get_sigma_u_s(T, t_b, t_c, q, t, M, z, H, c_v, eta, N, E_oed)

fig, axs = plt.subplots(3, figsize=(10, 12.5))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

axs[0].plot(t, sigma, lw=0.5, color='b')
axs[1].plot(t, u, lw=0.5, color='b')
axs[2].plot(t, s, lw=0.5, color='b')


axs[0].grid(True)
# axs[0].set_ylim(-max(sigma_total)*0.01, max(sigma_total)*1.2)
# axs[0].set_ylim(-max(sigma_total)*0.01, 1.4)
# axs[0].set_xlim(0, 8)
axs[0].set_title("Stresses over time")
# axs[0].set_xlabel("time [years]")
axs[0].set_ylabel("$\sigma_v$   $[kN/m^2]$")


axs[1].grid(True)
# axs[1].set_ylim(-max(u_total)*0.01, max(u_total)*1.2) #turn this one on again
axs[1].set_ylim(-max(u)*0.01, max(u)*1.2)
# axs[1].set_xlim(0, 8)
axs[1].set_title("Pore pressures over time")
# axs[1].set_xlabel("time [years]")
axs[1].set_ylabel("$u$  $[kN/m^2]$")




axs[2].grid(True)
# axs[2].set_ylim(-max(s)*0.01, max(s)*1.2)
# axs[2].set_xlim(0, 8)
axs[2].set_title("Settlement over time")
axs[2].set_xlabel("time [days]")
axs[2].set_ylabel("settlement  $[mm]$")

plt.show()

