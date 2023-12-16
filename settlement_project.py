import numpy as np
import matplotlib.pyplot as plt


def get_sigma_u_s(T, t_b, t_c, q, t, k_max, z, H, c_v, eta, N, E_oed):
    m_v = 1 / E_oed
    A_0 = 2 * q / T * (t_b - t_c / 2)
    T_v = c_v * t / H ** 2

    sigma_temp = 0
    u_temp = 0
    s_k_temp = 0

    for k in range(1, k_max):
        s_k = 0

        omega_k = (2 * k * np.pi) / T
        A_k = (q * T) / (2 * np.pi ** 2 * k ** 2 * t_c) * (
                    np.cos(omega_k * t_c) + omega_k * t_c * np.sin(omega_k * t_b) - 1)
        B_k = (q * T) / (2 * np.pi ** 2 * k ** 2 * t_c) * (
                    np.sin(omega_k * t_c) - omega_k * t_c * np.cos(omega_k * t_b))

        sigma_temp += A_k * np.cos(omega_k * t) + B_k * np.sin(omega_k * t)

        delta = eta * c_v / (omega_k * H ** 2)

        for j in range(1, N):
            xi = (2 * j - 1) * np.pi / 2
            Y = ((A_k + B_k * delta * xi**2) * (np.cos(omega_k * t) - np.exp(-eta * xi**2 * T_v)) -
                 (A_k * delta * xi**2 - B_k) * np.sin(omega_k * t))
            u_temp += (-1) ** j / (xi + delta**2 * xi**5) * Y * np.cos(xi * z / H)
            s_k_temp += Y / (xi**2 + delta**2 * xi**6)

        s_k += A_k * np.cos(omega_k * t) + B_k * np.sin(omega_k * t) - 2 * eta * s_k_temp

    u = -2 * eta * u_temp
    sigma = A_0/2 + sigma_temp
    s = -(m_v * H * sigma + m_v * H * s_k) * 2

    return sigma, u, s


gamma = 20
gamma_prime = 10
E_oed = 1200


k_max = 20
t_b = 9
T = 1.1 * t_b
t = np.arange(0, T, 1/12)

z = 0
H = 2.5
c_v = (5 * 10**-6) * 60**2 * 24 * 365
eta = 1
N = 10

fig, axs = plt.subplots(5, figsize=(10, 12.5))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

hight = np.arange(0, T, 1/12) * 0
hight[1:] = 0.5
hight[7] = 1
hight[8] = 1.5
hight[9:] = 2
hight[25] = 2.25
hight[26] = 2.5
hight[27] = 2.75
hight[28:] = 3



axs[0].grid(True)
axs[0].plot(t, hight, lw=0.5, color='black')
axs[0].set_ylim(-max(hight)*0.01, max(hight)*1.2)
axs[0].set_xlim(0, 8)
axs[0].set_title("Reclamation hight")
# axs[0].set_xlabel("time [years]")
axs[0].set_ylabel("reclamation hight  $[m]$")
# fig.suptitle('Loading over time')

## 1.loading
t_c = 0.5/12
q = 0.5 * gamma_prime

E_oed = 2400
z = 0.2
H = 1
c_v = (1 * 10**-7) * 60**2 * 24
t_b = 2.6 * H**2 / c_v
t_c = 0.13 * t_b
T = 1.1 * t_b
t = np.arange(0, 400, 1)
q = 1

sigma, u, s_1 = get_sigma_u_s(T, t_b, t_c, q, t, k_max, z, H, c_v, eta, N, E_oed)
u_total = u
sigma_total = sigma
s_total = s_1
axs[1].plot(t, sigma, lw=0.5, color='b')
axs[2].plot(t, u, lw=0.5, color='b')
axs[4].plot(t, s_1, lw=0.5, color='b')


'''
## 2.loading
t_c = 3/12
q = 1.5 * gamma_prime

sigma, u, s_2 = get_sigma_u_s(T, t_b, t_c, q, t, k_max, z, H, c_v, eta, N, m_v)
u_total[6::] += u[0:-6]
sigma_total[6::] += sigma[0:-6]
s_total[6::] += s_2[0:-6]
axs[1].plot(t+0.5, sigma, lw=0.5, color='b')
axs[2].plot(t+0.5, u, lw=0.5, color='b')
axs[4].plot(t+0.5, s_2, lw=0.5, color='b')


## 3.loading
t_c = 4/12
q = 1 * gamma

sigma, u, s_3 = get_sigma_u_s(T, t_b, t_c, q, t, k_max, z, H, c_v, eta, N, m_v)
u_total[24::] += u[0:-24]
sigma_total[24::] += sigma[0:-24]
s_total[24::] += s_3[0:-24]
axs[1].plot(t+2, sigma, lw=0.5, color='b')
axs[2].plot(t+2, u, lw=0.5, color='b')
axs[4].plot(t+2, s_3, lw=0.5, color='b')

axs[1].plot(t, sigma_total, lw=1, color='r', label='total stress')
axs[2].plot(t, u_total, lw=1, color='r', label='total pore pressure')
'''

axs[1].grid(True)
# axs[1].set_ylim(-max(sigma_total)*0.01, max(sigma_total)*1.2)
axs[1].set_ylim(-max(sigma_total)*0.01, 1.4)
#axs[1].set_xlim(0, 8)
axs[1].set_title("Stresses over time")
# axs[1].set_xlabel("time [years]")
axs[1].set_ylabel("$\sigma_v$   $[kN/m^2]$")


axs[2].grid(True)
# axs[2].set_ylim(-max(u_total)*0.01, max(u_total)*1.2) #turn this one on again
axs[2].set_ylim(-max(u_total)*0.01, max(u)*1.2)
#axs[2].set_xlim(0, 8)
axs[2].set_title("Pore pressures over time")
# axs[2].set_xlabel("time [years]")
axs[2].set_ylabel("$u$  $[kN/m^2]$")


sigma_prime = sigma_total - u_total
axs[3].plot(t, sigma_prime, lw=1, color='r', label='effective stress')
axs[3].grid(True)
axs[3].set_ylim(-max(sigma_prime)*0.01, max(sigma_prime)*1.2)
#axs[3].set_xlim(0, 8)
axs[3].set_title("Effective stress over time")
# axs[3].set_xlabel("time [years]")
axs[3].set_ylabel("$\sigma'_v$   $[kN/m^2]$")

# s = -2.5 * sigma_prime / E_oed * 1000
axs[4].plot(t, s_total, lw=1, color='r', label='settlement')
axs[4].grid(True)
# axs[4].set_ylim(-max(s)*0.01, max(s)*1.2)
#axs[4].set_xlim(0, 8)
axs[4].set_title("Settlement over time")
axs[4].set_xlabel("time [years]")
axs[4].set_ylabel("settlement  $[mm]$")


plt.show()