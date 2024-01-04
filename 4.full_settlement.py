import numpy as np
import matplotlib.pyplot as plt

'''
function to determine the settlement
'''
def get_sigma_u_s(T, t_b, t_c, q, t, M, z, H, c_v, eta, N, E_oed):
    m_v = 1 / E_oed
    A_0 = 2 * q / T * (t_b - t_c / 2)
    T_v = c_v * t / H ** 2
    sigma_0 = A_0 / 2

    sigma_temp = 0
    u_k = 0
    s_k_temp = 0

    for k in range(1, M):
        s_k = 0

        omega_k = (2 * k * np.pi) / T
        A_k = (q * T) / (2 * np.pi ** 2 * k ** 2 * t_c) * (
                    np.cos(omega_k * t_c) + omega_k * t_c * np.sin(omega_k * t_b) - 1)
        B_k = (q * T) / (2 * np.pi ** 2 * k ** 2 * t_c) * (
                    np.sin(omega_k * t_c) - omega_k * t_c * np.cos(omega_k * t_b))

        sigma_temp += A_k * np.cos(omega_k * t) + B_k * np.sin(omega_k * t)

        delta = eta * c_v / (omega_k * H ** 2)
        u_temp = 0
        for j in range(1, N):
            xi = (2 * j - 1) * np.pi / 2
            Y = ((A_k + B_k * delta * xi ** 2) * (np.cos(omega_k * t) - np.exp(-eta * xi ** 2 * T_v)) -
                 (A_k * delta * xi ** 2 - B_k) * np.sin(omega_k * t))
            u_temp += (-1) ** j / (xi + delta ** 2 * xi ** 5) * Y * np.cos(xi * z / H)
            s_k_temp += Y / (xi**2 + delta**2 * xi**6)

        u_k += u_temp
        s_k += m_v * H * (A_k * np.cos(omega_k * t) + B_k * np.sin(omega_k * t) - 2 * eta * s_k_temp)

    sigma = A_0 / 2 + sigma_temp
    u = -2 * eta * u_k
    s = m_v * H * sigma + s_k
    return sigma, u, s

'''
parmeter definition
'''
# soil parametes valid for all layers
gamma = 20
gamma_prime = 10
eta = 1
eff_stress_lab = [4.5, 10.4, 16.4, 31.3, 46.1, 75.9, 135.4, 195.0]
c_v_lab = [3.44*10**-5, 6.70*10**-6, 3.73*10**-7, 1.70*10**-7,
           7.57*10**-8, 5.06*10**-8, 3.14*10**-8, 1.30*10**-8]
E_oed_lab = [208.00, 91.23, 66.95, 136.52, 198.72, 299.99, 596.18, 1154.79]

# time definition
t_b = 9
T = 1.1 * t_b
t = np.arange(0, T, 1/12)

# values for hight
z = 0
H = 2.5

# loop maxima
M = 200
N = 100

'''
setting up the plot and plotting the loading
'''
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
axs[0].set_ylabel("reclamation hight  $[m]$")
fig.suptitle('Loading over time for $\eta=1.0$')

'''
first loading
'''
t_c = 3/12
q = 0.5 * gamma_prime
c_v = (np.interp(5, eff_stress_lab, c_v_lab)) * 60**2 * 24 * 365
E_oed = np.interp(5, eff_stress_lab, E_oed_lab)
print('2. Loading:')
print(f'The c_v used for the 1.loading is:\n      {c_v}')
print(f'The E_oed used for the 1.loading is:\n      {E_oed}')
print()

sigma_1, u_1, s_1 = get_sigma_u_s(T, t_b, t_c, q, t, M, z, H, c_v, eta, N, E_oed)
sigma_total = sigma_1
u_total = u_1
s_total = s_1

'''
second loading
'''
t_c = 3/12
q = 1.5 * gamma_prime
c_v = (np.interp(20, eff_stress_lab, c_v_lab)) * 60**2 * 24 * 365
E_oed = np.interp(20, eff_stress_lab, E_oed_lab)
print('2. Loading:')
print(f'The c_v used for the 2.loading is:\n      {c_v}')
print(f'The E_oed used for the 2.loading is:\n      {E_oed}')
print()

sigma_2, u_2, s_2 = get_sigma_u_s(T, t_b, t_c, q, t, M, z, H, c_v, eta, N, E_oed)
sigma_total[6::] += sigma_2[0:-6]

u_total[6::] += u_2[0:-6]
s_total[6::] += s_2[0:-6]

'''
third loading
'''
t_c = 4/12
q = 1 * gamma
c_v = (np.interp(40, eff_stress_lab, c_v_lab)) * 60**2 * 24 * 365
E_oed = np.interp(40, eff_stress_lab, E_oed_lab)
print('3. Loading:')
print(f'The c_v used for the 3.loading is:\n      {c_v}')
print(f'The E_oed used for the 3.loading is:\n      {E_oed}')
print()

sigma_3, u_3, s_3 = get_sigma_u_s(T, t_b, t_c, q, t, M, z, H, c_v, eta, N, E_oed)
sigma_total[24::] += sigma_3[0:-24]
u_total[24::] += u_3[0:-24]
s_total[24::] += s_3[0:-24]
s_total = s_total * 2   # due to double drainage
s_total_eta_1 = s_total
u_total_eta_1 = u_total

'''
plot
'''
axs[1].plot(t, sigma_total, lw=1, color='r', label='total stress')
axs[2].plot(t, u_total, lw=1, color='r', label='total pore pressure')
sigma_prime = sigma_total - u_total
axs[3].plot(t, sigma_prime, lw=1, color='r', label='effective stress')
axs[4].plot(t, s_total, lw=1, color='r', label='settlement')

'''
plot setup
'''

axs[1].grid(True)
axs[1].set_ylim(min(sigma_total), max(sigma_total)*1.2)
axs[1].set_xlim(0, 8)
axs[1].set_title("Stresses over time")
axs[1].set_ylabel("$\sigma_v$   $[kN/m^2]$")


axs[2].grid(True)
axs[2].set_ylim(-max(u_total)*0.01, max(u_total)*1.2)
axs[2].set_xlim(0, 8)
axs[2].set_title("Pore pressures over time")
axs[2].set_ylabel("$u$  $[kN/m^2]$")


axs[3].grid(True)
axs[3].set_ylim(-max(sigma_prime)*0.01, max(sigma_prime)*1.2)
axs[3].set_xlim(0, 8)
axs[3].set_title("Effective stress over time")
axs[3].set_ylabel("$\sigma'_v$   $[kN/m^2]$")


axs[4].grid(True)
axs[4].set_ylim(max(s_total)*1.1, 0)
axs[4].set_xlim(0, 8)
axs[4].set_title("Settlement over time")
axs[4].set_xlabel("time [years]")
axs[4].set_ylabel("settlement  $[m]$")

plt.show()




'''
setting up the plot and plotting the loading
'''
eta = 0.8
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
axs[0].set_ylabel("reclamation hight  $[m]$")
fig.suptitle('Loading over time for $\eta=0.8$')

'''
first loading
'''
t_c = 3/12
q = 0.5 * gamma_prime
c_v = (np.interp(5, eff_stress_lab, c_v_lab)) * 60**2 * 24 * 365
E_oed = np.interp(5, eff_stress_lab, E_oed_lab)
print('2. Loading:')
print(f'The c_v used for the 1.loading is:\n      {c_v}')
print(f'The E_oed used for the 1.loading is:\n      {E_oed}')
print()

sigma_1, u_1, s_1 = get_sigma_u_s(T, t_b, t_c, q, t, M, z, H, c_v, eta, N, E_oed)
sigma_total = sigma_1
u_total = u_1
s_total = s_1

'''
second loading
'''
t_c = 3/12
q = 1.5 * gamma_prime
c_v = (np.interp(20, eff_stress_lab, c_v_lab)) * 60**2 * 24 * 365
E_oed = np.interp(20, eff_stress_lab, E_oed_lab)
print('2. Loading:')
print(f'The c_v used for the 2.loading is:\n      {c_v}')
print(f'The E_oed used for the 2.loading is:\n      {E_oed}')
print()

sigma_2, u_2, s_2 = get_sigma_u_s(T, t_b, t_c, q, t, M, z, H, c_v, eta, N, E_oed)
sigma_total[6::] += sigma_2[0:-6]

u_total[6::] += u_2[0:-6]
s_total[6::] += s_2[0:-6]

'''
third loading
'''
t_c = 4/12
q = 1 * gamma
c_v = (np.interp(40, eff_stress_lab, c_v_lab)) * 60**2 * 24 * 365
E_oed = np.interp(40, eff_stress_lab, E_oed_lab)
print('3. Loading:')
print(f'The c_v used for the 3.loading is:\n      {c_v}')
print(f'The E_oed used for the 3.loading is:\n      {E_oed}')
print()

sigma_3, u_3, s_3 = get_sigma_u_s(T, t_b, t_c, q, t, M, z, H, c_v, eta, N, E_oed)
sigma_total[24::] += sigma_3[0:-24]
u_total[24::] += u_3[0:-24]
s_total[24::] += s_3[0:-24]
s_total = s_total * 2   # due to double drainage

'''
plot
'''
axs[1].plot(t, sigma_total, lw=1, color='r', label='total stress')
axs[2].plot(t, u_total, lw=1, color='r', label='total pore pressure')
sigma_prime = sigma_total - u_total
axs[3].plot(t, sigma_prime, lw=1, color='r', label='effective stress')
axs[4].plot(t, s_total, lw=1, color='r', label='settlement')

'''
plot setup
'''

axs[1].grid(True)
axs[1].set_ylim(min(sigma_total), max(sigma_total)*1.2)
axs[1].set_xlim(0, 8)
axs[1].set_title("Stresses over time")
axs[1].set_ylabel("$\sigma_v$   $[kN/m^2]$")


axs[2].grid(True)
axs[2].set_ylim(-max(u_total)*0.01, max(u_total)*1.2)
axs[2].set_xlim(0, 8)
axs[2].set_title("Pore pressures over time")
axs[2].set_ylabel("$u$  $[kN/m^2]$")


axs[3].grid(True)
axs[3].set_ylim(-max(sigma_prime)*0.01, max(sigma_prime)*1.2)
axs[3].set_xlim(0, 8)
axs[3].set_title("Effective stress over time")
axs[3].set_ylabel("$\sigma'_v$   $[kN/m^2]$")


axs[4].grid(True)
axs[4].set_ylim(max(s_total)*1.1, 0)
axs[4].set_xlim(0, 8)
axs[4].set_title("Settlement over time")
axs[4].set_xlabel("time [years]")
axs[4].set_ylabel("settlement  $[m]$")

plt.show()


'''
comparison
'''

fig, axs = plt.subplots(1, figsize=(10, 2.5))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

axs.plot(t, u_total_eta_1, lw=1, color='orange', label='u at $\eta=1.0$')
axs.plot(t, u_total, lw=1, color='blue', label='u at $\eta=0.8$')

axs.grid(True)
axs.set_ylim(-max(u_total_eta_1)*0.01, max(u_total_eta_1)*1.2)
axs.set_xlim(0, 8)
axs.set_title("Pore pressure over time for $\eta=1.0$ and $\eta=0.8$")
axs.set_xlabel("time [years]")
axs.set_ylabel("$u$  $[kN/m^2]$")
plt.legend(loc="upper right")
plt.show()



fig, axs = plt.subplots(1, figsize=(10, 2.5))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

axs.plot(t, s_total_eta_1, lw=1, color='orange', label='settlement $\eta=1.0$')
axs.plot(t, s_total, lw=1, color='blue', label='settlement $\eta=0.8$')

axs.grid(True)
axs.set_ylim(max(s_total)*1.1, 0)
axs.set_xlim(0, 8)
axs.set_title("Settlement over time for $\eta=1.0 and \eta=0.8$")
axs.set_xlabel("time [years]")
axs.set_ylabel("settlement  $[m]$")
plt.legend(loc="upper right")
plt.show()


