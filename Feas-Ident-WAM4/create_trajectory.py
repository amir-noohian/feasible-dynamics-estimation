import numpy as np

# ==== User Parameters ====
K = 7                   # Number of joints
L = 5                   # Number of Fourier terms
omega_f = 0.314         # Base frequency [rad/s]
duration = 60          # Total time [seconds]
frequency = 250         # Sampling frequency [Hz]
output_file = "data/trajectories/traj_4.dat"  # Output filename

# ==== Derived Parameters ====
h = 1.0 / frequency               # Time step [s]
num_points = int(duration * frequency)  # Number of time steps
t = np.linspace(0, duration, num_points)
omega_f_l = np.arange(1, L + 1) * omega_f  # Harmonics [w1, ..., wL]

# ==== Input Coefficients ====
a_b_q0 = np.array([
    -0.22453075, -0.06509415,  0.0084825,   0.7463596,   0.04482256,
     0.15292831,  0.17945343,  0.13698774,  0.15005761,  0.40428706,
    -0.06912397,  0.64039236,  0.41406122, -0.05345818,  0.46419411,
     0.27601025,  0.55370932,  0.065848,    0.04071695,  0.41272949,
    -0.17373824,  0.89269347,  0.45028845, -0.26077073,  0.10794414,
    -0.16322598,  0.22051963, -0.13005116,  0.38388683,  0.93214031,
     0.36874565,  0.11058303, -0.13207445,  0.39799543,  0.28705749,
     0.09974301, -0.09950086,  0.77226083, -0.28470714,  0.06468834,
     0.0187677,  -0.05244391, -0.40604485,  0.43133299, -0.57178278,
    -0.12185623,  0.51881781,  0.27173848,  0.15961459, -0.29374854,
     0.2251367,  -0.29917266,  0.09730888,  0.24762081,  0.07477825,
    -0.11031733,  0.08532278,  0.21811307, -0.10056108,  0.56254872,
    -0.02235595,  0.34816564, -0.24721517, -0.1987875,  -0.29918626,
     0.19194917, -0.39180923, -0.60461852,  0.00473514, -0.24540649,
    -0.24181945,  0.03114698, -0.08510632,  1.54641938, -1.67238949,
    -0.17570127,  0.5569734
])

# ==== Reshape Coefficients ====
a = a_b_q0[:K*L].reshape(K, L)
b = a_b_q0[K*L:2*K*L].reshape(K, L)
q0 = a_b_q0[2*K*L:]

# ==== Compute joint positions ====
q_t = np.zeros((num_points, K))
for k in range(K):
    for l in range(L):
        w = omega_f_l[l]
        q_t[:, k] += (a[k, l] / w) * np.sin(w * t) - (b[k, l] / w) * np.cos(w * t)
    q_t[:, k] += q0[k]

# ==== Save only the first 4 DoFs ====
np.savetxt(output_file, q_t[:, :4], fmt="%.6f")
print(f"Saved first 4 joint positions to '{output_file}' with shape {q_t[:, :4].shape}")
