"""
Simulation: sub-nanometer 2D FDTD and angle-resoved Fano Engine
Polariton Optical Memory - Si3N4/perovskite metasurface

Parameters:
    -Spatial resolution: dx=dy=1.0nm (sub nanometer grid, 500*960 cells)
    -Time resolution : dt = 0.00235 fs(Courant stable for dx=1.0 nm)
    -Time steps: 50,000,000 steps per run (t_max = 117.5 ps)
    -Asymmetries swept: \delta w \in [50,40,30,20,15,10,6,4,2,1] nm(10 points)
    -Angle sweep: 0 \in [0^circ, 1^circ, 2^circ, 5^circ, 10^circ] angle-resolved band structure
    -Polarizations: TM (Hz=0, Ez active) and TE (Ez=0, Hz active)

Computed quantities and saved artifacts:
    -Direct fft transmission and reflection spectra
    -Spatial electric field intensity maps |E|^2(x,y) at resonance
    -Direct Q-factors Q=\lambda\dot\theta / FWHM and Fano assymetric line shape parameter q
    -Exported arrays: fdtd_master_2d_spectrum_dw{dw}_angle{theta}.npy
    
    
"""

import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time, json

OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "../plots"
os.makedirs(OUT_DIR, exist_ok=True)

#Constants
c_theta=299_792_458.0
mu_theta=4.0 * np.pi * 1e-7
eps_theta= 1.0 / (mu_theta * c_theta**2)
eta_theta = np.sqrt(mu_theta /eps_theta)

#Refractive indices at 800nm
n_SiN = 2.00; eps_SiN = 4.00
n_SiO2 = 1.45; eps_SiO2 = 2.1025
n_perov = 2.40; eps_perov = 5.7600

#sub-nanometer grid (dx=1.0 nm)
DX = 1.0e-9; DY =DX
DT = 0.99 / (c_theta * np.sqrt(1.0/DX**2 +1.0/DY**2))

NX = 500  #500 nm unit cell period (PBC in X)
NY = 960 #960 nm domain height
NPML = 60 #6O nm PML thickness (top and bottom)

Y_SUB_END = 320 #320 nm SiO2 substrate
Y_SIN_START = 320
Y_SIN_END = 720 #400 nm SiN height
Y_PEROV_END = 770 #50nm perovskite overlayer

J_SRC = 140 #source plane
J_TRANS =880 #transmission monitor
J_REFL = 80 #reflection monitor

def build_perm_grid_1nm(delta_w_nm):
    eps = np.ones((NX, NY), dtype=np.float64)
    eps[:, :Y_SUB_END] = eps_SiO2

    w1_cells = 150 #150 nm
    w2_cells = max(1, int(150 - delta_w_nm))
    gap_cells = 70 #70nm gap

    x1_l = 40
    x1_r = x1_l + w1_cells
    x2_l = x1_r + gap_cells
    x2_r = x2_l + w2_cells

    eps[x1_l:x1_r, Y_SIN_START:Y_SIN_END] = eps_SiN
    eps[x2_l:x2_r, Y_SIN_START:Y_SIN_END] = eps_SiN
    eps[:, Y_SIN_END:Y_PEROV_END] = eps_perov
    return eps

def build_cpml_1nm():
    m, R = 4, 1e-12
    sigma_max = -(m + 1) * np.log(R) / (2.0 * NPML * DY * eta_theta)

    sigma_y = np.zeros(NY, dtype=np.float64)
    sigma_y_h = np.zeros(NY, dtype=np.float64)
    for j in range(NPML):
        sigma_y[j] = ((NPML - j) / NPML) ** m * sigma_max
        sigma_y[NY - NPML + j] = ((j+1) / NPML) ** m * sigma_max
        sigma_y_h[j] = ((NPML - j - 0.5) / NPML) ** m * sigma_max
        sigma_y_h[NY - NPML + j] = ((j + 0.5) / NPML) ** m * sigma_max

    b_Ez = np.exp(-sigma_y * DT / eps_theta)[np.newaxis, :]
    b_Hx = np.exp(-sigma_y_h * DT / mu_theta)[np.newaxis, :]

    c_Ez = np.where(sigma_y >0, (1.0 - b_Ez) / (sigma_y * DY / eps_theta), np.zeros((1, NY)))