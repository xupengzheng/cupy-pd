# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:18:13 2021

@author: Eric
"""
import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt

time_start = time.time()

def generate_H(NR,NT):
    real = cp.random.randn(NR,NT)
    image = cp.random.randn(NR,NT)
    H = (real +1j*image)/2**0.5
    return H
           
def bisource(shape):
    out = cp.random.randint(0,2,shape)
    return out

def modulation(source):
    a = cp.zeros(source.shape)+1j*cp.zeros(source.shape)
    a[cp.where(source == 0)] =-1+1j*0
    a[cp.where(source == 1)] = 1+1j*0
    return a
     
def demodulation(receivesignal):
    a = cp.zeros(receivesignal.shape)
    a[cp.where(receivesignal.real >= 0)] = 1
    a[cp.where(receivesignal.real < 0)] = 0
    return a      

scenario = np.array([2,2])
SNR_scope = np.arange(0,35,5)

N_MonterCarlo = 5000
Nloop = 5
Nsym = 1000000
BER_zfp = np.ones([SNR_scope.shape[0]])
BER_mmsep = np.ones([SNR_scope.shape[0]])
BER_zfd = np.ones([SNR_scope.shape[0]])
BER_mmsed = np.ones([SNR_scope.shape[0]])
NT = int(scenario[0])
NR = int(scenario[1])
for ith_SNR,SNR in enumerate(SNR_scope):
    Pt = 10**(SNR/10)
    zf_p_nume = 0
    zf_d_nume = 0
    mmse_p_nume = 0
    mmse_d_nume = 0
    for k in range(N_MonterCarlo):
        H = generate_H(NR, NT)
        for nth_loop in range(Nloop):
            noise =(cp.random.randn(NR,Nsym)+1j**cp.random.randn(NR,Nsym))/2**0.5
            source = bisource((NR,Nsym))
            sourcesignal = modulation(source)
            #ZF PRECODING
            w_zf = cp.dot(H.conj().T,cp.linalg.pinv(cp.dot(H,H.conj().T)))
            nfro = cp.linalg.norm(w_zf, ord = 'fro')
            w_zf = w_zf*Pt**0.5/nfro
            recsym = cp.dot(cp.dot(H,w_zf),sourcesignal)+noise
            recsym = demodulation(recsym)
            zf_p_nume += cp.where((recsym-source) != 0)[0].size
            #ZF DETECTION
            receivesignal = (Pt/NR)**0.5*cp.dot(H,sourcesignal)+noise
            recsym  = cp.dot(cp.dot(cp.linalg.pinv(cp.dot(H.conj().T,H)),H.conj().T),receivesignal)
            recsym = demodulation(recsym)
            zf_d_nume +=  cp.where((recsym-source) != 0)[0].size
            #MMSE PRECODING
            tempm = cp.dot(H,H.conj().T)+NR/Pt*cp.identity(NR)
            w_mmse = cp.dot(H.conj().T,cp.linalg.pinv(tempm))
            w_mmse = Pt**0.5*w_mmse/cp.linalg.norm(w_mmse,ord = 'fro')
            recsym  = cp.dot(cp.dot(H,w_mmse),sourcesignal) + noise
            recsym = demodulation(recsym)
            mmse_p_nume += cp.where((recsym-source) != 0)[0].size
            #MMSE DETECTION
            receivesignal = (Pt/NR)**0.5*cp.dot(H,sourcesignal)+noise
            tempm = cp.dot(H.conj().T,H)+NR/Pt*cp.identity(NR)
            tempm = cp.linalg.pinv(tempm)
            recsym  = cp.dot(cp.dot(tempm,H.conj().T),receivesignal)
            recsym = demodulation(recsym )
            mmse_d_nume +=  cp.where((recsym-source) != 0)[0].size       
    BER_zfp[ith_SNR] = float(zf_p_nume)/(N_MonterCarlo*Nsym*NR*Nloop)
    BER_zfd[ith_SNR] = float(zf_d_nume)/(N_MonterCarlo*Nsym*NR*Nloop)
    BER_mmsed[ith_SNR] = float(mmse_d_nume)/(N_MonterCarlo*Nsym*NR*Nloop)
    BER_mmsep[ith_SNR] = float(mmse_p_nume)/(N_MonterCarlo*Nsym*NR*Nloop)
time_end=time.time()
print('totally cost',time_end-time_start)
fig, ax = plt.subplots()
ax.plot(SNR_scope,BER_zfp,label='ZF-P')
ax.plot(SNR_scope,BER_zfd,label = 'ZF-D')
ax.plot(SNR_scope,BER_mmsep,label='MMSE-P')
ax.plot(SNR_scope,BER_mmsed,label='MMSE-D')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.semilogy(SNR_scope,BER_zfp,label='ZF-P')
ax.semilogy(SNR_scope,BER_zfd,label = 'ZF-D')
ax.semilogy(SNR_scope,BER_mmsep,label='MMSE-P')
ax.semilogy(SNR_scope,BER_mmsed,label='MMSE-D')
ax.legend()
plt.show()






# import numpy as np
# import matplotlib.pyplot as plt

# x = np.linspace(0, 10, 500)
# y = np.sin(x)

# fig, ax = plt.subplots()

# # Using set_dashes() to modify dashing of an existing line
# line1, = ax.plot(x, y, label='Using set_dashes()')
# line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

# # Using plot(..., dashes=...) to set the dashing when creating a line
# line2, = ax.plot(x, y - 0.2, dashes=[6, 2], label='Using the dashes parameter')

# ax.legend()
# plt.show()