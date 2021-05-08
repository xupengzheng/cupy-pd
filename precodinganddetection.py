# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:18:13 2021

@author: Eric
"""
import cupy as cp
import time

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
 
scenario = cp.array([2,2])
SNR_scope = cp.arange(0,40,5)
N_MonterCarlo = 10000
Nsym = 1000000
BER_zfp = cp.ones([SNR_scope.shape[0]])
BER_mmsep = cp.ones([SNR_scope.shape[0]])
BER_zfd = cp.ones([SNR_scope.shape[0]])
BER_mmsed = cp.ones([SNR_scope.shape[0]])
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
        noise =(cp.random.randn(NR,1)+1j**cp.random.randn(NR,1))/2**0.5
        source = bisource((NR,Nsym))
        sourcesignal = modulation(source)
        #ZF PRECODING
        w_zf = cp.dot(H.conj().T,cp.linalg.pinv(cp.dot(H,H.conj().T)))
        nfro = cp.linalg.norm(w_zf, ord = 'fro')
        w_zf = w_zf*Pt**0.5/nfro
        receivesignal = cp.dot(cp.dot(H,w_zf),sourcesignal)+noise
        recsym = demodulation(receivesignal)
        zf_p_nume += cp.where((recsym-source) != 0)[0].size
        #ZF DETECTION
        receivesignal = (Pt/NR)**0.5*cp.dot(H,sourcesignal)+noise
        receivesignal = cp.dot(cp.dot(cp.linalg.pinv(cp.dot(H.conj().T,H)),H.conj().T),receivesignal)
        recsym = demodulation(receivesignal)
        zf_d_nume +=  cp.where((recsym-source) != 0)[0].size
        #MMSE PRECODING
        tempm = cp.dot(H,H.conj().T)+NR/Pt*cp.identity(NR)
        w_mmse = cp.dot(H.conj().T,cp.linalg.pinv(tempm))
        w_mmse = Pt**0.5*w_mmse/cp.linalg.norm(w_mmse,ord = 'fro')
        receivesignal = cp.dot(cp.dot(H,w_mmse),sourcesignal) + noise
        recsym = demodulation(receivesignal)
        mmse_p_nume += cp.where((recsym-source) != 0)[0].size
        #MMSE DETECTION
        receivesignal = (Pt/NR)**0.5*cp.dot(H,sourcesignal)+noise
        tempm = cp.dot(H.conj().T,H)+NR/Pt*cp.identity(NR)
        tempm = cp.linalg.pinv(tempm)
        receivesignal = cp.dot(cp.dot(tempm,H.conj().T),receivesignal)
        recsym = demodulation(receivesignal)
        mmse_d_nume +=  cp.where((recsym-source) != 0)[0].size       
    BER_zfp[ith_SNR] = float(zf_p_nume)/(N_MonterCarlo*Nsym*NR)
    BER_zfd[ith_SNR] = float(zf_d_nume)/(N_MonterCarlo*Nsym*NR)
    BER_mmsed[ith_SNR] = float(mmse_d_nume)/(N_MonterCarlo*Nsym*NR)
    BER_mmsep[ith_SNR] = float(mmse_p_nume)/(N_MonterCarlo*Nsym*NR)
time_end=time.time()
print('totally cost',time_end-time_start)