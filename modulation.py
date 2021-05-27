# -*- coding: utf-8 -*-
"""
Created on Thu May 27 09:35:31 2021

@author: Eric
"""
import cupy as cp
import math
import numpy as np

def grayCode(n):
        """
        :type n: int
        :rtype: List[int]
        """
        return [i^(i>>1) for i in range(2**n)]
    
def binlist2int(list_):
    return sum(1 << i for (i, b) in enumerate(list_) if b != 0)



def int2binlist(int_, width=None):
    if width is None:
        width = max(int_.bit_length(), 1)
    return [(int_ >> i) & 1 for i in range(width)]



class Modulation:
    def __init__(self, constellation, labeling):
        self.constellation = constellation #a1+jb1,a2+jb2,...,aM+bM
        self.labeling = labeling #label:0,1,2,...,M-1
        self.mod_dict = {label: cons for label,cons in zip(self.labeling,self.constellation)}
        self.bits_per_symbol = int(math.log2(self.constellation.size))
       


    def bits_to_symbols(self, bits):
        n_symbols = int(bits.size/self.bits_per_symbol)
        symbols = cp.empty(n_symbols, dtype=cp.int_)
        for i, bit_sequence in enumerate(cp.reshape(bits, newshape=(n_symbols,self.bits_per_symbol))):
            symbols[i] = binlist2int(bit_sequence)
        return symbols

    def symbols_to_bits(self, symbols):
        m = self.bits_per_symbol
        n_bits = len(symbols) * m
        bits = cp.empty(n_bits, dtype=cp.int_)
        for i, symbol in enumerate(symbols):
            bits[i*m : (i + 1)*m] = int2binlist(symbol,m)
        return bits

    def modulate(self, bits): #symbol2signal
        """
        Modulates a sequence of bits to its corresponding constellation points.
        """
        symbols = self.bits_to_symbols(bits)
        signal = cp.empty([symbols.size], dtype=cp.complex_)
        for ith_symbol,symbol in enumerate(symbols):
            signal[ith_symbol] = self.mod_dict[int(symbol)]
        return signal

    def demodulate(self, received, decision_method='hard'):#signal2symbol
        """
        Demodulates a sequence of received points to a sequence of bits.
        """
        



class PSKModulation(Modulation):
    def __init__(self, order, amplitude=1.0, phase_offset=0.0, labeling='reflected'):
        constellation = amplitude * cp.exp(2j*cp.pi*cp.arange(order) / order) * cp.exp(1j * phase_offset)
        labeling = grayCode(int(math.log2(order)))
        super().__init__(cp.array(constellation, dtype=cp.complex_), labeling)


a = PSKModulation(16)
signal = a.modulate(cp.random.randint(0,2,[1000000]))
