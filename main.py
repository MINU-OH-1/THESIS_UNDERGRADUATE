import sympy
import numpy as np
import pandas as pd
from collections import deque
from queue import PriorityQueue as pq

RHO = 1.025 #ton / m^3
PERMISSIBLE_ERROR = 0.000001 
VLD = 0.000001
MAX_BALLAST_CASE = 1

# Fixed Value #
L = 76
B = 26
H = 5.1
W = 1000
###############

class Compartment:
    def __init__(self, x, y, z, dx, dy, dz, density, rate = 0):
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.density = density
        self.rate = rate

    def SetRate(self, rate):
        self.rate = rate

    def GetCrossSectionTank(self, psi, delta, zq, zr, zl, zh, zt, type):
        special_case = zl > zh
        v = 0
        origin_shift = np.zeros(3)
        f = 12 * np.sin(psi) * np.cos(psi) * np.tan(delta)
        moment_of_volume = np.zeros(3)
        yt = zh * np.cos(psi) - zl * np.tan(psi)

        if type == 1:
            v = zq ** 3
            moment_of_volume[2] = zq ** 4 / f
            moment_of_volume[1] = (1/np.tan(2*psi)) * moment_of_volume[2]
            moment_of_volume[0] = (1/2/np.tan(delta)) * moment_of_volume[2]
        
        elif type == 2:
            v = 3 * zl ** 2 * (zq - zl)
            moment_of_volume[2] = 2 * zl / 3 * v
            moment_of_volume[1] = (1/np.tan(2*psi)) * moment_of_volume[2]
            moment_of_volume[0] = ((zq - zl)/2/np.tan(delta)) * moment_of_volume[2]
        
        elif type == 3:
            v = zl ** 3
            origin_shift[0] = (zq - zl) / np.tan(delta)
            moment_of_volume[2] = zl ** 4 / f
            moment_of_volume[1] = (1/np.tan(2*psi)) * moment_of_volume[2]
            moment_of_volume[0] = ((zq - zl)/2/np.tan(delta)) * moment_of_volume[2]
        
        elif type == 4:
            v = -zr ** 3
            origin_shift[0] = -self.dx
            moment_of_volume[2] = -zr ** 4 / f
            moment_of_volume[1] = -(1/np.tan(2*psi)) * moment_of_volume[2]
            moment_of_volume[0] = -(1/2/np.tan(delta)) * moment_of_volume[2]
        
        elif type == 5:
            v = 3 * zl * (zq - zl) ** 2
            origin_shift[1] = zl / np.tan(2*psi)
            origin_shift[2] = zl
            moment_of_volume[2] = (zq-zl) / 3 * v
            if not special_case:
                moment_of_volume[1] = (1/np.tan(psi)) * moment_of_volume[2]
            else:
                moment_of_volume[1] =  -np.tan(psi) * moment_of_volume[2]
            moment_of_volume[0] = ((zq - zl)/2/np.tan(delta)) * moment_of_volume[2]
        
        elif type == 6:
            v = 6 * zl * (zh - zl) * (zq -zh)
            origin_shift[1] = zl / np.tan(2 * psi)
            origin_shift[2] = zl
            moment_of_volume[2] = (zh - zl) / 2 * v
            if not special_case:
                moment_of_volume[1] = (1/np.tan(psi)) * moment_of_volume[2]
            else:
                moment_of_volume[1] =  -np.tan(psi) * moment_of_volume[2]
            moment_of_volume[0] = ((zq - zh)/2/np.tan(delta)) * moment_of_volume[2]
        
        elif type == 7:
            v = 3 * zl * (zh - zl) ** 2
            origin_shift[0] = (zq - zh) / np.tan(delta)
            origin_shift[1] = zl / np.tan(2 * psi)
            origin_shift[2] = zl
            moment_of_volume[2] = (zh - zl) / 3 * v
            if not special_case:
                moment_of_volume[1] = (1/np.tan(psi)) * moment_of_volume[2]
            else:
                moment_of_volume[1] =  -np.tan(psi) * moment_of_volume[2]
            moment_of_volume[0] = 1 /np.tan(delta) * moment_of_volume[2]
        
        elif type == 8:
            v = 3 * zl * (zq - zr) ** 2
            origin_shift[0] = 0
            origin_shift[1] = zl / np.tan(2 * psi)
            if not special_case:
                origin_shift[1] += (zr - zl) / np.tan(psi)
            else:
                origin_shift[1] -= (zr - zl) * np.tan(psi)
            origin_shift[2] = zr
            moment_of_volume[2] = (zq - zr) / 3 * v
            if not special_case:
                moment_of_volume[1] = (1/np.tan(psi)) * moment_of_volume[2]
            else:
                moment_of_volume[1] =  -np.tan(psi) * moment_of_volume[2]
            moment_of_volume[0] = 1 /np.tan(delta) * moment_of_volume[2]
        
        elif type == 9:
            v = 6 * zl * (zh - zr) * (zq - zh)
            origin_shift[0] = 0
            origin_shift[1] = zl / np.tan(2 * psi)
            if not special_case:
                origin_shift[1] += (zr - zl) / np.tan(psi)
            else:
                origin_shift[1] -= (zr - zl) * np.tan(psi)
            origin_shift[2] = zr
            moment_of_volume[2] = (zh - zr) / 2 * v
            if not special_case:
                moment_of_volume[1] = (1/np.tan(psi)) * moment_of_volume[2]
            else:
                moment_of_volume[1] =  -np.tan(psi) * moment_of_volume[2]
            moment_of_volume[0] = (zq - zh) / 2 / np.tan(delta) * v

        elif type == 10:
            v = 3 * zl * (zh - zr) ** 2
            origin_shift[0] = (zq - zh) / np.tan(delta)
            origin_shift[1] = zl / np.tan(2 * psi)
            if not special_case:
                origin_shift[1] += (zr - zl) / np.tan(psi)
            else:
                origin_shift[1] -= (zr - zl) * np.tan(psi)
            origin_shift[2] = zr
            moment_of_volume[2] = (zh - zr) / 3 * v
            if not special_case:
                moment_of_volume[1] = (1/np.tan(psi)) * moment_of_volume[2]
            else:
                moment_of_volume[1] =  -np.tan(psi) * moment_of_volume[2]
            moment_of_volume[0] = 1 / np.tan(delta) * moment_of_volume[2]

        elif type == 11:
            v = (zq - zh) ** 2 * (3 * zt - zq - 2 * zh)
            origin_shift[0] = 0
            origin_shift[1] = yt - zl / np.tan(2 * psi)
            origin_shift[2] = zh
            moment_of_volume[2] = (2 * zt - zq - zh) * (zq -zh) ** 3 / f
            moment_of_volume[1] = moment_of_volume[2] / np.tan(2 * psi)
            moment_of_volume[0] = (4 * zt - zq - 3 * zh) / (2 * np.tan(delta) * (2 * zt - zq - zh)) * moment_of_volume[2]

        elif type == 12:
            v = (zq - zr) ** 2 * (3 * zt - zq - 2 * zr)
            origin_shift[0] = 0
            origin_shift[1] = yt - (zt - zr) / np.tan(2 * psi)
            origin_shift[2] = zr
            moment_of_volume[2] = (2 * zt - zq - zr) * (zq -zr) ** 3 / f
            moment_of_volume[1] = moment_of_volume[2] / np.tan(2 * psi)
            moment_of_volume[0] = (4 * zt - zq - 3 * zr) / (2 * np.tan(delta) * (2 * zt - zq - zr)) * moment_of_volume[2]

        elif type == 13:
            v = 3 * (zt - zh) ** 2 * (zq - zt)
            origin_shift[0] = 0
            origin_shift[1] = yt - zl / np.tan(2 * psi)
            origin_shift[2] = zh
            moment_of_volume[2] = (zt - zh) / 3 * v
            moment_of_volume[1] = moment_of_volume[2] / np.tan(2 * psi)
            moment_of_volume[0] = (zq - zt) / (2 * np.tan(delta)) * v

        elif type == 14:
            v = 2 * (zt - zh) ** 3
            origin_shift[0] = (zq - zt) / np.tan(delta)
            origin_shift[1] = yt - zl / np.tan(2 * psi)
            origin_shift[2] = zh
            moment_of_volume[2] = (zt - zh) ** 4 / f
            moment_of_volume[1] = moment_of_volume[2] / np.tan(2 * psi)
            moment_of_volume[0] = 3 * moment_of_volume[2] / (2 * np.tan(delta))
        
        elif type == 15:
            v = 3 * (zt - zr) ** 2 * (zq - zt)
            origin_shift[0] = 0
            origin_shift[1] = yt - (zt - zr) / np.tan(2 * psi)
            origin_shift[2] = zr
            moment_of_volume[2] = (zt - zr) / 3 * v
            moment_of_volume[1] = moment_of_volume[2] / np.tan(2 * psi)
            moment_of_volume[0] = (zq - zt) / (2 * np.tan(delta)) * v

        elif type == 16:
            v = 2 * (zt - zr) ** 3
            origin_shift[0] = (zq - zt) / np.tan(delta)
            origin_shift[1] = yt - (zt - zl) / np.tan(2 * psi)
            origin_shift[2] = zr
            moment_of_volume[2] = (zt - zr) ** 4 / f
            moment_of_volume[1] = moment_of_volume[2] / np.tan(2 * psi)
            moment_of_volume[0] = (3 * moment_of_volume[2]) / (2 * np.tan(delta)) * v

        elif type == 17:
            v = 3 * zl ** 2 * self.dx * np.tan(delta)
            origin_shift[0] = 0
            origin_shift[1] = 0
            origin_shift[2] = 0
            moment_of_volume[2] = 2 * zl / 3 * v
            moment_of_volume[1] = moment_of_volume[2] / np.tan(2 * psi)
            moment_of_volume[0] = self.dx / 2 * v
       
        elif type == 18:
            v = 6 * zl * (zr - zl) * self.dx * np.tan(delta)
            origin_shift[0] = 0
            origin_shift[1] = zl / np.tan(2 *psi)
            origin_shift[2] = zl
            moment_of_volume[2] = (zr - zl) / 2 * v
            if not special_case:
                moment_of_volume[1] = moment_of_volume[2] / np.tan(2 * psi)
            else:
                moment_of_volume[1] = -moment_of_volume[2]* np.tan(psi)
            moment_of_volume[0] = self.dx / 2 * v

        elif type == 19:
            v = 6 * zh * zl * self.dx * np.tan(delta)
            origin_shift[0] = 0
            origin_shift[1] = 0
            origin_shift[2] = 0
            moment_of_volume[2] = zt * 2 * v
            moment_of_volume[1] = yt / 2 * v
            moment_of_volume[0] = self.dx / 2 * v

        elif type == 20:
            v = -3 * (zt - zr) ** 2 * self.dx * np.tan(delta)
            origin_shift[0] = 0
            origin_shift[1] = -yt
            origin_shift[2] = -zt
            moment_of_volume[2] = -2 * (zt - zr) / 3 * v
            moment_of_volume[1] = -moment_of_volume[2] / np.tan(2 * psi)
            moment_of_volume[0] = -self.dx / 2 * v
        
        elif type == 21:
            v = (zq ** 2 * self.dx) / (2 * np.sin(psi) * np.cos(psi))
            origin_shift[0] = 0
            origin_shift[1] = 0
            origin_shift[2] = 0
            moment_of_volume[2] = 2 * zq / 3 * v
            moment_of_volume[1] = moment_of_volume[2] / np.tan(2 * psi)
            moment_of_volume[0] = self.dx / 2 * v

        elif type == 22:
            v = zq ** 2 * self.dy / 2 / np.tan(delta)
            origin_shift[0] = 0
            origin_shift[1] = 0
            origin_shift[2] = 0
            moment_of_volume[2] = zq / 3 * v
            moment_of_volume[1] = self.dy * v / 2
            moment_of_volume[0] = moment_of_volume[2] / np.tan(delta)

        elif type == 23:
            v = zr * self.dx * self.dy
            origin_shift[0] = 0
            origin_shift[1] = 0
            origin_shift[2] = 0
            moment_of_volume[2] = zr / 2 * v
            moment_of_volume[1] = self.dy * v / 2
            moment_of_volume[0] = self.dx / 2 * v

        elif type == 24:
            v = (zq - zr) * self.dx * self.dy / 2
            origin_shift[0] = 0
            origin_shift[1] = 0
            origin_shift[2] = zr
            moment_of_volume[2] = (zq - zr) / 3 * v
            moment_of_volume[1] = self.dy * v / 2
            moment_of_volume[0] = moment_of_volume[2] / np.tan(delta)

        elif type == 25:
            v = (zq - zt) * self.dy * self.dz / np.tan(delta)
            origin_shift[0] = 0
            origin_shift[1] = 0
            origin_shift[2] = 0
            moment_of_volume[2] = self.dz / 2 * v
            moment_of_volume[1] = self.dy * v / 2
            moment_of_volume[0] = (zq - zt) / 2 / np.tan(delta) * v

        elif type == 26:
            v = (zq - zt) * self.dy * self.dz / np.tan(delta)
            origin_shift[0] = 0
            origin_shift[1] = 0
            origin_shift[2] = 0
            moment_of_volume[2] = zt / 3 * v
            moment_of_volume[1] = self.dy * v / 2
            moment_of_volume[0] = moment_of_volume[2] / np.tan(delta)

        elif type == 27:
            v = self.dx * self.dy * self.dz
            origin_shift[0] = 0
            origin_shift[1] = 0
            origin_shift[2] = 0
            moment_of_volume[2] = self.dz * v / 2
            moment_of_volume[1] = self.dy * v / 2
            moment_of_volume[0] = self.dx * v / 2
        
        elif type == 28:
            v = -(zt - zr) ** 2 * self.dz / np.tan(delta) / 2
            origin_shift[0] = -self.dx
            origin_shift[1] = 0
            origin_shift[2] = -zt
            moment_of_volume[2] = -(zt - zr) / 3 * v
            moment_of_volume[1] = -self.dy * v / 2
            moment_of_volume[0] = -moment_of_volume[2] / np.tan(delta)

        elif type == 29:
            v = zq * self.dx * self.dy
            origin_shift[0] = 0
            origin_shift[1] = 0
            origin_shift[2] = 0
            moment_of_volume[2] = zq / 2 * v
            moment_of_volume[1] = self.dy * v / 2
            moment_of_volume[0] = self.dx / 2 * v
        
        return [special_case, v, origin_shift, moment_of_volume]
    
    def GetVolumebySumofPart(self, psi, delta, xp):
        partial_volume = list()
        volume_total = 0
        zq = xp * np.tan(delta) # z=(xp-x)tan(delta)
        zr = (xp - self.dx) * np.tan(delta) 
        zl = self.dy * np.sin(delta)
        zh = self.dz * np.cos(delta)
        zt = zl + zh

        if np.tan(delta) == 0:
            if np.tan(psi) == 0: #case 7
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 29))
            elif zr < np.minimum(zh, zl): #case 5a 21
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 21))
            elif zr > np.maximum(zh, zl): #case 5c 19 20
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 19))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 20))
            else: #case 5b 17 18
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 17))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 18))
        elif np.tan(psi) == 0:
            if zq < self.dz and zr < 0: # case 6a 22
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 22))
            elif zq < self.dz and zr > 0 : # case 6b 23 24
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 23))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 24))
            elif zq > self.dz and zr < 0 : # case 6c 25 26
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 25))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 26))
            elif zq > self.dz and zr < 0 : # case 6c 27 28
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 27))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 28))
        elif zq < self.dz and zr < 0: #case 1
            if zq < np.minimum(zh, zl): #case 1a #1
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 1))
            elif zq > np.maximum(zh, zl): #case 1c #2, 3, 6, 7, 11                
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 2))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 3))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 6))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 7))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 11))
            else: #case 1b # 2, 3, 5
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 2))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 3))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 5))
                mode = '1b'
        elif zq < self.dz and zr > 0: #case 2
            if zq < np.minimum(zh, zl) and zr < np.minimum(zh, zl): # case 2a 1, 4
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 1))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 4))
            elif zq < np.maximum(zh, zl) and zq > np.minimum(zh, zl) and zr < np.minimum(zh, zl): #case 2b 2 3 4 5
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 2))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 3))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 4))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 5))
            elif zq > np.maximum(zh, zl) and zr < np.minimum(zh, zl): #case 2c 2, 3, 4, 6, 7, 11
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 2))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 3))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 4))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 6))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 7))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 11))
            elif zq < np.maximum(zh, zl) and zq > np.minimum(zh, zl) and zr < np.maximum(zh, zl) and zr > np.minimum(zh, zl): # case 2d 8 17 18
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 8))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 17))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 18))
            elif zq < np.maximum(zh, zl) and zq > np.minimum(zh, zl) and zr > np.maximum(zh, zl): #case 2e 9 10 11 17 18
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 9))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 10))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 11))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 17))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 18))
            elif zq > np.maximum(zh, zl) and zr > np.maximum(zh, zl): #case 2f 12, 19, 20
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 12))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 19))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 20))
        elif zq > self.dz and zr > 0:
            if zr < np.minimum(zh, zl): #3a 2 3 4 6 7 13 14
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 2))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 3))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 4))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 6))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 7))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 13))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 14))
            elif zr > np.maximum(zh, zl): #3c 15 16 19 20
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 15))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 16))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 19))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 20))
            else: #3b 9 10 13 14 17 18
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 9))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 10))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 13))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 14))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 17))
                partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 18))
        else: #case 2 3 6 7 13 14
            partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 2))
            partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 3))
            partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 6))
            partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 7))
            partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 13))
            partial_volume.append(self.GetCrossSectionTank(psi, delta, zq, zr, zl, zh, zt, 14))

        return partial_volume
    
    def GetMoment(self, ship, trim, heel_trim):
        h_water = self.dz * self.rate

        trim_angle = np.arctan(trim * 2 / ship.l)
        htrim_angle = np.arctan(heel_trim * 2 / ship.b)

        volume = np.abs(self.dx * self.dy * self.dz * self.rate)
        
        if self.rate == 0:
            volume_p = [[0, 0, [0, 0, 0], [0, 0, 0]]]
        elif self.rate == 1:
            volume_p = [[0, volume, [self.dx / 2, self.dy / 2, self.dz / 2], [volume * self.dx / 2, volume * self.dy / 2, volume * self.dz / 2]]]
        else:
            xp_i = 0
            volume_p = self.GetVolumebySumofPart(xp_i, htrim_angle, trim_angle)
            volume_p_total = 0
            for i in volume_p:
                volume_p_total += i[1]

            while (volume_p_total - volume) > PERMISSIBLE_ERROR:
                dvolume_p = self.GetVolumebySumofPart(xp_i + VLD, htrim_angle, trim_angle)
                dvolume_p_total = 0
                for i in dvolume_p:
                    dvolume_p_total += i[1]

                xp_i = xp_i - volume_p_total / dvolume_p_total
                volume_p = self.GetVolumebySumofPart(xp_i, htrim_angle, trim_angle)
                volume_p_total = 0
                for i in volume_p:
                    volume_p_total += i[1]

        
        cog = np.zeros(3)
        if self.rate:
            cog = np.zeros(3)
            cog[0] = self.x + self.dx / 2
            cog[1] = self.y + self.dy / 2
            cog[2] = self.z + self.dz / 2
            for i in volume_p:
                origin_shift = i[2] + i[3] / volume
                if trim_angle > 0:
                    cog[0] += origin_shift[0]
                else:
                    cog[0] -= origin_shift[0]

                cog[1] += np.cos(htrim_angle) * origin_shift[1] + np.sin(htrim_angle) * origin_shift[2]
                cog[2] += -np.sin(htrim_angle) * origin_shift[2] + np.cos(htrim_angle) * origin_shift[2]

        return volume * self.density * cog

                


    

# |            /
# dz         dy
# |         /
# (x, y, z) - dx -

# FP = 0, Midship = 0, Bottom = 0

class Load:
    def __init__(self, mass, x, y, z, dx = 0, dy = 0, dz = 0):
        self.mass = mass
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz

class Ship:
    def __init__(self, l, b, h, w):
        self.l = l
        self.b = b
        self.h = h
        self.w = w

class Scenario:
    def __init__(self, Ship, comp_id, load_id):
        self.Ship = Ship
        self.compartments = list()
        self.loads = list()
        self.comp_id = comp_id 
        self.next = list()
        self.load_id = load_id

    def add_load(self, load):
        self.loads.append(load)

    def GetMomentbyHeeling(self, trim, heel_trim):
        required_buyoancy = self.Ship.w
        for l in self.loads:
            required_buyoancy += l.mass
        for c in self.compartments:
            required_buyoancy += c.dx * c.dy * c.dz * c.density * c.rate

        required_draft = required_buyoancy / self.Ship.l / self.Ship.b
        x = -RHO * (trim * self.Ship.l * self.Ship.l * self.Ship.b / 6 + self.Ship.l * self.Ship.l * self.Ship.b * required_draft / 2)
        y = -RHO * heel_trim * self.Ship.l * self.Ship.b * self.Ship.b / 6

        return np.array([x, y, 0])
        

    def GetEquilibrium(self):
        required_buyoancy = self.Ship.w
        moment = [self.Ship.l / 2 * self.Ship.w, 0]
        for l in self.loads:
            required_buyoancy += l.mass
            moment[0] += ((l.x + l.dx / 2) * l.mass)
            moment[1] += (l.y + l.dy / 2) * l.mass

        required_draft = required_buyoancy / (RHO * self.Ship.l * self.Ship.b)

        #error = self.MomentbyHeeling(required_draft, trim, heel_trim) - moment

        self.trim = (moment[0] / RHO - self.Ship.l * self.Ship.l * self.Ship.b * required_draft / 2) * 6 / (self.Ship.l * self.Ship.l * self.Ship.b)
        self.heel_trim = 6 * moment[1] / RHO / self.Ship.l / self.Ship.b / self.Ship.b

    def GetTotalMoment(self, trim, heel_trim):
        required_buyoancy = self.Ship.w
        moment = np.array([self.Ship.l / 2 * self.Ship.w, 0, self.Ship.h / 2 * self.Ship.w])
        for l in self.loads:
            required_buyoancy += l.mass
            moment[0] += (l.x + l.dx / 2) * l.mass
            moment[1] += (l.y + l.dy / 2) * l.mass
            moment[2] += (l.z + l.dz / 2) * l.mass

        for c in self.compartments:
            required_buyoancy = c.dx * c.dy * c.dz * c.rate * c.density
            moments_comp = c.GetMoment(self.Ship, trim, heel_trim)
            moment[0] += moments_comp[0]
            moment[1] += moments_comp[1]
            moment[2] += moments_comp[2]

        moment += self.GetMomentbyHeeling(trim, heel_trim)

        return moment

    def GetEquilibriumNewtonMethod(self):
        trim = 0
        heel_trim = 0

        moment = self.GetTotalMoment(trim, heel_trim)
        if self.load_id == 0 and self.comp_id == 11101110:
            print("k")
        time_in_loof = 0

        while(np.sqrt(moment[0]*moment[0] + moment[1]*moment[1]) > PERMISSIBLE_ERROR):
            j = np.zeros((2, 2), dtype=float)
            j[0, 0] = (self.GetTotalMoment(trim + VLD, heel_trim)[0] - moment[0])/ VLD
            j[0, 1] = (self.GetTotalMoment(trim, heel_trim + VLD)[0] - moment[0])/ VLD # I know its value is zero
            j[1, 0] = (self.GetTotalMoment(trim + VLD, heel_trim)[1] - moment[1])/ VLD  # I know its value is zero
            j[1, 1] = (self.GetTotalMoment(trim, heel_trim + VLD)[1] - moment[1])/ VLD

            np.nan_to_num(j, copy=False)
            tht = np.array([trim, heel_trim]).T
            jtj_inv = np.linalg.inv(j.T.dot(j))
            tht = tht - jtj_inv.dot(j.T).dot(moment[0:2])
            trim = tht[0]
            heel_trim = tht[1]

            moment = self.GetTotalMoment(trim, heel_trim)
            np.nan_to_num(moment, copy=False)
            time_in_loof += 1

            if time_in_loof > 1000:
                print("k")
                break
            #print(moment)

        if np.sqrt(moment[0]*moment[0] + moment[1]*moment[1]) <= PERMISSIBLE_ERROR:
            self.trim = trim
            self.heel_trim = heel_trim
            self.kgtimesweight = moment[2]
        elif time_in_loof > 1000000:
            self.trim = 100000
            self.heel_trim = 100000
            self.kgtimesweight = 0

    def GetValdityofScenario(self, df, df_i):
        required_buyoancy = self.Ship.w
        for l in self.loads:
            required_buyoancy += l.mass
        for c in self.compartments:
            required_buyoancy += c.dx * c.dy * c.dz * c.rate * c.density
        draft = required_buyoancy / self.Ship.l / self.Ship.b

        self.GetEquilibriumNewtonMethod()

        kg = self.kgtimesweight / required_buyoancy
        kb = ((draft - self.heel_trim)**2 / 2 + (draft - 2 * self.heel_trim / 3) * self.heel_trim / 3) / (draft)
        bm = self.Ship.b ** 2 / 12 / draft
        ggo = 0
        for i in self.compartments:
            ggo += np.abs(c.density / RHO * c.dx * c.dy ** 3 / 12 / self.Ship.l / self.Ship.b / draft)

        gm = kb + bm - kg - ggo


        r = 1
        lispace = np.linspace(0,r,self.Ship.l)
        dim = len(lispace)
        load = np.zeros(dim)
        buoyancy = np.zeros(dim)
        shear = np.zeros(dim)
        moment = np.zeros(dim)
        c1 = 0.03 * self.Ship.l
        c2 = np.zeros(dim)

        for l in self.loads:
            if l.dx > 0:
                load[l.x:np.floor(l.dx/dim).astype(int)] += l.mass / l.dx
            elif l.dx < 0:
                load[l.x-l.dx:np.floor(l.dx/dim).astype(int)] += l.mass / l.dx
            else:
                load[l.x] += l.mass
        for l in self.compartments:
            if l.dx > 0:
                load[l.x:np.floor(l.dx/dim).astype(int)] += l.dy * l.dz * l.density
            elif l.dx < 0:
                load[l.x-l.dx:np.floor(l.dx/dim).astype(int)] += l.dy * l.dz * l.density
# B(T-trim)    B(T+trim)           
# 0            L

        for i in range(dim):
            buoyancy[i] = lispace[i] * self.Ship.b * self.trim / self.Ship.l + self.Ship.b * (draft - self.trim)
            if lispace[i] <= 0.4 * self.Ship.l:
                c2[i] = 1 / (0.4 * self.Ship.l) * lispace[i]
            elif lispace[i] >= 0.65 * self.Ship.l:
                c2[i] = 1 - 1 / (0.35 * self.Ship.l) * (lispace[i] - 0.65 * self.Ship.l)
            else:
                c2[i] = 1
        
        load -= buoyancy
        shear[0] = 0
        for i in range(dim - 1):
            shear[i + 1] = shear[i] + load[i] * r

        if sum(shear):
            shear = shear - sum(shear) / self.Ship.l
        moment[0] = 0
        for i in range(dim - 1):
            moment[i + 1] = moment[i] + shear[i] * r 
        moment[-1] = 0

        msw_sagging = 0.19 * c1 * c2 * self.Ship.l ** 2 * self.Ship.b
        msw_hogging = -0.11 * c1 * c2 * self.Ship.l ** 2 * self.Ship.b * 1.7

        z1_sagging = np.max(np.abs(moment + msw_sagging) / 175) * 10**3
        z1_hogging = np.max(np.abs(moment + msw_hogging) / 175) * 10**3
        z_min = c1 * self.Ship.l ** 2 * (self.Ship.b + 0.7)

        if draft > self.Ship.h:
            self.validity = True
            
        elif np.abs(self.trim) + draft > self.Ship.h or np.abs(self.heel_trim) + draft > self.Ship.h:
            self.validity = True
            
        elif (gm < 0.164 * self.Ship.b and len(self.loads) == 0) or gm < 0.095 * self.Ship.b:
            self.validity = True
            
        elif z1_sagging < z_min and z1_hogging <z_min:
            self.validity = True
        else:
            self.validity = False

        df.loc[df_i] = [self.comp_id, self.load_id, draft, self.trim, self.heel_trim, gm, z1_sagging, z1_hogging, self.validity]
        return self.validity  
                
    def SetCompartmentsRate(self, comps):
        self.compartments = comps[:]

        for i in range(8):
            i_comp_rate = self.comp_id % 10 ** (i + 1) // 10 ** i
            self.compartments[i].rate = i_comp_rate / MAX_BALLAST_CASE

    def SetLoadingState(self, loads):
        self.loads = loads[0:(15 - self.load_id)]

    def ConnectNextScenario(self, item):
        self.next.append(item)

class Scenario_BIG:
    def __init__(self, ship, df):
        self.ship = ship
        self.scenarioes = dict()
        self.loads = list()
        self.standard_comp = list()
        self.df = df
        self.A = 18
        self.B = 12
        self.C = 18
        self.D = 18

        #Bottom-Tower-1
        self.loads.append(Load(80, 33, -2.25, H, 4.5, 4.5, 18.0))
        #Mid-Tower-1
        self.loads.append(Load(67, 26, -2.2, H, 4.5, 4.4, 32.0))
        #Top-Tower-1
        self.loads.append(Load(40, 20, -1.9, H, 3.8, 3.8, 25.0))
        #Nacelle-1
        self.loads.append(Load(30, 3, 1, H, 13.3, 4.8, 5.6))
        #Blade1-1
        self.loads.append(Load(14, 3, 6.5, H + 8.4, 65.0, 4.2, 4.2))
        #Blade2-1
        self.loads.append(Load(14, 3, 6.5, H + 4.2, 65.0, 4.2, 4.2))
        #Blade3-1
        self.loads.append(Load(14, 3, 6.5, H, 65.0, 4.2, 4.2))
        #Hub-1
        self.loads.append(Load(30, 60, 1, H, 5.1, 5.1, 4.3))

        #Bottom-Tower-2
        self.loads.append(Load(80, 53, -2.25, H, 4.5, 4.5, 18.0))
        #Mid-Tower-2
        self.loads.append(Load(67, 46, -2.2, H, 4.5, 4.4, 32.0))
        #Top-Tower-2
        self.loads.append(Load(40, 40, -1.9, H, 3.8, 3.8, 25.0))
        #Nacelle-2
        self.loads.append(Load(30, 3, -1, H, 13.3, -4.8, 5.6))
        #Blade1-2
        self.loads.append(Load(14, 3, -6.5, H + 8.4, 65.0, -4.2, 4.2))
        #Blade2-2
        self.loads.append(Load(14, 3, -6.5, H + 4.2, 65.0, -4.2, 4.2))
        #Blade3-2
        self.loads.append(Load(14, 3, -6.5, H, 65.0, -4.2, 4.2))
        #Hub-2
        self.loads.append(Load(30, 66, -1, H, 5.1, -5.1, 4.3))

        wbt_1_p = Compartment(0, 6.5, 0, self.A, 6.5, H, RHO)
        wbt_1_cp = Compartment(0, 0, 0, self.B, 6.5, H, RHO)
        wbt_1_s = Compartment(0, -13, 0, self.A, 6.5, H, RHO)
        wbt_1_cs = Compartment(0, -6.5, 0, self.B, 6.5, H, RHO)
        wbt_2_p = Compartment(self.ship.l - self.C, 6.5, 0, self.C, 6.5, H, RHO)
        wbt_2_cp = Compartment(self.ship.l - self.D, 0, 0, self.D, 6.5, H, RHO)
        wbt_2_s = Compartment(self.ship.l - self.C, -13, 0, self.C, 6.5, H, RHO)
        wbt_2_cs = Compartment(self.ship.l - self.D, -6.5, 0, self.D, 6.5, H, RHO)
        self.standard_comp.append(wbt_1_p)
        self.standard_comp.append(wbt_1_cp)
        self.standard_comp.append(wbt_1_s)
        self.standard_comp.append(wbt_1_cs)
        self.standard_comp.append(wbt_2_p)
        self.standard_comp.append(wbt_2_cp)
        self.standard_comp.append(wbt_2_s)
        self.standard_comp.append(wbt_2_cs)

        comp_id_init = 0
        ls_init = 0
        queue = deque()
        queue.append((comp_id_init, ls_init))
        timesss = 0

        while len(queue):
            v, ls = queue.popleft()
            self.scenarioes[(v, ls)] = Scenario(self.ship, v, ls)
            for i in range(8):
                if (v / (10 ** i)) % 10 ** (i + 1) > 0:
                    pv = nv - 10 ** i
                    self.scenarioes[(v, ls)].ConnectNextScenario((v, ls))

            print(v, ls, timesss)
            timesss += 1
            for i in range(8):
                if (v / (10 ** i)) % 10 ** (i + 1) < MAX_BALLAST_CASE:
                    nv = v + 10 ** i
                    if not (nv, ls) in self.scenarioes and not (nv, ls) in queue:
                        queue.append((nv, ls))
                    self.scenarioes[(v, ls)].ConnectNextScenario((nv, ls))
            if ls < 15:
                if not (v, ls + 1) in self.scenarioes and not (v, ls + 1) in queue:
                    queue.append((v, ls + 1))
                    self.scenarioes[(v, ls)].ConnectNextScenario((v, ls + 1))

        df_i = 0
        for i in self.scenarioes.values():
            print(i.comp_id, i.load_id, sep="\t")
            i.SetCompartmentsRate(self.standard_comp)
            i.SetLoadingState(self.loads)
            i.loads = self.loads[0:(15-i.load_id)]
            val = i.GetValdityofScenario(df, df_i)
            df_i += 1

    def GetShortestPath(self):
        startNode = (0, 0)
        queue = deque()
        dist = dict()
        nearest_place = dict()
        dist[startNode] = 0
        nearest_place[startNode] = startNode
        astarq = pq()
        astarq.put((0, startNode))

        while astarq:
            q_dist, node = astarq.get()
            cid, lid = node

            if lid == 15:
                answer = (cid, lid)
                break

            if self.scenarioes[(cid, lid)].validity:
                if (cid, lid) in dist.keys():
                    dist[(cid, lid)] = np.minimum(q_dist, dist[(cid, lid)])
                else:
                    dist[(cid, lid)] = q_dist

            for i in range(8):
                if (cid / (10 ** i)) % 10 ** (i + 1) < MAX_BALLAST_CASE:
                    ncid = cid + 10 ** i
                    f = dist[(cid, lid)] + self.Heuristic((cid, lid), (ncid, lid))
                    if self.scenarioes[(cid, lid)].validity:
                        if (ncid, lid) in dist.keys():
                            dist[(ncid, lid)] = np.minimum(f, dist[(ncid, lid)])
                        else:
                            dist[(ncid, lid)] = f
                        astarq.put((f, (ncid, lid)))
                if (cid / (10 ** i)) % 10 ** (i + 1) > 0:
                    ncid = cid - 10 ** i
                    f = dist[(cid, lid)] + self.Heuristic((cid, lid), (ncid, lid))
                    if self.scenarioes[(cid, lid)].validity:
                        if (ncid, lid) in dist.keys():
                            dist[(ncid, lid)] = np.minimum(f, dist[(ncid, lid)])
                        else:
                            dist[(ncid, lid)] = f
            
            f = dist[(cid, lid)] + self.Heuristic((cid, lid), (cid, lid + 1))
            if self.scenarioes[(cid, lid + 1)].validity:
                if (cid, lid + 1) in dist.keys():
                    dist[(cid, lid + 1)] = np.minimum(f, dist[(cid, lid + 1)])
                else:
                    dist[(cid, lid + 1)] = f
                nlid = lid + 1
                astarq.put((f, (cid, nlid)))

        self.final_destination = answer
        self.shortest_val = q_dist
        print(self.final_destination, self.shortest_val)
                



    def Heuristic(self, prev, next):
        prev_c = self.scenarioes[prev]
        next_c = self.scenarioes[next]

        if prev_c.load_id < next_c.load_id and prev_c.comp_id == next_c.comp_id:
            return 0
        else:
            prev_c_v = 0
            next_c_v = 0
            for i in range(8):
                prev_c_v += prev_c.compartments[i].dx * prev_c.compartments[i].dy * prev_c.compartments[i].dz * prev_c.compartments[i].rate
                next_c_v += next_c.compartments[i].dx * next_c.compartments[i].dy * next_c.compartments[i].dz * next_c.compartments[i].rate

            return np.abs(prev_c_v - next_c_v)


s = Ship(L, B, H, W)
df = pd.DataFrame(columns=["Comp_ID", "Load_ID", "Draft", "Trim", "Heeling", "GM", "Z1_SAG", "Z1_HOG", "Validity"])
scenarioes_1 = Scenario_BIG(s, df)
scenarioes_1.GetShortestPath()
df.to_csv("result.csv",  sep=',', na_rep='NaN')