import numpy as np
import numpy.fft as fft
from scipy import interpolate

from tqdm import tqdm
import random

from numpy import genfromtxt
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

import glob

# Source parameters
SOURCE_DECL       = 30
SOURCE_D          = 337   * 3.086*1e16 # Distance = 337pc
SOURCE_R          = 0.169 * 6.957*1e8  # Radius = 0.179 solar radius
ELLIPTICITY       = 0.07               # Ellipticity (cf. tidal guys)
LAMBDA            = 3.5e-7             # radiation wavelength
FLUX              = 2.85e-9

# Control the masks used for plotting and splines:
BLINE_MASK_WIDTH  = 2              # In LAMBDA/SOURCE_D units
DELTA_MASK_WIDTH  = 4              # In SOURCE_D units
RAD_TO_MAS        = 1e3 * (3600 * 180) / np.pi

# Default resolution
SOURCE_RESOLUTION = 32       # Resolution of the source
IMAGE_SIZE        = 32       # In source resolution units... (that's a bit weird, I know)

# Array Properties
T_RES             = 0.5e-9
NCHANNEL          = 1024
ALPHA             = 0.4
LATTITUDE         = 24
TELESCOPE_POS     = ['23m_telescope.csv', '04m_telescope.csv', '12m_telescope.csv','08m_telescope.csv', '39m_telescope.csv'] 
TELESCOPE_AREAS   = [np.pi*(23./2)**2, np.pi*2**2, np.pi*6**2, np.pi*(8./2)**2, np.pi*(8./2)**2]

class LightSource:
    def __init__(self, shape='tdb', distance = SOURCE_D,
                 radius = SOURCE_R, ellipticity = ELLIPTICITY):
        self.d = distance
        self.r = radius
        self.e = ellipticity
        self.delta_rad = radius / distance

        self.delta = self.delta_rad * RAD_TO_MAS
        self.sdBfile = 'theta_vs_delta.dat'
        self.shape = shape

    def getEllipticDistribution(self, x, y):
        a = self.delta * (1 + self.e)
        b = self.delta * (1 - self.e)
        ret = 1.0 * ( (x/a)**2 + (y/b)**2 <= 1)
        return ret

    def getTidallyStretchedDistribution(self, x, y):
        angle, f = np.loadtxt(self.sdBfile, delimiter=',', unpack=True)
        idx = np.argsort(angle)
        angle = angle[idx]
        f = f[idx]

        phi = np.arctan2(y,x) - np.pi/2
        f2 = np.interp(phi, angle, f**2, period=2*np.pi)

        return x**2 + y**2 <= f2

    def getDistribution(self, x, y):
        if self.shape == 'tdb':
            return self.getTidallyStretchedDistribution(x,y)
        elif self.shape == 'ellipse':
            return self.getEllipticDistribution(x,y)
        else:
            print("Shape {} not implemented. must be either 'tdb' or 'ellipse'".format(self.shape))
            return None


class UVplane:
    def __init__(self, LightSource, source_resolution = SOURCE_RESOLUTION,
                 image_size = IMAGE_SIZE, bline_mask_w = BLINE_MASK_WIDTH,
                 delta_mask_w = DELTA_MASK_WIDTH):
        self.source = LightSource

        self.N_s = source_resolution
        self.n   = image_size
        self.N   = self.N_s * self.n

        self.delta_min = self.source.delta / self.N_s
        self.delta_max = self.source.delta * self.n

        self.bline_min = LAMBDA / self.source.delta_rad / self.n
        self.bline_max = LAMBDA / self.source.delta_rad * self.N_s

        self.bline_mask_w = bline_mask_w * LAMBDA / self.source.delta_rad
        self.delta_mask_w = delta_mask_w * self.source.delta

        ext = np.array([-.5, .5, -.5, .5])
        self.ext_physical = self.delta_max * ext
        self.ext_physical_masked = self.delta_mask_w * ext
        self.ext_spectral = self.bline_max * ext
        self.ext_spectral_masked = self.bline_mask_w * ext

    def idx_to_delta(self, i, bMask=False):
        if (bMask):
            n = self.N_s
        else:
            n = self.N
        ret = (i-n//2) * self.delta_min
        return ret

    def idx_to_bline(self, i, bMask=False):
        if (bMask):
            n = self.n*BLINE_MASK_WIDTH
        else:
            n = self.N

        ret = (i-n//2) * self.bline_min
        return ret

    def getAngularDistribution(self, bMask):
        # Real space mesh
        i = np.arange(self.N)
        j = np.arange(self.N)
        d_x = self.idx_to_delta(i, bMask=False)
        d_y = self.idx_to_delta(j, bMask=False)
        delta_x, delta_y = np.meshgrid(d_x, d_y)
        rho  = self.source.getDistribution(delta_x, delta_y)
        if not bMask:
            return rho

        else:
            mask_x = (d_x < 0.5*self.delta_mask_w) & (d_x > -0.5*self.delta_mask_w)
            mask_y = (d_y < 0.5*self.delta_mask_w) & (d_y > -0.5*self.delta_mask_w)
            rho = rho[mask_x, :]
            rho = rho[:, mask_y]
            return rho

    def getCorrelationFunction(self, bMask):
        # Real space mesh
        i = np.arange(self.N)
        j = np.arange(self.N)
        delta_x = self.idx_to_delta(i)
        delta_y = self.idx_to_delta(j)
        delta_x, delta_y = np.meshgrid(delta_x, delta_y)
        rho  = self.source.getDistribution(delta_x, delta_y)
        g_12 = fft.fft2(rho)
        g_12 = fft.fftshift(g_12)      # Shift the (0,0) wavelength to the center
        g_12 = g_12 / np.max(g_12)     # Normalize with max value
        ret  = np.absolute(g_12)**2
        if not bMask:
            return ret

        else:
            b_x = self.idx_to_bline(i, bMask=False)
            b_y = self.idx_to_bline(j, bMask=False)
            mask_x = (b_x <= 0.5*self.bline_mask_w) & (b_x >= -0.5*self.bline_mask_w)
            mask_y = (b_y <= 0.5*self.bline_mask_w) & (b_y >= -0.5*self.bline_mask_w)
            ret = ret[mask_x, :]
            ret = ret[:, mask_y]
            return ret

    def plotSource(self):
        fig, ax = plt.subplots(1,2,figsize=(36, 24))
        fig.subplots_adjust(wspace = 0.32)
        circ = plt.Circle( xy = (0,0), radius = self.source.delta*1000,
                           fill = False, lw = 4, ls = '--', color = 'Red')

        ext_p = self.ext_physical_masked * 1000
        ext_s = self.ext_spectral_masked
        delta = self.source.delta
        rho  = self.getAngularDistribution(bMask=True)
        im0 = ax[0].imshow(rho, origin='lower', extent=ext_p, cmap = 'afmhot')
        ax[0].add_artist(circ)

        corr = self.getCorrelationFunction(bMask=True)
        im1 = ax[1].imshow(corr, origin='lower', extent=ext_s, cmap = 'afmhot_r', vmin=0.0, vmax=1.5)

        i, j = np.meshgrid(np.arange(corr.shape[0]), np.arange(corr.shape[1]))

        b_x = self.idx_to_bline(i, bMask=True)
        b_y = self.idx_to_bline(j, bMask=True)
        
        con = ax[1].contour(b_x, b_y, corr, levels=[0.01, 0.1,0.2,0.3,.4,.5,.6,.7,.8,.9,.975], colors='black')

        ax[0].tick_params(labelsize=26)
        scale_h = 1.4
        ax[0].set_xlim(-scale_h * delta /1.2 * 1000, scale_h * delta/1.2 * 1000)
        ax[0].set_xlabel(r'$x$ [$\mu$as]', fontsize = 40)
        ax[0].set_ylim(-scale_h * delta/1.2 * 1000, scale_h * delta/1.2 * 1000)
        ax[0].set_ylabel(r'$y$ [$\mu$as]', fontsize = 40)

        # Settings Spectral image
        circ2500 = plt.Circle(xy = (0,0), radius = 23000,
                           fill = False, hatch = '__',lw = 4, ls = '--', color = 'k')
        
        ax[1].add_artist(circ2500)
        ax[1].tick_params(labelsize=26)
        ax[1].set_xlim(ext_s[0]/1.2, ext_s[1]/1.2)
        ax[1].set_ylim(ext_s[2]/1.2, ext_s[3]/1.2)
        ax[1].set_xlabel(r'$\lambda \cdot u$ [m]', fontsize = 40)
        ax[1].set_ylabel(r'$\lambda \cdot v$ [m]', fontsize = 40)
        ax[1].clabel(con, inline=True,inline_spacing = 15, fontsize=25)
        plt.tight_layout(pad = 10)
        
        plt.show()
        fig.savefig('poster.pdf', bbox_inches = 'tight')

    def getCorrelationSpline(self, bMask=True, kind='cubic'):
        corr = self.getCorrelationFunction(bMask)
        i, j = np.meshgrid(np.arange(corr.shape[0]), np.arange(corr.shape[1]))

        b_x = self.idx_to_bline(i, bMask)
        b_y = self.idx_to_bline(j, bMask)

        ret = interpolate.interp2d(b_x, b_y, corr, kind=kind)

        return ret

def Rx(a):
    ret = [ [     1    ,     0    ,      0    ],
            [     0    , np.cos(a), -np.sin(a)],
            [     0    , np.sin(a),  np.cos(a)] ]
    return ret

def Ry(a):
    ret = [ [ np.cos(a),     0    ,  np.sin(a)],
            [     0    ,     1    ,      0    ],
            [-np.sin(a),     0    ,  np.cos(a)] ]
    return ret

def R(h,d,l):
    ret = np.dot(Rx(d), Ry(h))
    ret = np.dot(  ret, Rx(-l))
    return ret

class TelescopeArray:
    def __init__(self, UVplane, layout='layouts/basic_cta', nGrid=128, nPerTraj=128):
        self.latt = LATTITUDE
        self.decl = SOURCE_DECL

        self.bline_max  = 2500
        self.N_snr_grid = nGrid
        self.N_per_traj = nPerTraj
        self.T_per_traj = 12

        self.files = []
        self.areas = []

        self.UVplane = UVplane
        self.spline = self.UVplane.getCorrelationSpline(bMask=True)
        
        self.readLayout(layout)

        self.bGrids = False
        
    def read_diameter(self, file):
        return np.float(file.split('/')[-1].split('_')[0][:-1])
        
    def readLayout(self, layout):
        self.layout=layout
        files = glob.glob(layout+'/*telescope.csv')
        n = len(files)
        areas = np.zeros(n)
        self.telescope_files = files
        for i, f in enumerate(files):
            diam = self.read_diameter(f)
            areas[i] = np.pi*(diam / 2.0)**2
        self.telescope_areas = areas
        
        self.files = []
        self.areas = []
        for k in range(n):
            for l in range(k+1):
                self.files.append([self.telescope_files[k], self.telescope_files[l]])
                self.areas.append(np.sqrt(self.telescope_areas[k] * self.telescope_areas[l]))

    def getTrajectory(self, blines):
        if len(blines) == 2 :
            blines = np.append(blines,0)

        uv = np.zeros((2,self.N_per_traj))
        hour_angle  = np.linspace(0, 24./self.T_per_traj * 2*np.pi, self.N_per_traj)
        declination = np.full((self.N_per_traj), self.decl*np.pi/180)
        lattitude   = np.full((self.N_per_traj), self.latt*np.pi/180)

        for i in range(self.N_per_traj):
            out = np.dot(R(hour_angle[i],declination[i],lattitude[i]), blines)
            uv[0,i] = out[0]
            uv[1,i] = out[1]
        return uv

    def getGridIndex(self, u, v):
        du = 2*self.bline_max / self.N_snr_grid

        idx = (u + self.bline_max) / du - 0.5
        idy = (v + self.bline_max) / du - 0.5

        return(int(idx), int(idy))

    def getPosition(self, idx, idy):
        du = 2*self.bline_max / self.N_snr_grid

        u = (idx+0.5) * du - self.bline_max
        v = (idy+0.5) * du - self.bline_max

        return(u, v)

    def integrateGrids(self, traj, area, areaGrid, nInGrid):
        for i in range(self.N_per_traj):
            u = traj[0,i]
            v = traj[1,i]
            uu, vv = self.getGridIndex(u, v)
            if (uu < self.N_snr_grid and vv < self.N_snr_grid):
                areaGrid[uu, vv] += area / self.N_per_traj
                nInGrid [uu, vv] += 1.0  / self.N_per_traj
                
    def getPairs(self, l1, l2):
        if np.prod(l1 != l2):
            return [[a,b] for a in l1 for b in l2 if np.prod(a != b)]
        else:
            return [[l1[i],l2[j]] for i in range(len(l1)) for j in range(i, len(l2)) if np.prod(l1[i] != l2[j])]

    def setGrids(self):
        print('Computing SNR grid from layout:', self.layout, flush=True)
        areaGrid = np.zeros((self.N_snr_grid, self.N_snr_grid))
        nInGrid  = np.zeros((self.N_snr_grid, self.N_snr_grid))

        meanArea = 0
        nPairs = 0
        for i, area in enumerate(self.areas):
            posA = np.array(genfromtxt(self.files[i][0], delimiter=','))
            posB = np.array(genfromtxt(self.files[i][1], delimiter=','))
            #print(posA.shape, posB.shape, )
            if posA.ndim == 1:
                posA = np.expand_dims(posA, axis=0)
            if posB.ndim == 1:
                posB = np.expand_dims(posB, axis=0)
            pos = np.concatenate((posA, posB))
            #pairs = np.array(list(combinations(pos,2)))
            pairs = self.getPairs(posA, posB)
            #print(len(comb))
            #pairs = set(comb)
            #print(len(pairs))
            nPairs += 2*len(pairs)
            meanArea += area * 2 * len(pairs)

            pbar = tqdm(pairs)
            diam1 = self.read_diameter(self.files[i][0])
            diam2 = self.read_diameter(self.files[i][1])
            pbar.set_description('Pairs ({}-{})'.format(diam1, diam2))
            for p in pbar:
                B = (p[0] - p[1])
                uv = self.getTrajectory(B)
                self.integrateGrids(uv, area, areaGrid, nInGrid)
                uv = self.getTrajectory(-B)
                self.integrateGrids(uv, area, areaGrid, nInGrid)

        self.nPairs = nPairs
        self.meanArea = meanArea / nPairs

        fac_snr  = FLUX * ALPHA * np.sqrt(NCHANNEL / T_RES / 2)
        areaGrid = np.where(nInGrid == 0, 0, areaGrid / nInGrid)
        timeGrid = nInGrid * self.T_per_traj
        snrGrid  = areaGrid * fac_snr * np.sqrt(timeGrid*3600)

        for i in range(self.N_snr_grid):
            for j in range(self.N_snr_grid):
                u, v = self.getPosition(i,j)
                snrGrid[i,j] *= self.spline(u,v)

        self.timeGrid = timeGrid
        self.snrGrid = snrGrid
        self.areaGrid = areaGrid
        self.bGrids = True

    def getSNRGrid(self):
        if not self.bGrids:
            self.setGrids()
        return self.snrGrid

    def getTimeGrid(self):
        if not self.bGrids:
            self.setGrids()
        return self.timeGrid

    def getAreaGrid(self):
        if not self.bGrids:
            self.setGrids()
        return self.areaGrid

    def getSignal(self, nNights, phi_source, bNoise):
        time = self.getTimeGrid() * nNights
        snr  = self.getSNRGrid() * np.sqrt(nNights)
        area = self.areaGrid
        signal = np.zeros((self.N_snr_grid, self.N_snr_grid))
        std = np.zeros((self.N_snr_grid, self.N_snr_grid))

        fac_signal = nNights * (ALPHA * FLUX)**2 * NCHANNEL# / T_RES

        
        for i in range(self.N_snr_grid):
            for j in range(self.N_snr_grid):
                u0, v0 = self.getPosition(i, j)
                u = u0 * np.cos(phi_source) - v0 * np.sin(phi_source)
                v = u0 * np.sin(phi_source) + v0 * np.cos(phi_source)
                mean = self.timeGrid[i,j] * self.areaGrid[i,j]**2 * fac_signal * self.spline(u, v)
                if (bNoise):
                    if (snr[i,j]>0):
                        sigma = np.fabs(mean / (snr[i,j] * np.sqrt(nNights)))
                        val = np.random.normal(mean, sigma)
                        signal[i,j] = val
                        std[i,j] = sigma
                else :
                    signal[i,j] = mean
        return signal, std
