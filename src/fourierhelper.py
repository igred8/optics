
import numpy as np
from scipy import fft
from scipy.constants import pi
import cv2
import matplotlib.pyplot as plt

import diffractsim as ds

mm_ = 1e-3
um_ = 1e-6
nm_ = 1e-9

def pad_next2( arr, augmentfactor=1 ):
    '''
    pad array to the next power of 2 for faster FFT
    '''
    nsize = 2**(int(np.log2(augmentfactor*np.max(arr.shape))) + 1)
    npad_diff = nsize - np.array(arr.shape)
    npad = npad_diff//2
    
    arr = np.pad(arr, [[npad[0], npad[0] + npad_diff[0]%2], [npad[1], npad[1] + npad_diff[1]%2]])

    return arr

def rect2d(xx, yy, x0, y0, xwidth, ywidth, zeroval=0.0):
    """
    Rectangular slit.
    Center at (x0, y0) with xwidth,ywidth dimensions

    zeroval - 0.0, value of function at discontinuity

    """
    # zz = np.heaviside( -(xx - x0 - xwidth/2), zeroval)
    # zz = zz * np.heaviside( xx - x0 + xwidth/2, zeroval)
    # zz = zz * np.heaviside( -(yy - y0 - ywidth/2), zeroval)
    # zz = zz * np.heaviside( yy - y0 + ywidth/2, zeroval)

    # 10x faster than np.heaviside and more intuitive
    zz = (xx <= x0 + xwidth/2) & (xx >= x0 - xwidth/2)
    zz = zz & (yy <= y0 + ywidth/2) & (yy >= y0 - ywidth/2)

    return np.array(zz, dtype=np.float64)

def deltafunc_2d( xx, yy, x0, y0 ):
    """
    Returns the Dirac delta function for the discrete mesh defined by xx,yy arrays.
    
    x0,y0 is the center of the delta function

    x0,y0 are rounded to the nearest mesh point 

    """
    delta_x = xx[0,1] - xx[0,0]
    delta_y = yy[1,0] - yy[0,0]
    
    # round centers to nearest mesh point
    x0rounded = (x0 // delta_x) 
    y0rounded = (y0 // delta_y)

    zz = ( xx//delta_x == x0rounded ) & ( yy//delta_y == y0rounded )

    return np.array(zz, dtype=np.float64)

def zrayleigh( wavelen, waist0, refractive_index=1.0):
    zR = pi * waist0**2 * refractive_index / wavelen
    return zR
def radius_curve( dist, zR ):
    if (dist < 0.0) or (dist > 0.0):
        Rcurve = dist * ( 1 + (zR/dist)**2)
    else:
        Rcurve = 1e34 # needs to be infinite at waist, but avoid div by 0
    return Rcurve

def waist_gaussian( waist0, dist, zR ):
    w = waist0 * np.sqrt( 1 + (dist/zR)**2 )
    return w

def efield_guassian_2d( xx, yy, wavelen, waist0, dist_from_waist=0.0, E0=1.0, refractive_index=1.0 ):

    zR = zrayleigh( wavelen, waist0, refractive_index=refractive_index )
    waistz = waist_gaussian( waist0, dist_from_waist, zR )
    Rcurve = radius_curve( dist_from_waist, zR )
    rr = np.sqrt( xx**2 + yy**2 )
    
    r_phase = (2*pi / wavelen) * rr**2 / (2*Rcurve)
    Efld = E0 * (waist0 / waistz) * np.exp( -( rr / waistz )**2 ) * np.exp( -1j * r_phase )

    return Efld

def prop_fresnel( U_0, opl, wavelen, px_size ):
    '''
    H(f_x, f_y) = e^{ikz} e^{-i\pi\lambda(f_x^2, f_y^2)}
    '''
    # source array size
    Nx, Ny = U_0.shape
    opl_max_TF = (Nx*px_size**2) / wavelen
    if opl > opl_max_TF:
        print(f'Validity of calculation warning: OPL = {opl:0.6f} m > OPL_max = {opl_max_TF:0.6f} m')
    else:
        print(f'Calculation in validity region: OPL = {opl:0.6f} m < OPL_max = {opl_max_TF:0.6f} m')

    wavenum = 2*np.pi/wavelen # wave number, k

    # spatial frequencies
    fmax = 0.5*(1/px_size) # max spatial frequency in FFT
    delta_fx = 1 / (Nx*px_size)
    delta_fy = 1 / (Ny*px_size)
    fx = np.linspace( -fmax, fmax-delta_fx, Nx )
    fy = np.linspace( -fmax, fmax-delta_fy, Ny )
    ffx, ffy = np.meshgrid(fx, fy, indexing='xy')
    
    # Fresnel transfer function eval at fx, fy
    fresnel_TF = np.exp(-1j*np.pi*opl*(ffx**2 + ffy**2))
    fresnel_TF = fft.fftshift( fresnel_TF )

    # FFT of source field
    U_0_fft = fft.fft2( fft.fftshift(U_0) )
    # multiply in Fourier space
    U_1_fft = U_0_fft * fresnel_TF
    # invert FFT
    U_1 = fft.ifftshift( fft.ifft2(U_1_fft) )
    # U_1 = ( fft.ifft2(U_1_fft) )

    return U_1, fresnel_TF, U_0_fft, U_1_fft

def diffract_size( wavelen, source_halfwidth, smallest_feature_halfwidth, distance ):
    '''
    From the angular spectrum formulation, a square will have a sinc(angle) type of angular spectrum. The angle at sinc's first zero is used to estimate the diffration angle in the observation plane.


    '''

    ang_diffract = np.arccos( wavelen / (2*smallest_feature_halfwidth) )
    print(f'angle = {np.rad2deg(ang_diffract):0.3f} deg')
    observation_plane_extent_min = source_halfwidth + distance*np.tan( np.pi/2 - ang_diffract )

    return observation_plane_extent_min

def FFTpts_required( field_points, wavelength, spatial_resolution):
    """
    N_FFT >= wavelen * field_points / spatial_res

    This relation is required for FFT of field to yield reasonable results.

    Assumes that the discrete field representation is a square 2D array of size field_points. 

    """
    N_FFT = wavelength * field_points / spatial_resolution

    return int(N_FFT) + 1

def prop_angspect( U_0, opl, wavelen, px_size ):
    '''
    Propagate source plane using Angular spectrum approach.
    '''
    wavenum = 2*np.pi/wavelen
    
    Nx, Ny = U_0.shape
    
    A_0 = fft.fftshift(fft.fft2( U_0 ) ) # angular spectrum of init intensity

    fx = fft.fftshift(fft.fftfreq( Nx, d=px_size ))
    fy = fft.fftshift(fft.fftfreq( Ny, d=px_size ))

    ffx,ffy = np.meshgrid(fx, fy)

    musquare = wavenum**2 - (2*np.pi)**2 * (ffx**2 + ffy**2)
    # propagating and evenescent waves
    temp = np.sqrt(np.abs(musquare))
    mu = np.where(musquare >= 0, temp, 1j*temp)
    
    # calc angular spectrum at Z
    A_z = A_0 * np.exp(1j * opl * mu)
    
    U_z = fft.ifft2( fft.ifftshift( A_z) )

    return U_z, A_0, A_z

