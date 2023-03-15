# ==========
# Created by Ivan Gadjev
# 2020.02.01
#
# Library of custom functions that aid in projects using pyRadia, genesis, elegant, and varius other codes and analyses. 
#   
# 
# ==========

# import sys
# import os
import time
import bisect
import json

import scipy.constants as pc
import numpy as np

from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt


# make a custom JSON encoder
class NdArrEncoder(json.JSONEncoder):
    """
    Class to encode nd.array into JSON. Inhehrits from json.JSONEncoder.
    Useful for storing PSO output to file.
    """
    def default(self, obj):
        """
        default() method of JSONEncoder that is re-implemented to handle np.ndarray
        """

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # this return statement allows JSONEncoder to handle errors and stuff
        return json.JSONEncoder.default(self, obj)


#
# ===== global defs =====
#

xhat = np.array([1,0,0])
yhat = np.array([0,1,0])
zhat = np.array([0,0,1])
origin = np.array([0,0,0])

# physical constants
mc2 = 1e-6 * (pc.m_e*pc.c**2)/pc.elementary_charge # MeV. electron mass
mpc2 = 1e-6 * (pc.m_p*pc.c**2)/pc.elementary_charge # MeV. proton mass

#
# ===== helper functions =====
#

def rot_mat(angle, dim=3, axis='z'):
    """ Create a rotation matrix for rotation around the specified axis.
    """
    # sine cosine
    c = np.cos(angle)
    s = np.sin(angle)

    if dim == 3:
        if axis == 'x':
            rr = np.array( [[1, 0, 0], 
                            [0, c, -s], 
                            [0, s, c]] )
        elif axis == 'y':
            rr = np.array( [[c, 0, s], 
                            [0, 1, 0], 
                            [-s, 0, c]] )
        elif axis == 'z':
            rr = np.array( [[c, -s, 0], 
                            [s, c, 0], 
                            [0, 0, 1]] )
    elif dim == 2:
        rr = np.array( [[c, -s], 
                        [s, c]] )
    else:
        print(' error: `dim` variable must be 2 or 3. Specifies the dimension of the vectors that the rotation is acting on.')
    return rr

def rgb_color(value, cmapname='viridis'):
    """ returns the RGB values of the color from the colormap specified with scaled value.
    value - between 0 and 1
    cmapname='viridis' name of the colormap
    """ 
    cmap = ScalarMappable(cmap=cmapname)
    return list(cmap.to_rgba(value, bytes=False, norm=False)[:-1])

def contourplot(xx, yy, zz, 
                 zlims='auto', 
                 arat=1, contours=True, 
                 cbarlabel='<color_bar_label>', colormap='viridis', 
                 xticks='auto', yticks='auto',
                 imkwargs=None, ckwargs=None):
    """ make a contour plot on top of an imshow figure for the given arrays
    xx, yy, zz are arrays from np.meshgrid() see numpy docs for ref
    s"""
    
    # compute limits for the plots
    xylims = [xx.min(),xx.max(), yy.min(),yy.max()]
    
    if (type(xticks) is np.ndarray) and (type(yticks) is np.ndarray):
        try:
            xt = np.unique(xx)
            xstep = np.abs(xt[1] - xt[0])
            yt = np.unique(yy)
            ystep = np.abs(yt[1] - yt[0])
            xylims = [xticks.min() - xstep/2, xticks.max() + xstep/2, yticks.min() - ystep/2, yticks.max() + ystep/2]
        except AttributeError:
            pass

    if zlims == 'auto':
        zmin = zz.min()
        zmax = zz.max()
    else:
        zmin = zlims[0]
        zmax = zlims[1]
    
    fig, ax = plt.subplots(figsize=(12,12))
    
    if imkwargs == None:
        imkwargs = dict(cmap=colormap, alpha=0.99, extent=xylims, vmin=zmin,vmax=zmax, interpolation='none', origin='lower')

    im = ax.imshow(zz, **imkwargs)
    ax.set_aspect(arat)

    CS = 0
    if contours:
        if ckwargs == None:
            ckwargs = dict(colors='white', extent=xylims, vmin=zmin,vmax=zmax,levels=19, linewidths=0.5, origin='lower')

        CS = ax.contour(xx,yy,zz,**ckwargs)
        
        # ax.clabel(CS, fontsize=14, inline=True, fmt='%1.0f')
    
    
    fs = dict(fontsize=18)
    if cbarlabel != 'none':
        cbar = fig.colorbar(im, shrink=0.7)
        
        # cbar.ax.set_ylabel('Tesla', **fs)
        cbar.ax.set_ylabel(cbarlabel, **fs)
        # cbar.add_lines(CS)
        cbar.ax.tick_params(labelcolor='k', labelsize=14, width=1)

#     plt.axis([-10,10,-10,10])

    if type(xticks) is np.ndarray:
        print('HELLO-------')
        ax.set_xticks(xticks)
    elif xticks == 'auto':
        xticks = np.unique(np.round(xx, decimals=3))
        # print(xticks)
        plt.xticks(xticks)
    
    if type(yticks) is np.ndarray:
        ax.set_yticks(yticks)
    elif yticks == 'auto':
        yticks = np.unique(np.round(xx, decimals=3))
        # print(yticks)
        plt.yticks(yticks)
        
    return fig, ax, CS

def gaussian(xvec, mean, std, norm='max'):
    """ Returns the Gaussian of the xvec normalized so that the peak is unity.
    xvec - ndarray of floats 
    mean - float. the center of the Gaussian
    std - float. the standard deviation of the Gaussian
    norm - {'max', 'area'} whether to normalize so that the maximum is unity or the area under the curve is unity

    yvec = normconstant * Exp( - (x - mean)^2 / 2std^2 )

    """

    yvec = np.exp( - (xvec - mean)**2 / (2 * std**2) )

    if norm == 'area':
        normconstant = 1 / (std * np.sqrt(2 * pc.pi))
    else:
        normconstant = 1

    return normconstant * yvec

def gaussianDx(x, mu, sig):
    """ First derivative of a Gaussian. Normalized so that its maximum is 1.
    x - ndarray
    mu - float. mean
    sig - float. std dev
    
    """
    
    y = (-(x - mu)/sig**2) * gaussian(x, mu, sig)
    # NOTE: normalization to maximum value = 1
    y = y/np.abs(y).max()
    
    return y

def gaussianDx2(x, mu, sig):
    """ Second derivative of a Gaussian (inverted sombrero). Normalized so that its maximum is 1.
    x - ndarray
    mu - float. mean
    sig - float. std dev
    
    """
    
    y = ( -1/sig**2 + ((x - mu)/sig**2)**2 ) * gaussian(x, mu, sig)
    # NOTE: normalization to maximum value = 1
    y = y/np.abs(y).max()
    return y

def stepfunc(xvec, center, width, norm='max'):
    """
    Returns a step function with center and total width.
    xvec - ndarray of floats 
    center - float
    width - float. total width of box or step
    norm - {'max', 'area'} whether to normalize so that the maximum is unity or the area under the curve is unity
    """

    yvec = [ int((xx >= center - width/2) and (xx <= center + width/2)) for xx in xvec]
    if norm == 'area':
        normconstant = 1 / np.trapz(yvec, xvec)
    else:
        normconstant = 1

    return normconstant * yvec



#
# === physics specific functions
#
class laser():

    def bwlimit(freq0, timeFWHM, mode='gauss'):
        """ Calculate FWHM BW of a laser pulse, given a central frequency and the pulse length FWHM.

        return freqFWHM, wlFWHM
        """

        bwfactor = {'unit':1.0, 'gauss':0.441, 'sech':0.315}
        freqFWHM = bwfactor[mode] / timeFWHM

        # convert to wavelength
        wlFWHM = (pc.c / freq0**2) * freqFWHM

        return freqFWHM, wlFWHM

    def zrayleigh(w0, wavelength):
        """
            Rayleigh range. For a Gaussian beam, it is the distance from the focus at which the area of the laser spot-size is doubled -> beam radius is increased by sqrt(2). 
        """
        zr = pc.pi * w0**2 / wavelength
        
        return zr

    def waist(w0, z, wavelength, M2):
        
        """
        Waist of the laser beam with minimum size w0. Distance z away from minimum.
        W(z) = W0 * ( 1 + M2^2  ( z / zR )^2 )
        """

        wz = w0 * np.sqrt( 1 + M2**2 * (z / laser.zrayleigh(w0, wavelength) )**2 )

        return wz

    def fluence(waist, energy):
        """
        The fluence per pulse.
        fluence units are Joules / cm^2 

        waist - [m] waist of laser pulse 
        energy - [J] energy of laser pulse
        """
        waist = 100 * waist # convert to cm
        flu = energy / ( pc.pi * waist**2)
        return flu

    def efld(waist, power):
        """
        Electric field corresponding to a given power and spot-size. 
         
        """
        E0 = np.sqrt( np.sqrt(2)*power / (pc.epsilon_0*pc.c*pc.pi*waist**2) )
        return E0

    def a0potential(E0, wavelength):
        """
        Normalized vector potential of the laser pulse.
        """
        const = pc.elementary_charge / (2*pc.pi*pc.m_e*pc.c**2)
        a0 = const * E0 * wavelength
        return a0

    def photon_energy(wavelength):
        """
        Returns the energy of a photon with the specified wavelength.
        output is in units of Joules.
        divide by elementary charge ~ 1.6e-19 to convert to eV
        """
        return pc.h * pc.c / wavelength




#
##
### === Transfer matrix beamlines
##
#

class BLElement(object):
    """
    Optical element represented by an ABCD matrix. 

    drift:
        properties = {'eletype':'drift',
                      'position':0,
                      'length':0}
    lens:
        properties = {'eletype':'lens',
                      'position':0,
                      'focal_len':0}
    quad:
        properties = {'eletype':'quad',
                      'position':0.0,
                      'maglen':0.0,
                      'gradient':0.0,
                      'totengMeV':0.0,
                      'restmassMeV':0.0,
                      'charge':1,
                      'focus':{'focus','defocus'}}
    """
    def __init__(self, name, eleprops={'eletype':'drift','position':0,'length':0}):
        
        self.name = name
        self.properties = eleprops
        
        if self.properties['eletype'] == 'drift':
            self.mat2x2 = np.array([[1, self.properties['length']],
                                      [0, 1]])
        
        elif self.properties['eletype'] == 'lens':
            self.mat2x2 = np.array([[1, 0],
                                      [-1 / self.properties['focal_len'], 1]])
        elif self.properties['eletype'] == 'quad':
            matargs = [  self.properties['maglen']
                        ,self.properties['gradient']
                        ,self.properties['totengMeV']
                        ,self.properties['restmassMeV']
                        ]
            matkwargs ={  'charge':self.properties['charge']
                         ,'focus':self.properties['focus'] 
                         }
            self.mat2x2 = mat_quad(*matargs,**matkwargs)

        else:
            print('WARNING: Element type not supported. Created drift with zero length.')

        

class BeamLine(object):
    """
    Optical beamline class. 
    Store elements as matricies (ABCD or electron transfer matricies). 

   
    """

    def __init__(self):
        
        self.element_list = []
        self.element_names = []
        self.element_position = []

    def add_element(self, element):
        """ Adds an element class to the beamline.
        
        element is a class BLElement() or similar.
        The element is inserted into the list based on its position, using `bisect`. 
        """
        try:
            # insert the element based on its positionin along the beamline
            posi = element.properties['position']
            bisect.insort(self.element_position, posi)
            addindex = self.element_position.index(posi)
            self.element_list.insert(addindex, element)
            self.element_names.insert(addindex, element.name)
            
        except AttributeError:
            print('ERROR! element does not possess the required attributes. Need class with .name and .properties()')
            return 1

        return 0

    def del_element(self, elementname):
        """ Deletes the specified element from the beamline
        """
        try:
            delindex = self.element_names.index(elementname)
            del self.element_list[delindex]
            del self.element_names[delindex]
            del self.element_position[delindex]

        except ValueError:
            print('WARNING: Beamline does not contain element with that name.')
            return 1

        return 0

    def make_mat(self, senter, sexit, ytransport=False):
        """ Based on the given location in the beamline, find the total transfer matrix.
        Assumes that the elements are sorted by their position. This is the case when the .add_element() method was used to add the element to the beamline.
        """
        # init transport matrix
        transportmatrix = np.eye(2)
        # print(transportmatrix)

        # position temporary
        si = senter
        
        # include elements up to sexit point
        s0index = bisect.bisect(self.element_position, senter) 
        s1index = bisect.bisect(self.element_position, sexit)
        # print(s0index)
        # print(s1index)
        for i,ele in enumerate(self.element_list[s0index:s1index],start=s0index):
            # print(ele.name)
            # drift to element from previous position
            driftlen = self.element_position[i] - si
            # print(driftlen)
            # print(transportmatrix)
            matdrift = BLElement('tempdrift', eleprops={'eletype':'drift','position':0, 'length':driftlen}).mat2x2
            # print(matdrift)
            transportmatrix = np.matmul(matdrift, transportmatrix)
            # print(transportmatrix)
            # element
            if ytransport and (ele.properties['eletype'] == 'quad'):
                elemat2x2 = ele.mat2x2 * np.array([[1,1],[-1,1]])
            else:
                elemat2x2 = ele.mat2x2
            transportmatrix = np.matmul(elemat2x2, transportmatrix)
            # print(transportmatrix)
            try:
                # sold = si
                si = self.element_position[i]
            except IndexError:
                
                continue

        

        # drift to final position from last element before final position
        driftlen = sexit - si
        
        matdrift = BLElement('tempdrift', eleprops={'eletype':'drift', 'position':0, 'length':driftlen}).mat2x2

        transportmatrix = np.matmul(matdrift, transportmatrix)
        # print(transportmatrix)


        return transportmatrix 

    def ray_trace(self, invec, inpos, outpos, ytransport=False):
        """ Given a set of input vectors with initial transverse postion and angle, calculate their transport through the beamline at the specified outpos.

        invec - ndarray.shape = (2,n)
        inpos - float position at which invec is specified
        outpos = ndarray.shape = (m,)

        """
        # inital transport to first point
        transmat = self.make_mat(inpos, outpos[0])
        outvec = np.matmul(transmat, invec) # include start position in output
        outvec = np.stack((invec, outvec)) # joins arrays along new axis
        
        for i,ss in enumerate(outpos[1:], start=1):

            transmat = self.make_mat(outpos[i-1], ss, ytransport=ytransport)
            outvectemp = np.matmul(transmat, outvec[-1])
            outvec = np.concatenate((outvec, [outvectemp]), axis=0)

        outpos = np.insert(outpos, 0, inpos)

        # calculate centroid position
        outvec_posmean = np.array( [ outvec[i,0,:].mean() for i in range(outvec.shape[0]) ] )
        outvec_angmean = np.array( [ outvec[i,1,:].mean() for i in range(outvec.shape[0]) ] )
        outvecMEAN = np.c_[outvec_posmean, outvec_angmean]
        # calculate RMS size and angle
        outvec_posstd = np.array( [ outvec[i,0,:].std() for i in range(outvec.shape[0]) ] )
        outvec_angstd = np.array( [ outvec[i,1,:].std() for i in range(outvec.shape[0]) ] )
        outvecRMS = np.c_[outvec_posstd, outvec_angstd]

        return outpos, outvec, outvecMEAN, outvecRMS
            
