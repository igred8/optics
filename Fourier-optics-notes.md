# Scalar Diffraction Theory


## Angular Spectrum Propagation
---
__Ersoy - Diffraction, Fourier optics, and imaging 2007__

__Rafael de la Fuente [Diffraction Simulations](https://rafael-fuente.github.io/simulating-diffraction-patterns-with-the-angular-spectrum-method-and-python.html)__

## Continuous formulation

The phasor representation of a scalar field propagating by the Helmholtz equation is used for mathematical ease. The scalar field is complex valued, but only the real part is taken for computing measurable quantities.

$$ U(x,y,z) = \mathfrak{Re}\big[U(\vec{r})e^{i\phi(\vec{r})}e^{i\omega t} \big]$$
Usually the $\mathfrak{Re}$ is implicitly understood from context and omitted inside equations.

The 2D Fourier transform of the scalar field is referred to as the angular spectrum of the field. 

$$ A(k_x,k_y, z) = \mathbb{F^{-1}}[ U(x,y,z) ] $$

The electric field at a screen located at some propagation distance, z, is the complex valued function, $U(x,y,z)$. By separating out the time dependence and rearranging terms in the electric field Maxwell equations for a free-space wave, we arrive at the Helmholtz equation, which the complex electric field must satisfy:

$$ (\nabla^2 + k^2)U(x,y,z) = 0 $$

, where the wavenumber is, $k = \frac{2\pi}{\lambda}$.

Inserting the angular spectrum representation of $U$ into the Helmholtz eq:
$$ (\nabla^2 + k^2)\mathbb{F}[ A(k_x, k_y, z) ] = 0$$
$$ \int\int (\nabla^2_{x,y,z} + k^2)A(k_x, k_y, z) e^{i2\pi(k_x x + k_y y)} dk_x dk_y = 0$$

Taking the x and y derivatives, the integrand has to be itself equal to zero for the equation to be satisfied regardless of $A$. 

$$ \frac{d^2}{dz^2}A(k_x, k_y, z) + (k^2 -4\pi^2(k_x^2 + k_y^2))A(k_x, k_y, z) = 0 $$

The solution to this differential equation is: 
$$ A(k_x, k_y, z) = A(k_x, k_y, 0)e^{i k_z z} $$
, where $k_z = \sqrt{k^2 -4\pi^2(k_x^2 + k_y^2)}$

The angular spectrum is a function of the spatial frequencies, $k_x, k_y$ and the distance $z$.

If the angular spectrum of the field is known at some earlier point in $z$, then it can be propagated by the above relation. The field at this later point is obtained by the Fourier transform of $A(k_x, k_y, z)$.

$$ U(x,y,z) = \mathbb{F}\Big[ A(k_x, k_y, 0)e^{i k_z z} \Big] = \mathbb{F}\Big[ \mathbb{F^{-1}}[ U(x,y,0) ]e^{i k_z z} \Big]  $$

When $k_z$ is imaginary, the solutions have an exponentially decaying term and are called evenascent. For wave propagtion, the region of interest is the circular region where:  

$$ 4\pi^2(k_x^2 + k_y^2) \le k^2 $$

and looking at distances in the far-field so that $z\gg\lambda$ and evenascent waves are negligible. 

## FFT (discrete and practical stuff)

The discrete representation of the $U$ and $A$ functions can be written as:
$$ U(\Delta x n_x, \Delta y n_y, z) = U(\Delta s n_x, \Delta s n_y, z)$$ 
$$ A(\Delta k_x m_x, \Delta k_y m_y, z) = A(\Delta f m_x, \Delta f m_y, z) $$ 

where we assume that the number of space and frequency points in $x$ and $y$ are the same for simplicity without too much loss of generalization i.e. $N_x=N_y=N, M_x=M_y=M$ 
$$ -N \le n_x, n_y \le N$$
$$ -M \le m_x, m_y \le M$$

An approximation that satisfies the inequality $4\pi^2(k_x^2 + k_y^2) \le k^2$, is to take the rectangular region in Fourier space such that: 
$$ 2\pi|f_{max}| = 2\pi\Delta f M \le k $$

From here, we can conclude that the spatial frequency resolution, $\Delta f$, of the FFT is:
$$ \Delta f \le \frac{1}{\lambda M} $$
, where $\lambda = 2\pi/k$ is the wavelength of the electric field described by $U$.
  
In order to be able to use the FFT, we must satisfy: $\Delta s \Delta f = 1/N$. From here, we obtain a relation between the number of points in the FFT:
$$ N \ge \frac{\lambda}{\Delta s}M $$
M - number of points in the discrete representation of the E-field 

N - number of points in the FFT of the E-field

$\Delta s$ - spatial resolution of the E-field plane-wave front

$\Delta f$ - frequency resolution of the FFT of the E-field plane-wave front

$\lambda$ - wavelength of the E-field plane-wave represented by $U$

## Array of emitters
---




---

## Fresnel Transfer Function
---

__Voelz - Computational Fourier Optics 2010__

Due to finite sampling of transfer function, there is aliasing in numerical propagtion schemes.

Transfer function approach is valid when the distance of propagation is relatively short or the wavelength is short.

$$\Delta x > \frac{\lambda z}{L}$$ 
or 
$$ z < \frac{L \Delta x}{\lambda } $$  

where:

$\Delta x = L/M$ - is the sample interval

$\lambda$ - wavelength

$z$ - propagation distance

$L$ - the source plane extent, which must be larger than the source function support $L > D$

$M$ - the size of the source array

The Fresnel transfer function is:
$$ H(f_x, f_y) = e^{ikz} \exp{[-i\pi\lambda z(f_x^2 + f_y^2)]} $$

The second exponential is referred to as the chirp. The spatial frequencies are in the range $[-\frac{1}{2\Delta x}, \frac{1}{2\Delta x}]$ 

The propagation of a source field is then computed by evaluation of:

$$ U(x,y,z) = \mathbb{F^{-1}}\Big[ \mathbb{F}[U(x,y,0)] H(f_x, f_y) \Big] $$
