# A Cosmology Calculator 

The propose of the program is to provide some of the more commonly used cosmological parameters, as a function of redshift. The parameters provided are the comoving distance, $r$, the luminosity distance, $D_L$, the angular diameter distance $D_A$ and the age of the universe at redshift z, $t_{age}$. 

If you're in astro, this will be a program you will almost certainly use at some point, if you're not in astro, it will at least be a simple program for you to use to learn python. 

These notes are in no way intended to give you a full understanding of the parameters and their meaning, or derivation. They are sufficient to allow you to write a cosmology calculator, and no more. For a more thorough run down of the cosmology, and a full understanding of the derivation and significance of these parameters, I recommend Stuart Wyithe and Bart Pindor’s Cosmology Notes for the Masters Cosmology Course, in particular chapters 2 thought to 10. 

I will explain each of the parameters in question below, but first I need to introduce the Friedman Equation and some other things. The Friedman equation is 

$$H(z)^2 = H_0^2[\Omega_ma_{-3}+\Omega_Ra^{-4}+\Omega_\Lambda]$$$,

where 

$$a = 1/(1+z)$$.

This equation give the evolution of the Hubble parameter with redshift. The Hubble constant, which you may be familiar with doesn't take into account the evolution of the universe. It give the expansion of the universe *today* at redshift 0. The way the expansion of the universe evolves with redshift is encapsulated by the terms in the square brackets in the equation above. 
Note that redshift doesn't get larger with time, it gets larger the further *back* in time you go, and is zero now. In other words it is defined to be zero at the present, and to be infinite at the big bang. This makes sense when you think about what redshift means. Redshift gives the fractional shift in the wavelength of light emitted at some time in the past, relative to how we see it now. So it makes sense that something emitted now will have a redshift of 0 because it hasn't had its wavelength shifted at all. The further back you look (which is equivalent to saying the further away you look), the more the wavelength of the light has been stretched (i.e. redshifted) due to the expansion of the universe. 

A characteristic scale in cosmology is the *Hubble distance*, $d_H$. This is the size of the observable universe at a given redshift,

$$d_H = c/H(z)$$.

Another way of thinking of this is that the Hubble distance is the distance between the Earth and the galaxies which are currently receding from us at the speed of light (i.e. things which are just now becoming invisible to us because their light can never reach us). 

We now have what we need to get the comoving distance.

### Comoving distance (or proper distance)
This is the distance to an object in *comoving* coordinates. That is, this distance takes the expansion of the universe into account. 

The comoving distance is obtained by integrating the Hubble distance from the redshift in question to now (so from redshift z to redshift 0)

$$r = \int_0^z d_H dz$$,

or expanding

$$r = \int_0^z (c/H_0)/\sqrt(\Omega_ma_{-3}+\Omega_Ra^{-4}+\Omega_\Lambda)dz$$





