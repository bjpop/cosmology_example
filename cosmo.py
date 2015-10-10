#!/usr/bin/env python

'''
A cosmology calculator written by Robert L. Barone Nugent,
Catherine O. de Burgh-Day and Jaehong Park, 2014.

A simpler version of the course - compute less variables.
'''

import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import argparse

# Define the deafult values for the argument parser
DEFAULT_REDSHIFT_RNG = [-1.0, 4.0, 20]
DEFAULT_HUBBLE_CONSTANT = 70.0
DEFAULT_MASS_DENSITY = 0.3
DEFAULT_DARK_ENERGY_DENSITY = 0.7

def freidman_equation(redshift, mass_density, dark_energy_density):
    '''Return the *dimensionless* friedman equation
    (i.e. without the H_0**2 term)'''
    return np.sqrt(mass_density * (1.0 + redshift) ** 3.0 + \
        dark_energy_density * (1.0 + redshift))


def tage_integral(redshift, mass_density, dark_energy_density):
    '''Compute the age of the universe'''
    return 1.0 / ((1.0 + redshift) * \
        freidman_equation(redshift, mass_density, dark_energy_density))


def freidman_integral(redshift, mass_density, dark_energy_density):
    '''Compute the proper distance'''
    return 1.0 / freidman_equation(redshift, mass_density, dark_energy_density)


def cosmo(redshift, hubble_constant,
          mass_density=DEFAULT_MASS_DENSITY,
          dark_energy_density=DEFAULT_DARK_ENERGY_DENSITY):
    """
    A simple cosmology calculator.
    Default is concordance cosmology (Flat, negative pressure, radiation free,
    with hubble_constant = 70, mass_density = 0.3, dark_energy_density = 0.7)

    Inputs:
        redshift: the redshift (z) at which to compute the cosmology.
            Unfortunately because the integrator can't take arrays or lists
            this must be a number only.
        hubble_constant: Hubble's Constant (H0)
        mass_density: Omega_M
        dark_energy_density: Omega_Lambda.
    Returns: a Cosmology object
    """
    # These are all constants which you want to very high accuracy

    # units of km
    speed_of_light = 2.99792458e5
    # converts Mpc to km
    Mpc2km = 3.08567758147e+19
    # seconds in a julian year
    seconds_in_a_year = 31557600.0

    # hubble distance
    hubble_distance = speed_of_light / hubble_constant

    # Age at redshift in gigayears
    # The stuff after the integral is all unit conversions
    age_at_redshift = 1.0 / hubble_constant * \
        integrate.quad(lambda r: tage_integral(r, mass_density, dark_energy_density),
        redshift, np.inf)[0] * Mpc2km / (seconds_in_a_year) / (1.e9)

    proper_distance = hubble_distance * \
        integrate.quad(lambda r: freidman_integral(r, mass_density, dark_energy_density),
        0, redshift)[0]

    # The rest follows from the proper distance

    luminosity_distance = proper_distance * (1.0 + redshift)

    angular_diameter_distance = proper_distance / (1.0 + redshift)

    return proper_distance, luminosity_distance, angular_diameter_distance, age_at_redshift


def compute_cosmologies(arguments):
    '''Takes arguments from the argparser and uses them to
    1) generate an array of redshifts
    2) define a set of dictionaries to store the resulting cosmologies,
       containing empty lists
    3) for each cosmology, for each redshift, compute the cosmology
    4) append the resulting information into the relevant entry
       in the relevant dictionary
    5) return a list containing the array of redshifts,
       and a list of the dictionaries of cosmologies.'''
    # Array of log-spaced redshifts
    z_lower, z_upper, num_pts = arguments.redshift_range
    z_arr = np.logspace(z_lower, z_upper, num_pts)
    # Make an array of linearly-spaced redshifts
    # z = np.linspace(0.01, 1000, 20)

    # Initialise the dictionaries
    conc_cosm = {'r':[], 'DL':[], 'DA':[], 'tage':[]}
    user_cosm = {'r':[], 'DL':[], 'DA':[], 'tage':[]}
    FE_cosm = {'r':[], 'DL':[], 'DA':[], 'tage':[]}
    EdS_cosm = {'r':[], 'DL':[], 'DA':[], 'tage':[]}

    for z in z_arr:
        # Call the cosmology function for different scenarios
        conc = cosmo(z, arguments.hubble_const)
        user = cosmo(z, arguments.hubble_const, mass_density=arguments.omega_m, 
                                            dark_energy_density=arguments.omega_L)
        FE = cosmo(z, arguments.hubble_const, mass_density=0.0, dark_energy_density=1.0)
        EdS = cosmo(z, arguments.hubble_const, mass_density=1.0, dark_energy_density=0.0)

        # Put the results into dictionaries
        conc_cosm['r'].append(conc[0])
        conc_cosm['DL'].append(conc[1])
        conc_cosm['DA'].append(conc[2])
        conc_cosm['tage'].append(conc[3])
        user_cosm['r'].append(user[0])
        user_cosm['DL'].append(user[1])
        user_cosm['DA'].append(user[2])
        user_cosm['tage'].append(user[3])
        FE_cosm['r'].append(FE[0])
        FE_cosm['DL'].append(FE[1])
        FE_cosm['DA'].append(FE[2])
        FE_cosm['tage'].append(FE[3])
        EdS_cosm['r'].append(EdS[0])
        EdS_cosm['DL'].append(EdS[1])
        EdS_cosm['DA'].append(EdS[2])
        EdS_cosm['tage'].append(EdS[3])

    # Putting everything after z_arr in square brackets means they 
    # will be returned as one list. 
    return z_arr, [conc_cosm, user_cosm, EdS_cosm, FE_cosm]


def plot_cosmology(output_filename, z_arr, cosmologies):
    ''' Plots the four parameters for each of the cosmologies
        on a 2x2 plot. '''
    # Unpack the cosmologies
    conc_cosm, user_cosm, EdS_cosm, FE_cosm = cosmologies

    # set the tick label and axis label size for all plots
    # ** This won't work if you use Ipython becasue rcPramas gets 
    # imported when you use matplotlib the first time only in Ipython.
    # You'll need to re-open Ipython every time you chage these **
    matplotlib.rcParams['axes.labelsize'] = 30
    matplotlib.rcParams['xtick.labelsize'] = 30
    matplotlib.rcParams['ytick.labelsize'] = 30
    
    plt.close()
    # Create the figure instances in a 4x4 plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 18))

    # Plot onto the top left axis. 'label' is what is used to make the legend.
    ax1.plot(z_arr, EdS_cosm['r'], color='g', linewidth=3, label='Flat Empty')
    ax1.plot(z_arr, conc_cosm['r'], color='r', linewidth=3, label='Concordance')
    ax1.plot(z_arr, user_cosm['r'], color='m', linewidth=3, label='User Specified')
    ax1.plot(z_arr, FE_cosm['r'], color='b', linewidth=3, label='Einstein de Sitter')
    ax1.set_yscale('log') # Make the plots logscale
    ax1.set_xscale('log')
    ax1.set_ylabel('Proper distance (Mpc)')
    ax1.set_xlabel('Redshift, z')
    # increase the space between the y axis ticklabels and the axis label
    ax1.yaxis.labelpad = 20

    ax2.plot(z_arr, EdS_cosm['DA'], color='g', linewidth=3)
    ax2.plot(z_arr, conc_cosm['DA'], color='r', linewidth=3)
    ax2.plot(z_arr, user_cosm['DA'], color='m', linewidth=3)
    ax2.plot(z_arr, FE_cosm['DA'], color='b', linewidth=3)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylabel('Angular diameter distance (Mpc)')
    ax2.set_xlabel('Redshift, z')
    ax2.yaxis.labelpad = 20

    ax3.plot(z_arr, EdS_cosm['tage'], color='g', linewidth=3)
    ax3.plot(z_arr, conc_cosm['tage'], color='r', linewidth=3)
    ax3.plot(z_arr, user_cosm['tage'], color='m', linewidth=3)
    ax3.plot(z_arr, FE_cosm['tage'], color='b', linewidth=3)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_ylim(ax3.get_ylim()[0], 1e4)
    ax3.set_ylabel('Age at redshift z (Gyr)')
    ax3.set_xlabel('Redshift, z')
    ax3.yaxis.labelpad = 20

    ax4.plot(z_arr, EdS_cosm['DL'], color='g', linewidth=3)
    ax4.plot(z_arr, conc_cosm['DL'], color='r', linewidth=3)
    ax4.plot(z_arr, user_cosm['DL'], color='m', linewidth=3)
    ax4.plot(z_arr, FE_cosm['DL'], color='b', linewidth=3)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.set_ylabel('Luminosity distance (Mpc)')
    ax4.set_xlabel('Redshift, z')
    ax4.yaxis.labelpad = 20

    # Make the legend on the top left axis
    ax1.legend(fontsize=25, loc=2)

    # Tidy up the layout
    plt.tight_layout(rect = (0.,0,1.,0.9))
    # Save the fig
    plt.savefig(output_filename)


def parse_arguments():
    '''Parse the command line arguments of the program'''
    
    parser = argparse.ArgumentParser(description='Cosmology example')
    
    parser.add_argument(
        '--plot_name', # The name of the input variable
        required=True, # Is it an optional variable?
        metavar='OUTPUT_PLOT_NAME', # For errormessage printing purposes
        type=str, # The datatype of the input variable
        # A description of the variable or error message purposes
        help='name of output file for plotted graphs'
        )

    parser.add_argument(
        '--redshift_range',
        required=False,
        metavar='REDSHIFT_RNG',
        type=float,
        nargs='+',
        help='A list containing the range of redshifts (log-spaced) and the number of points, defaults to {}'.format(DEFAULT_REDSHIFT_RNG),
        # If the variable is optional, this is the default value
        # (defined at the top of the code)
        default=DEFAULT_REDSHIFT_RNG
        )
    
    parser.add_argument(
        '--hubble_const',
        required=False,
        metavar='HUBBLE_CONSTANT',
        type=float,
        help='The hubble constant, H_0, defaults to {}'.format(DEFAULT_HUBBLE_CONSTANT),
        default=DEFAULT_HUBBLE_CONSTANT
        )

    parser.add_argument(
        '--omega_m',
        required=False,
        metavar='OMEGA_M',
        type=float,
        help='Mass density, Omeaga_m, defaults to {}'.format(DEFAULT_MASS_DENSITY),
        default=DEFAULT_MASS_DENSITY
        )

    parser.add_argument(
        '--omega_L',
        required=False,
        metavar='OMEGA_L',
        type=float,
        help='Dark energy density, Omeaga_L, defaults to {}'.format(DEFAULT_DARK_ENERGY_DENSITY),
        default=DEFAULT_DARK_ENERGY_DENSITY
        )

    return parser.parse_args()


def main():
    '''This is a special function name. Everything in your code should
        run inside main()'''
    arguments = parse_arguments()
    print 'Computing cosmology for:'
    print 'H_0 = %s, Omega_M = %s, Omega_L = %s' \
           % (arguments.hubble_const, arguments.omega_m, arguments.omega_L)
    z_arr, cosmologies = compute_cosmologies(arguments)
    print 'Plotting results, along with some common cosmologies.'
    print "Saving to: '%s'" % (arguments.plot_name)
    plot_cosmology(arguments.plot_name, z_arr, cosmologies)


if __name__ == '__main__':
    main()
