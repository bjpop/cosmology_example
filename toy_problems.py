'''
Some toy problems to practice python
'''

import numpy as np
import matplotlib.pyplot as plt


def lists_of_list(l, m):
    '''
    Makes lists of a list.
    Returns a nested list.
    '''
    abet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z']
    lst = [[abet[ll] for ll in xrange(l)] for mm in xrange(m)]
    return lst

def list_of_lists(l, m):
    '''
    Makes a list of lists.
    Returns a nested list.
    '''
    abet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z']
    lst = [[abet[mm] for ll in xrange(l)] for mm in xrange(m)]
    return lst

def tup_to_list(list_of_tups):
    '''
    Takes a list of tuples, and returns a
    list of lists, with the first list containing
    the first entry in the tuples, the
    second list containing the second
    entry in the tuples.
    Returns a list of lists.
    '''
    for tup in list_of_tups:
        lst1.append(tup[0])
        lst2.append(tup[1])
    return lst1, lst2


def tuple_to_dict(data):
    '''
    Takes a list of tuples, and returns a
    dictionary with 'key' that first entry in
    tuples, and entry a list of the sencond
    entry in tuples with first entry key.
    Returns a dictionary.
    '''
    dict_tup = {}
    for tup in data:
        try:
            dict_tup[tup[0]].append(tup[1])
        except KeyError:
            dict_tup[tup[0]] = [tup[1]]
    return dict_tup

def sort_tups(data):
    '''
    Takes a list of tuples, and returns a
    list of the same tups, sorted by the first
    tuple entry.
    Returns a list of tups.
    '''
    sorted_tup = sorted(data, key=lambda tup: tup[0])
    return sorted_tup


def f_sum_arr(x0, x1, xi=1.0):
    '''
    A function to sum from x0, with x1 steps of size
    xi, keeping the result for each succesive step.
    default is xi = 1
    returns a numpy array
    '''
    step = x0
    nsteps = x1
    output = np.zeros(nsteps+1)
    count = 0
    output[0] = step
    while count < x1:
        count += 1
        step += xi
        output[count] = step
    return output


def remove_empty_tuples(tup_list):
    '''
    Removes empty tuples from a list of tuples.
    returns a list of tuples
    '''
    tple = [tpl for tpl in tup_list if tpl]
    return tple


def factorial(x):
    '''
    A function to compute factorial(x),
    where x is an intiger.
    That is, fact(x) = 1*2*3*4*...*x
    returns an int
    '''
    i = 1
    y = x

    while i < y:
        x *= (i)
        i += 1

    return x


def modulus(x, modu):
    '''
    Computes x mod modu.
    returns a float or an int
    '''
    signx = x/abs(x)
    signmod = modu/abs(modu)

    y = abs(x)
    moduy = abs(modu)

    while y > moduy:
        y -= moduy

    if signx < 0.:
        if signmod > 0.:
            return signx*y+modu
        else:
            return -y

    else:
        if signmod < 0.:
            return signx*y+modu
        else:
            return y

def plot_x_xsq(xmin, xmax, numpts=100):
    '''
    Plots x vs x^2 and displays it.
    Returns nothing
    '''
    x = np.linspace(xmin, xmax, numpts)
    plt.close()
    plt.plot(x, x**2)
    plt.show()

def main():
    '''
    This is a special function name. 
    Everything in your code should
    run inside main()
    '''

    sumtest_arr = f_sum_arr(5, 10, xi=1.5)
    print 'sumtest_arr', sumtest_arr
    x_mod_z = modulus(100, 12)
    print 'x_mod_z', x_mod_z
    fact = factorial(10)
    print 'factorial', fact
    tpl = remove_empty_tuples([('f', 3), ('e', 9), ('j', 5), (), ('d', 1), ('g', 2)])
    print 'tuple', tpl
    lst = lists_of_list(5, 3)
    print 'list', lst
    lst2 = list_of_lists(5, 3)
    print 'list', lst2
    sorted_tup = sort_tups([('f', 3), ('e', 9), ('j', 5), ('d', 1), ('g', 2),
                                ('g', 3), ('g', 12), ('i', 2), ('e', 2)])
    print 'sorted_tup', sorted_tup
    dict_tup = tuple_to_dict([('f', 3), ('e', 9), ('j', 5), ('d', 1), ('g', 2),
                                ('g', 3), ('g', 12), ('i', 2), ('e', 2)])
    print 'dict_tup', dict_tup
    # plot_x_xsq(0.,100.)


if __name__ == '__main__':
    main()
