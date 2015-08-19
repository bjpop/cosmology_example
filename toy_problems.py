'''
Some toy problems to practice python. These are not the most 'pythonic' 
ways to solve these problems, but they are simple ways, which are very 
basic-syntax-heavy. The purpose of these problems is to practice using
basic python syntax, and to start thinking in a cody-y sort of way (for 
those for whom this is their first coding language)
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import re
import random
from scipy import integrate
import argparse

### 1 ###
def factorial(x):
    '''
    A function to compute factorial(x),
    where x is an intiger.
    That is, fact(x) = 1*2*3*4*...*x

    Returns an int.
    '''
    i = 1
    y = x

    while i < y:
        x *= i
        i += 1

    return x

### 2 ###
def modulus(x, modu):
    '''
    Computes x mod modu.

    Returns a float or an int.
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

### 3 ###
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

### 4 ###
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

### 5 ###
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

### 6 ###
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

### 7 ###
def sort_tups(data):
    '''
    Takes a list of tuples, and returns a
    list of the same tups, sorted by the first
    tuple entry.

    Returns a list of tupless.
    '''
    sorted_tup = sorted(data, key=lambda tup: tup)

    return sorted_tup

### 8 ###
def remove_empty_tuples(tup_list):
    '''
    Removes empty tuples from a list of tuples.

    Returns a list of tuples.
    '''
    tple = [tpl for tpl in tup_list if tpl]
    return tple

### 9 ###
def sum_arr(x0, x1, xi=1.0):
    '''
    A function to sum from x0, with x1 steps of size
    xi, keeping the result for each succesive step.
    default is xi = 1

    Returns a numpy array.
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

### 10 ###
def plot_x_pow_n(xmin, xmax, n, numpts=100):
    '''
    Plots x vs x^2 and displays it.

    Returns nothing.
    '''
    x = np.linspace(xmin, xmax, numpts)
    plt.close()

    # Basic plot
    plt.plot(x, x**n)
    # With some tweaking and stuff. There's so mauch you can customize...
    plt.plot(x, x**n, color = 'm', linewidth = 8, linestyle = ':', dashes = (30,10) )

    plt.axes().set_xlabel("here's an xlabel")
    plt.axes().set_ylabel('and a ylabel')
    plt.savefig('plot_eg.png')

### 11 ###
def fourxfour_plot(xmin, xmax, numpts=100):
    '''
    Plots a 4x4 set of axes, showing
    x^1, x^2, x^3, and x^4 respectively.

    Returns nothing.
    '''
    x = np.linspace(xmin, xmax, numpts)
    plt.close()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # Basic plot
    ax1.plot(x, x**1)
    ax2.plot(x, x**2)
    ax3.plot(x, x**3)
    ax4.plot(x, x**4)
    
    fig.tight_layout()
    fig.savefig('fxf_plot_eg.png')

### 12 ###
def scatterplot_random(xmin, xmax, numpts=100):
    '''
    Produces a scatterplot of random points
    in the range [xmin,xmax]

    Returns nothing.
    '''
    numpts = xrange(numpts)
    x = [random.randrange(xmin, xmax, 1.) for n in numpts]
    y = [random.randrange(xmin, xmax, 1.) for n in numpts]
    
    plt.close()
    plt.scatter(x,y, 30, color = 'r')
    plt.savefig('scatter_eg.png')

### 13 ###
def twodim_xsq_ysq(xmin, xmax, numpix=100):
    '''
    Produces a two-dimensional image
    (as opposed to scatter plot)
    of x^2 + y^2 in the range [xmin,xmax]

    Note that when using matplotlib.pyplot.imshow()
    you should always add the arguments 
    interpolation='nearest' and origin='lower'. 

    For some reason imshow by default smooths your image,
    and setting interpolation='nearest' turns this off 
    so the true pixels are shown. 
    The second argument is to set the origin of the image
    to the lower left hand corner of the image. For some 
    reason this is not the default!

    Returns nothing.
    '''
    
    x = np.linspace(0, numpix-1, numpix)
    y = np.linspace(0, numpix-1, numpix)
    
    xgrid, ygrid = np.meshgrid(x,y)

    plt.close()
    plt.imshow(xgrid**2+ygrid**2, interpolation='nearest', origin='lower')
    plt.savefig('imshow_eg.png')

### 14 ###
def read_data(filename, ncols, delimiter = ','):
    '''
    Reads data from a .csv file and
    returns each column in a list,
    witout using the csv module.

    'U' means universal, and allows for newline charachters
    from any operating system. E.g. Excel on OSX will use
    funky newlines that otherwise woudl cause issues.
    The alternative is to use 'r+', which means regular read
    with extra permissions
    Go to section 7.2 here:
    https://docs.python.org/2/tutorial/inputoutput.html
    for more on differnt read/write instructions for
    python.

    Returns an array containing the data.
    '''
    with open(filename, 'U') as csvfile:
        result = [[] for n in xrange(ncols)]
        for row in csvfile:
            cols = row.strip('\n').split(delimiter)
            for i,col in enumerate(cols):
                result[i].append(col)
    return np.array(result)

### 14 *alternative* ###
def read_csv(filename, ncols, delimiter = ','):
    '''
    Reads data from a .csv file and
    returns each column in a list,
    using the csv module.
    The csv module handles data containing
    quote charachters better than simple row
    reading as in read_data. The alternative is
    to force the data to be read as a regular
    expression by using the regex module.
    Google this to learn more.

    'U' means universal, and allows for newline charachters
    from any operating system. E.g. Excel on OSX will use
    funky newlines that otherwise woudl cause issues.
    The alternative is to use 'r+', which means regular read
    with extra permissions
    Go to section 7.2 here:
    https://docs.python.org/2/tutorial/inputoutput.html
    for more on differnt read/write instructions for
    python.

    Returns an array containing the data.
    '''
    with open(filename, 'U') as csvfile:
        reader = csv.reader(csvfile,
                            delimiter=delimiter,
                            quotechar='"')
        result = [[] for n in xrange(ncols)]
        for row in reader:
            for i,col in enumerate(row):
                result[i].append(col)
    return np.array(result)

### 15 ###
def write_data(data, savename, delimiter=','):
    '''
    writes the data to file called
    savename, with fields separated by
    the delimiter.
    'data' must be row by column
    Returns nothing.
    '''
    with open(savename,'w+') as f:
        for row in data:
            row = [str(r) for r in row] # make sure they're strings
            f.write(delimiter.join(row)+'\n')

### 16 ###
def join_csvs(filenames, ncols, join_ons, delimiters=None, savename='joined_data.csv', header=None):
    '''
    Takes a list of csv files and joins them on a given common column.
    Assumes there is no missing data in the files.
    You need to specify the coumn in each csv which is common with the
    other csv files, that is to be joined on.
    You also need to specify how many columns each csv has.
    Writes the joined data to savename, with the joining column first.
    If a header is specified for the joining column this is put at the
    top of the output file.

    Returns nothing.
    '''
    if delimiters==None:
        # Assume commas sep vals if no delimiter is specified
       delimiters = [',' for f in filenames]

    # This forces each data to be a numpy array cos we assume it is later on
    datas = [np.array(read_csv(f, nc, delimiter=d)) \
            for f,nc,d in zip(filenames,ncols,delimiters)]

    dicts = [{} for d in datas]

    for data, join_on, a_dict in zip(datas, join_ons, dicts):
        # data is assumed to be an array
        data = data.T # change from col by row to row by col
        for row in data:
            # when you add lists they join together, 
            # when you add arrays they must be the same shape and 
            # you add the values of the elements, elementwise
            # Hence we make these lists so we can avoid the row 
            # we join on
            a_dict[row[join_on]]=list(row[:join_on])+list(row[join_on+1:]) 

    # A defaultdict will automatically produce an entry for 
    # any key that doesn't yet exist. 
    # A regular dict like {} doesn't do this. 
    dd = defaultdict(list)
    # Join the dictionaries on the selected columns
    for d in dicts:
        for key, values in d.iteritems():
            for value in values:
                dd[key].append(value)

    # Put the joined info back into an array and write to file
    data = []
    # Ensure the header is the first thing written to file
    if header:
        data.append([header]+dd[header])
    for key, values in dd.iteritems():
        if key==header:
            continue # We already have it at the top of the data
        data.append([key]+values)
    data = np.array(data)
    write_data(data, savename, delimiter=',')

### 17 ###
def hist_plot(filename, savename, ncol, histcol, delimiter=',', header=False):
    '''
    Plots a histogram of a column from a datafile
    specified by filename. The column does NOT need to be
    numerical. If you're just trying to histogram numerical
    data you should use numpy.histogram()
    You need to specify the savename to save the plot to,
    the number of columns in the file, as well
    as the column to histogram need to be specified.
    Differet delimiters can also be optioanlly specified.
    Assumes there is no header. If there is, set header=True
    and the first row will be skipped.

    Returns nothing.
    '''
    data = np.array(read_csv(filename, ncol, delimiter=delimiter))
    data_to_hist = data[histcol]
    # for line in set(data_to_hist):
        # print line
    if header:
        data_to_hist = data_to_hist[1:]

    # A defaultdict will automatically produce an entry for 
    # any key that doesn't yet exist. 
    # A regular dict like {} doesn't do this. 
    datadict = defaultdict(int)
    for dat in data_to_hist:
        # Every time an item is found, add 1 to its entry
        datadict[dat] += 1

    # Split up the keys and data in the dictionary
    # into two tuples
    item, number = zip(*datadict.items())
    # Make a set of indices for the data bins (the keys)
    index, item = zip(*enumerate(item))

    plt.clf()
    # Plot the number of each item against the item index
    plt.bar(index, number, width = 1.)
    # Shift the ticks so they sit in the middle of the bins
    plt.axes().set_xticks([ind+0.5 for ind in index])
    # Label the bins wth the items (as opposed to the indices)
    plt.axes().set_xticklabels(item, rotation = 20, fontsize = 30)
    plt.savefig(savename)

### 18 ###
def count_instances_simple(filename, searchterm, ncol, thiscol = None, delimiter=',', header=False):
    '''
    Counts the number of instances of a string or
    number in a file, either from one column in the
    file, or from the whole file.

    Returns the number of whole word/whole number
    instances, the number of instances including
    within words/numbers, a list of locations
    giving the column and row number of each
    occurrence of the the word/number for the
    whole word/number search, and the same for
    the sub-word/number search.
    '''
    data = np.array(read_csv(filename, ncol, delimiter=delimiter))

    if header:
        data = data[:,1:]
    if thiscol != None:
        data = data[thiscol]

    n_instances_whole = 0
    n_instances_all = 0
    
    loc_whole = []
    loc_all = []
    for i, col in enumerate(data):
        for j, row in enumerate(col):

            # Whole word
            st = str(searchterm).upper()
            srow = str(row).upper()
            
            re_search_whole = r'\b'
            re_search_whole += '('+st+')'
            re_search_whole += r'\b'
            instances_whole = re.findall(re_search_whole, srow)
            
            re_search_all = r''
            re_search_all += '('+st+')'
            instances_all = re.findall(re_search_all, srow)

            n_instances_whole +=len(instances_whole)
            n_instances_all +=len(instances_all)

            # Python sees an empy list as a boolean False, 
            # and any other list as a boolean True, 
            # So the below line is the same as saying 
            # if len(instances)>0. 
            if n_instances_whole:
                loc_whole.append( (i, j) )
            if n_instances_all:
                loc_all.append( (i, j) )


    return n_instances_whole, n_instances_all, loc_whole, loc_all

### 19 ###
def integrate_invexp(a, b):
    '''
    Integrate the function a*e^(-b*x) from
    0 to infinity. 

    Returns the result of the integral.
    '''

    def invexp(x, a, b):
        return a * np.exp(- b * x)

    res, err = integrate.quad(invexp, 0, np.inf, args = (a, b))

    return res

### 20 ###
def parse_arguments():
    '''
    Parse the command line 
    arguments of a program.

    Returns an argument parser object.
    '''
    
    DEFAULT_ARG_VALUE = 'this_is_the_default_value'

    parser = argparse.ArgumentParser(description='Example parser')
    
    parser.add_argument(
        '--first_arg', # The name of the input variable
        required=True, # Is it an optional variable?
        metavar='THE_FIRST_ARG', # For errormessage printing purposes
        type=float, # The datatype of the input variable
        # A description of the variable or error message purposes
        help='the first argument in this test code is a float'
        )

    parser.add_argument(
        '--second_arg',
        required=False,
        metavar='THE_SECOND_ARG',
        type=str,
        help='the second argument in this test code is a string',
        default=DEFAULT_ARG_VALUE
        )

    return parser.parse_args()

def main():
    '''
    This is a special function name.
    Everything in your code should
    run inside main()
    '''

    ### 1 ###
    # fact = factorial(10)
    # print 'factorial', fact

    ### 2 ###
    # x_mod_z = modulus(100, 12)
    # print 'x_mod_z', x_mod_z

    ### 3 ###
    # lst2 = list_of_lists(5, 3)
    # print 'list', lst2

    ### 4 ###
    # lst = lists_of_list(5, 3)
    # print 'list', lst

    ### 5 ###
    # list1, list2 = tup_to_list([('f', 3), ('e', 9), ('j', 5), ('d', 1), ('g', 2)])
    # print list1, list2

    ### 6 ###
    # dict_tup = tuple_to_dict([('f', 3), ('e', 9), ('j', 5), ('d', 1), ('g', 2),
    #                             ('g', 3), ('g', 12), ('i', 2), ('e', 2)])
    # print 'dict_tup', dict_tup

    ### 7 ###
    # sorted_tup = sort_tups([('f', 3), ('e', 9), ('j', 5), ('d', 1), ('g', 2),
    #                             ('g', 3), ('g', 12), ('i', 2), ('e', 2)])
    # print 'sorted_tup', sorted_tup

    ### 8 ###
    # tpl = remove_empty_tuples([('f', 3), ('e', 9), ('j', 5), (), ('d', 1), ('g', 2)])
    # print 'tuple', tpl

    ### 9 ###
    # sumtest_arr = sum_arr(5, 10, xi=1.5)
    # print 'sumtest_arr', sumtest_arr

    ### 10 ###
    # plot_x_pow_n(0.,100., 3)

    ### 11 ###
    # fourxfour_plot(-100, 100, numpts=100)

    ### 12 ###
    # scatterplot_random(-50., 50., numpts=100)

    ### 13 ###
    # twodim_xsq_ysq(-50., 50., numpix=100)

    ### 14 ###
    # data1 = read_data('FL_insurance_sample.csv', 18, delimiter = ',')
    # data1 = read_data('gamma_true_gamma_obs.txt', 4, delimiter = ' ')
    # for datrow in data1.T:
    #     print datrow

    ### 14 *alternative* ###
    # data2 = read_csv('FL_insurance_sample.csv', 18, delimiter = ',')
    # data2 = read_csv('gamma_true_gamma_obs.txt', 4, delimiter = ' ')
    # print data1.shape, data2.shape
    # for datrow in data2:
        # print datrow

    ### 15 ###
    # write_data(data1.T, 'test.csv', delimiter=',')

    ### 16 ###
    # join_csvs(['data1.csv', 'data2.csv'], [4,3], [0,0], header='id')

    ### 17 ###
    # hist_plot('FL_insurance_sample.csv', 'itemhist.png', 18, 16, delimiter=',', header=True)

    ### 18 ###
    # n_instances_whole, n_instances_all, loc_whole, loc_all = \
    # count_instances_simple('FL_insurance_sample.csv', 'Coun', 18, delimiter=',', header=True)
    # print n_instances_whole, n_instances_all
    # print loc_whole, loc_all

    ### 19 ###
    # result = integrate_invexp(2., 1.)
    # print result

    ### 20 ###
    # arguments = parse_arguments()
    # print arguments.first_arg, arguments.second_arg

# This tells python to run the main() function by defualt
if __name__ == '__main__':
    main()
