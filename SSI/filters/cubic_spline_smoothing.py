from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

def cubic_spline_smoothing(input_data):
    '''
    Cubic Spline Smoothing Filter
    (Called it like this)
    p = 0.001; %smoothing parameter, [0,1]. decrease this for more smoothing
    hr_struct.avg = csaps(hr_struct.time,hr_struct.avg,p,hr_struct.time); 
    '''

    x = np.linspace(0, 10, num=11, endpoint=True)
    y = np.cos(-x**2/9.0)
    f = interp1d(x, y)
    f2 = interp1d(x, y, kind='cubic')

    xnew = np.linspace(0, 10, num=41, endpoint=True)

    print(f2(xnew))

    plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
    plt.legend(['data', 'linear', 'cubic'], loc='best')
    plt.show()

