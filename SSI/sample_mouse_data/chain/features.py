

import math
import numpy

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def var(numbers, mean):

    var = 0
    for i in range(len(numbers)):
        var = var + (numbers[i] - mean)**2

    var = var / len(numbers)
    return var

def calc_feature_from_sample(sample, opts):

    ret = []
    features = opts['features']
    listX, listY = zip(*sample)

    if features['min']:
        ret.append(min(listX))
        ret.append(min(listY))

    if features['max']:
        ret.append(max(listX))
        ret.append(max(listY))

    if features['dist']:
        distance = 0
        for i in range((sample.__len__()) - 1):
            distance = distance + math.sqrt((listX[i+1] - listX[i])**2 + (listY[i+1] - listY[i])**2)

        ret.append(distance)

    if features['avg']:
        ret.append(mean(listX))
        ret.append(mean(listY))

    if features['var']:
        distanceVector = numpy.zeros(sample.__len__()-1)
        for i in range((sample.__len__()) - 1):
            distanceVector[i] = math.sqrt((listX[i+1] - listX[i])**2 + (listY[i+1] - listY[i])**2)
        ret.append(var(distanceVector, mean(distanceVector)))

    if features['std']:
        distanceVector = numpy.zeros(sample.__len__()-1)
        for i in range((sample.__len__()) - 1):
            distanceVector[i] = math.sqrt((listX[i+1] - listX[i])**2 + (listY[i+1] - listY[i])**2)

        ret.append(math.sqrt(var(distanceVector, mean(distanceVector))))

    return ret



if __name__ == '__main__':
    x = [0,1,0,0,0]
    y = [0,0,0,0,0]

    opts = {
        'features' : {
        'min': True,
        'max' : True,
        'avg' : True,
        'dist' : True,
        'var' : True,
        'std' : True
        }
    }

    samples = list(zip(x,y))

    print(samples)
    print(calc_feature_from_sample(samples, opts))
    print('stop')



