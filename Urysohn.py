#The concept belongs to Andrew Polar and Mike Poluektov.
#details:
#https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149?via%3Dihub
#readers friendly format ezcodesample.com

import numpy as np
import math

#Next two classes represent Urysohn model
class PLL:
    def __init__(instance, xmin, xmax, points):
        instance.Initialize(xmin, xmax, points)

    def Initialize(instance, xmin, xmax, points):
        if (points < 2):
            print("Fatal: number of blocks is too low")
            exit(0)
        if (xmin >= xmax):
            xmax = xmin + 0.5
            xmin -= 0.5
               
        instance._xmin = xmin
        instance._xmax = xmax
        instance._points = points

        instance._deltax = (instance._xmax - instance._xmin) / (instance._points - 1)
        instance._y = []
        for idx in range(instance._points): 
            instance._y.append(0.0)

    def SetRandom(instance, ymin, ymax):
        for idx in range(instance._points):
            instance._y[idx] = np.random.randint(10, 1000)
        
        min = np.min(instance._y)
        max = np.max(instance._y)
        if (min == max):
            max = min + 1.0

        for idx in range(instance._points):
            instance._y[idx] = (instance._y[idx] - min) / (max - min)
            instance._y[idx] = instance._y[idx] * (ymax - ymin) + ymin

    def GetDerrivative(instance, x):
        length = len(instance._y)
        low = int((x - instance._xmin) / instance._deltax)
        if (low < 0): low = 0
        if (low > length - 2): low = length - 2
        return (instance._y[low + 1] - instance._y[low]) / instance._deltax
    
    def Update(instance, x, delta, mu):
        length = len(instance._y)
        if (x < instance._xmin):
            instance._deltax = (instance._xmax - x) / (instance._points - 1)
            instance._xmin = x
 
        if (x > instance._xmax):
            instance._deltax = (x - instance._xmin) / (instance._points - 1)
            instance._xmax = instance._xmin + (instance._points - 1) * instance._deltax

        left = int((x - instance._xmin) / instance._deltax)
        if (left < 0): left = 0
        if (left >= length - 1):
            instance._y[length - 1] += delta * mu 
            return

        leftx = x - (instance._xmin + left * instance._deltax)
        rightx = instance._xmin + (left + 1) * instance._deltax - x
        instance._y[left + 1] += delta * leftx / instance._deltax * mu
        instance._y[left] += delta * rightx / instance._deltax * mu

    def GetFunctionValue(instance, x):
        length = len(instance._y)
        if (x < instance._xmin):
            derrivative = (instance._y[1] - instance._y[0]) / instance._deltax
            return instance._y[1] - derrivative * (instance._xmin + instance._deltax - x)

        if (x > instance._xmax):
            derrivative = (instance._y[length - 1] - instance._y[length - 2]) / instance._deltax
            return instance._y[length - 2] + derrivative * (x - (instance._xmax - instance._deltax))

        left = int((x - instance._xmin) / instance._deltax)
        if (left < 0): left = 0
        if (left >= length - 1):
            return instance._y[length - 1]
        
        leftx = x - (instance._xmin + left * instance._deltax)
        return (instance._y[left + 1] - instance._y[left]) / instance._deltax * leftx + instance._y[left]        

class U:
    def __init__(instance, xmin, xmax, targetMin, targetMax, layers):
        length = len(layers)
        ymin = targetMin / length
        ymax = targetMax / length
        instance._plist = []
        for idx in range(0, length):
            pll = PLL(xmin[idx], xmax[idx], layers[idx])
            instance._plist.append(pll)
        instance.SetRandom(ymin, ymax)

    def Clear(instance):
        instance._plist.clear()

    def GetDerrivative(instance, layer, x):
        return instance._plist[layer].GetDerrivative(x)
    
    def SetRandom(instance, ymin, ymax):
        length = len(instance._plist)
        for pll in instance._plist: 
            pll.SetRandom(ymin / length, ymax / length)

    def Update(instance, delta, inputs, mu):
        i = 0
        length = len(inputs)
        for i in range(0, length):
            instance._plist[i].Update(inputs[i], delta / length, mu)
 
    def GetU(instance, inputs):
        f = 0.0
        length = len(inputs)
        for i in range(0, length):
            f += instance._plist[i].GetFunctionValue(inputs[i])
        return f

#end of description part, now we test it

#Generate training dataset
N = 1000 #number of training records
M = 100 #number of epochs
def ComputeTarget(inputs):
    return math.sin(inputs[0]) + math.cos(inputs[1]) + math.log2(inputs[2] + 1.0)

features = []
target = []
rows, cols = N, 3
for i in range(rows):
    col = []
    col.append(np.random.randint(10, 1000) / 1000.0)
    col.append(np.random.randint(10, 1000) / 1000.0)
    col.append(np.random.randint(10, 1000) / 1000.0)
    features.append(col)
    target.append(ComputeTarget(col))
#training dataset is generated

#properties of expected Urysohn operator and data
layers = [8, 8, 8]
#data properties
xmin = [0.0, 0.0, 0.0]
xmax = [1.0, 1.0, 1.0]
tmin = np.min(target)
tmax = np.max(target)
#end

#Identification process
u = U(xmin, xmax, tmin, tmax, layers)
for idm in range (0, M):
    error = 0.0
    for idx in range (0, N):
        model = u.GetU(features[idx])
        delta = target[idx] - model
        u.Update(delta, features[idx], 0.02)
        error += delta * delta
    error /= N
    error = math.sqrt(error)
    error /= (tmax - tmin)
    if (error < 0.001): break
#end

#making validation data
V = 100
Vfeatures = []
Vtarget = []
rows, cols = V, 3
for i in range(rows):
    col = []
    col.append(np.random.randint(10, 1000) / 1000.0)
    col.append(np.random.randint(10, 1000) / 1000.0)
    col.append(np.random.randint(10, 1000) / 1000.0)
    Vfeatures.append(col)
    Vtarget.append(ComputeTarget(col))
#end
    
#testing validation data
error = 0.0
for idx in range (0, V):
    model = u.GetU(Vfeatures[idx])
    delta = Vtarget[idx] - model
    error += delta * delta
error /= V
error = math.sqrt(error)
error /= (tmax - tmin)

print(f"Mean relative error for validation set = {error}")