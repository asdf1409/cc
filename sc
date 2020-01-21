1A Design a simple linear neural network model

print("Name : Kanagaraj  Roll No : 01")
print("Enter the Input X=")
x=float(input());
print("Enter the Bias : ")
b=float(input());
print("Enter the Weight : ")
w=float(input());
net=(w*x+b);
print("******OUTPUT******")
print("Net. Weight = ",net)
if net < 0:
    out=0;
elif(net>=0 and net<=1):
    out=net;
else:
    out=1;
print("Output : ",out)
=========================================================================================
1B output of neural network using both binary and bipolar sigmoidal function

print("Name : Pravin   Roll No : 11")
n=int(input("Enter number of elements : "))
print("Enter the inputs : ")
inputs=[] #creating and emplty list
for i in range(0,n):
    ele=float(input())
    inputs.append(ele) #adding the element
print(inputs)
print("Enter the Weights : ")
weights=[] #creating and emplty list
for i in range(0,n):
    ele=float(input())
    weights.append(ele) #adding the element
print(weights)
print("The net input can be calculated as Yin = x1w1+x2w2+x3w3")
Yin=[]
for i in range(0,n):
    Yin.append(inputs[i]*weights[i])
print(round(sum(Yin),3))
=====================================================================================================
2A Generate ANDNOT function using McCulloch-Pitts neural net

import numpy
# enter the no of inputs
num_ip = int(input("Enter the number of inputs : "))
#Set the weights with value 1
w1 = 1
w2 = 1
print("For the ", num_ip , " inputs calculate the net input using yin = x1w1 + x2w2 ")
x1 = []
x2 = []
for j in range(0, num_ip):
    ele1 = int(input("x1 = "))
    ele2 = int(input("x2 = "))
    x1.append(ele1)
    x2.append(ele2)
print("x1 = ",x1)
print("x2 = ",x2)
n = x1 * w1
m = x2 * w2
Yin = []
for i in range(0, num_ip):
	Yin.append(n[i] + m[i])
print("Yin = ",Yin)
==============================================================================================================
2B Generate XOR function using McCulloch-Pitts neural net
print("XOR function using Mc-Culloch Pitts neuron")	
print()
print("Enter 4 binary inputs.");
x1inputs=[]	
x2inputs=[]
c=input("Press 1 to enter inputs or Enter to use default inputs.")
if(c=="1"):
    for i in range(0,4):
            x1=int(input("Enter x1 : "))	
            x1inputs.append(x1)       
            x2=int(input("Enter x2 : "))	
            x2inputs.append(x2)        
else:
    x1inputs=[1,1,0,0]
    x2inputs=[1,0,1,0]

print("Calculating z1 = x1 x2'")
print("Considering one weight as excitatory and other as inhibitory.");
w1 = [1,1,1,1]	
w2 = [-1,-1,-1,-1]	
z1=[]          
for i in range(0,4):    
    z1.append(x1inputs[i]*w1[i] + x2inputs[i]*w2[i])
print("x1 " , "x2 " , "z1")
for i in range(0,4):    
    print(x1inputs[i] ," ",  x2inputs[i]," " ,  z1[i])
print("Calculating z2 = x1' x2")
print("Considering one weight as excitatory and other as inhibitory.");
w1 = [-1,-1,-1,-1]	
w2 = [1,1,1,1]	
z2=[]          
for i in range(0,4):    
    z2.append(x1inputs[i]*w1[i] + x2inputs[i]*w2[i])
print("x1 " , "x2 " , "z2")
for i in range(0,4):    
    print(x1inputs[i] ," " ,  x2inputs[i] ," ", z2[i])
print("Applying Threshold=1 for z1 and z2")
for i in range(0,4):
    if(z1[i]>=1):        
        z1[i]=1
    else:        
        z1[i]=0
if(z2[i]>=1):        
    z2[i]=1
else:        
    z2[i]=0
print("z1 ","z2")
for i in range(0,4):    
    print(z1[i] ," ",  z2[i])
y = []	
v1=1	
v2=1
for i in range(0,4):
    y.append( z1[i]*v1 + z2[i]*v2 )
print("x1" , "x2" , " y")
for i in range(0,4):    
    print(x1inputs[i] ," " ,  x2inputs[i] ," ", y[i])
====================================================================================================================
3A implement Hebb’s rule
import numpy as np
#first pattern 
x1=np.array([1,1,1,-1,1,-1,1,1,1]) 
#second pattern 
x2=np.array([1,1,1,1,-1,1,1,1,1]) 
#initialize bais value
b=0 
#define target 
y=np.array([1,-1]) 
wtold=np.zeros((9,)) 
wtnew=np.zeros((9,))
wtnew=wtnew.astype(int) 
wtold=wtold.astype(int) 
bais=0
print("First input with target =1")
for i in range(0,9): 
    wtold[i]=wtold[i]+x1[i]*y[0]
wtnew=wtold
b=b+y[0]
print("new wt =", wtnew) 
print("Bias value",b) 
print("Second input with target =-1")
for i in range(0,9): 
    wtnew[i]=wtold[i]+x2[i]*y[1] 
b=b+y[1]
print("new wt =", wtnew) 
print("Bias value",b) 
============================================================================================================================
3B implement of delta rule
#supervised learning
import numpy as np
import time
np.set_printoptions(precision=2)
x=np.zeros((3,))
weights=np.zeros((3,))
desired=np.zeros((3,)) 
actual=np.zeros((3,))  
for i in range(0,3):
    x[i]=float(input("Initial inputs:")) 
for i in range(0,3):
    weights[i]=float(input("Initial weights:")) 
for i in range(0,3):
    desired[i]=float(input("Desired output:")) 
a=float(input("Enter learning rate:")) 
actual=x*weights
print("actual",actual) 
print("desired",desired) 
while True:
    if np.array_equal(desired,actual):
        break
    #no change
    else:
        for i in range(0,3): 
            weights[i]=weights[i]+a*(desired[i]-actual[i])  
    actual=x*weights
    print("weights",weights)
    print("actual",actual)
    print("desired",desired)
    print("*"*30)
    print("Final output")
    print("Corrected weights",weights)
    print("actual",actual) 
print("desired",desired) 
===========================================================================================================
4A Back Propagation Algorithm
import numpy as np
import decimal
import math 
np.set_printoptions(precision=2)
v1=np.array([0.6, 0.3])
v2=np.array([-0.1, 0.4])
w=np.array([-0.2,0.4,0.1]) 
b1=0.3
b2=0.5
x1=0
x2=1
alpha=0.25 
print("calculate net input to z1 layer")
zin1=round(b1+ x1*v1[0]+x2*v2[0],4) 
print("z1=",round(zin1,3)) 
print("calculate net input to z2 layer")
zin2=round(b2+ x1*v1[1]+x2*v2[1],4) 
print("z2=",round(zin2,4)) 
print("Apply activation function to calculate output") 
z1=1/(1+math.exp(-zin1))
z1=round(z1,4)
z2=1/(1+math.exp(-zin2))
z2=round(z2,4)
print("z1=",z1)
print("z2=",z2)
print("calculate net input to output layer") 
yin=w[0]+z1*w[1]+z2*w[2]
print("yin=",yin) 
print("calculate net output")
y=1/(1+math.exp(-yin))
print("y=",y) 
fyin=y *(1- y)
dk=(1-y)*fyin 
print("dk",dk) 
dw1= alpha * dk * z1
dw2= alpha * dk * z2 
dw0= alpha * dk
print("compute error portion in delta") 
din1=dk* w[1]
din2=dk* w[2]
print("din1=",din1)
print("din2=",din2) 
print("error in delta")
fzin1= z1 *(1-z1)
print("fzin1",fzin1)
d1=din1* fzin1
fzin2= z2 *(1-z2)
print("fzin2",fzin2) 
d2=din2* fzin2 
print("d1=",d1)
print("d2=",d2) 
print("Changes in weights between input and hidden layer")
dv11=alpha * d1 * x1
print("dv11=",dv11)
dv21=alpha * d1 * x2
print("dv21=",dv21)
dv01=alpha * d1
print("dv01=",dv01)
dv12=alpha * d2 * x1
print("dv12=",dv12)
dv22=alpha * d2 * x2
print("dv22=",dv22)
dv02=alpha * d2
print("dv02=",dv02)
print("Final weights of network")
v1[0]=v1[0]+dv11
v1[1]=v1[1]+dv12
print("v=",v1)
v2[0]=v2[0]+dv21
v2[1]=v2[1]+dv22
print("v2",v2)
w[1]=w[1]+dw1
w[2]=w[2]+dw2
b1=b1+dv01
b2=b2+dv02
w[0]=w[0]+dw0
print("w=",w) 
print("bias b1=",b1, " b2=",b2)
============================================================================================================
4B Error Back Propagation Algorithm

import math
a0=-1
t=-1
w10=float(input("Enter weight first network: "))
b10=float(input("Enter base first network:"))
w20=float(input("Enter weight second network:"))
b20=float(input("Enter base second network:")) 
c=float(input("Enter learning coefficient:")) 
n1=float(w10*c+b10)
a1=math.tanh(n1)
n2=float(w20*a1+b20)
a2=math.tanh(float(n2))
e=t-a2
s2=-2*(1-a2*a2)*e 
s1=(1-a1*a1)*w20*s2 
w21=w20-(c*s2*a1)
w11=w10-(c*s1*a0)
b21=b20-(c*s2)
b11=b10-(c*s1)
print("The updated weight of first n/w w11=",w11)
print("The uploaded weight of second n/w w21= ",w21)
print("The updated base of first n/w b10=",b10) 
print("The updated base of second n/w b20= ",b20) 
=======================================================================================================
Linear Seperation 

import numpy as np
import matplotlib.pyplot as plt 
def create_distance_function(a, b, c):
    """ 0 = ax + by + c """
    def distance(x, y):
        """ returns tuple (d, pos)
            d is the distance
            If pos == -1 point is below the line, 
            0 on the line and +1 if above the line
        """
        nom = a * x + b * y + c
        if nom == 0:
            pos = 0
        elif (nom<0 and b<0) or (nom>0 and b>0):
            pos = -1
        else:
            pos = 1
        return (np.absolute(nom) / np.sqrt( a ** 2 + b ** 2), pos)
    return distance
    
points = [ (3.5, 1.8), (1.1, 3.9) ]
fig, ax = plt.subplots()
ax.set_xlabel("sweetness")
ax.set_ylabel("sourness")
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 8])
X = np.arange(-0.5, 5, 0.1)
colors = ["r", ""] # for the samples
size = 10
for (index, (x, y)) in enumerate(points):
    if index== 0:
        ax.plot(x, y, "o", 
                color="darkorange", 
                markersize=size)
    else:
        ax.plot(x, y, "oy", 
                markersize=size)
step = 0.05
for x in np.arange(0, 1+step, step):
    slope = np.tan(np.arccos(x))
    dist4line1 = create_distance_function(slope, -1, 0)
    #print("x: ", x, "slope: ", slope)
    Y = slope * X
    
    results = []
    for point in points:
        results.append(dist4line1(*point))
    #print(slope, results)
    if (results[0][1] != results[1][1]):
        ax.plot(X, Y, "g-")
    else:
        ax.plot(X, Y, "r-")
        
plt.show()
=-========================================================================================================

A) Aim: - Write a program for Hopfield Network.
Code: - 
pattern_size = 5
hopfield_net = network.HopfieldNetwork(nr_neurons= pattern_size**2)
factory = pattern_tools.PatternFactory(pattern_size, pattern_size)
checkerboard = factory.create_checkerboard()
pattern_list = [checkerboard]
pattern_list.extend(factory.create_random_pattern_list(nr_patterns=3, on_probability=0.5))
plot_tools.plot_pattern_list(pattern_list)
overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
plot_tools.plot_overlap_matrix(overlap_matrix)
hopfield_net.store_patterns(pattern_list)
noisy_init_state = pattern_tools.flip_n(checkerboard, nr_of_flips=4)
hopfield_net.set_state_from_pattern(noisy_init_state)
states = hopfield_net.run_with_monitoring(nr_steps=4)
states_as_patterns = factory.reshape_patterns(states)
plot_tools.plot_state_sequence_and_overlap(states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")
------------------------------------------------------------------------------------
B) Aim: - Write a program for Radial Basis function.
Code: -
from scipy import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt
 
class RBF:
     
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))
         
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
         
        print ("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        print (G)
         
        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
         
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y
 
if __name__ == '__main__':
    # ----- 1D Example ------------------------------------------------
    n = 100
     
    x = mgrid[-1:1:complex(0,n)].reshape(n, 1)
    # set y and add random noise
    y = sin(3*(x+0.5)**3 - 1)
    # y += random.normal(0, 0.1, y.shape)
     
    # rbf regression
    rbf = RBF(1, 10, 1)
    rbf.train(x, y)
    z = rbf.test(x)
       
    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')
     
    # plot learned model
    plt.plot(x, z, 'r-', linewidth=2)
     
    # plot rbfs
    plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')
     
    for c in rbf.centers:
        # RF prediction lines
        cx = arange(c-0.7, c+0.7, 0.01)
        cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)
     
    plt.xlim(-1.2, 1.2)
    plt.show()
-------------------------------------------------------------------------------------
A) Aim: - Implement Kohonen Self organizing map.
Code: -
import numpy as np
import matplotlib.pyplot as plt
def closest_node(data, t, map, m_rows, m_cols):
    result = (0,0)
    small_dist = 1.0e20
    for i in range(m_rows):
        for j in range(m_cols):
            ed = euc_dist(map[i][j], data[t])
            if ed < small_dist:
                small_dist = ed
            result = (i, j)
    return result
def euc_dist(v1, v2):
    return np.linalg.norm(v1 - v2)
def manhattan_dist(r1, c1, r2, c2):
    return np.abs(r1-r2) + np.abs(c1-c2)
def most_common(lst, n):
    if len(lst) == 0: return -1
    counts = np.zeros(shape=n, dtype=np.int)
    for i in range(len(lst)):
        counts[lst[i]] += 1
    return np.argmax(counts)
def main():
    np.random.seed(1)
    Dim = 4
    Rows = 30; Cols = 30
    RangeMax = Rows + Cols
    LearnMax = 0.5
    StepsMax = 5000
    print("\nLoading Iris data into memory \n")
    data_file = "C:/Users/exam/Documents/iris_data_012.txt"
    data_x = np.loadtxt(data_file, delimiter=",", usecols=range(0,4),dtype=np.float64)
    data_y = np.loadtxt(data_file, delimiter=",", usecols=[4],dtype=np.int)
    print("Constructing a 30x30 SOM from the iris data")
    map = np.random.random_sample(size=(Rows,Cols,Dim))
    for s in range(StepsMax):
        if s % (StepsMax/10) == 0: print("step = ", str(s))
        pct_left = 1.0 - ((s * 1.0) / StepsMax)
        curr_range = (int)(pct_left * RangeMax)
        curr_rate = pct_left * LearnMax
        t = np.random.randint(len(data_x))
        (bmu_row, bmu_col) = closest_node(data_x, t, map, Rows, Cols)
        for i in range(Rows):
            for j in range(Cols):
                if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
                    map[i][j] = map[i][j] + curr_rate * \
                    (data_x[t] - map[i][j])
    print("SOM construction complete \n")
    print("Constructing U-Matrix from SOM")
    u_matrix = np.zeros(shape=(Rows,Cols), dtype=np.float64)
    for i in range(Rows):
        for j in range(Cols):
            v = map[i][j] # a vector
            sum_dists = 0.0; ct = 0
            if i-1 >= 0: # above
                sum_dists += euc_dist(v, map[i-1][j]); ct += 1
            if i+1 <= Rows-1: # below
                sum_dists += euc_dist(v, map[i+1][j]); ct += 1
            if j-1 >= 0: # left
                sum_dists += euc_dist(v, map[i][j-1]); ct += 1
            if j+1 <= Cols-1: # right
                sum_dists += euc_dist(v, map[i][j+1]); ct += 1
            u_matrix[i][j] = sum_dists / ct
    print("U-Matrix constructed \n")
    plt.imshow(u_matrix, cmap='gray') # black = close = clusters
    plt.show()
    print("Associating each data label to one map node ")
    mapping = np.empty(shape=(Rows,Cols), dtype=object)
    for i in range(Rows):
        for j in range(Cols):
            mapping[i][j] = []
    for t in range(len(data_x)):
        (m_row, m_col) = closest_node(data_x, t, map, Rows, Cols)
        mapping[m_row][m_col].append(data_y[t])
    label_map = np.zeros(shape=(Rows,Cols), dtype=np.int)
    for i in range(Rows):
        for j in range(Cols):
            label_map[i][j] = most_common(mapping[i][j], 3)
    plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4))
    plt.colorbar()
    plt.show()
if __name__=="__main__":
    main()
------------------------------------------------------------------------------------
B) Aim: - Implement Adaptive resonance theory.
Code: -
from __future__ import print_function
from __future__ import division
import numpy as np

class ART:
    def __init__(self, n=5, m=10, rho=.5):
        
        self.F1 = np.ones(n)
        self.F2 = np.ones(m)
        self.Wf = np.random.random((m,n))
        self.Wb = np.random.random((n,m))
        self.rho = rho
        self.active = 0

    def learn(self, X):
        self.F2[...] = np.dot(self.Wf, X)
        I = np.argsort(self.F2[:self.active].ravel())[::-1]

        for i in I:
            
            d = (self.Wb[:,i]*X).sum()/X.sum()
            if d >= self.rho:
                
                self.Wb[:,i] *= X
                self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
                return self.Wb[:,i], i

        if self.active < self.F2.size:
            i = self.active
            self.Wb[:,i] *= X
            self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
            self.active += 1
            return self.Wb[:,i], i

        return None,None
if __name__ == '__main__':

    np.random.seed(1)
    network = ART( 5, 10, rho=0.5)
    data = ["   O ",
            "  O O",
            "    O",
            "  O O",
            "    O",
            "  O O",
            "    O",
            " OO O",
            " OO  ",
            " OO O",
            " OO  ",
            "OOO  ",
            "OO   ",
            "O    ",
            "OO   ",
            "OOO  ",
            "OOOO ",
            "OOOOO",
            "O    ",
            " O   ",
            "  O  ",
            "   O ",
            "    O",
            "  O O",
            " OO O",
            " OO  ",
            "OOO  ",
            "OO   ",
            "OOOO ",
            "OOOOO"]
    X = np.zeros(len(data[0]))
    for i in range(len(data)):
        for j in range(len(data[i])):
            X[j] = (data[i][j] == 'O')
        Z, k = network.learn(X)
        print("|%s|"%data[i],"-> class", k)
-------------------------------------------------------------------------------------
A) Aim: - Write a program for Linear separation.
Code: -
import numpy as np
import matplotlib.pyplot as plt
def create_distance_function(a, b, c):
    """ 0 = ax + by + c """
    def distance(x, y):
        """ returns tuple (d, pos)
            d is the distance
            If pos == -1 point is below the line, 
            0 on the line and +1 if above the line
        """
        nom = a * x + b * y + c
        if nom == 0:
            pos = 0
        elif (nom<0 and b<0) or (nom>0 and b>0):
            pos = -1
        else:
            pos = 1
        return (np.absolute(nom) / np.sqrt( a ** 2 + b ** 2), pos)
    return distance

points = [ (3.5, 1.8), (1.1, 3.9) ]
fig, ax = plt.subplots()
ax.set_xlabel("sweetness")
ax.set_ylabel("sourness")
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 8])
X = np.arange(-0.5, 5, 0.1)
index =10 
for (index, (x, y)) in enumerate(points):
        ax.plot(x, y, "o", color="darkorange")
    
step = 0.05

for x in np.arange(0, 1+step, step):
    slope = np.tan(np.arccos(x))
    dist4line1 = create_distance_function(slope, -1, 0)
    #print("x: ", x, "slope: ", slope)
    Y = slope * X

    results = []
    for point in points:
        results.append(dist4line1(*point))
    #print(slope, results)
    if (results[0][1] != results[1][1]):
        ax.plot(X, Y, "g-")
    else:
        ax.plot(X, Y, "r-")

plt.show()
----------------------------------------------------------------------------------
B) Aim: - Write a program for Hopfield network model for associative memory.
Code: -
import numpy as np
import matplotlib.pyplot as plt
# note: if this fails, try >pip uninstall matplotlib
# and then >pip install matplotlib
def closest_node(data, t, map, m_rows, m_cols):
    result = (0,0)
    small_dist = 1.0e20
    for i in range(m_rows):
        for j in range(m_cols):
            ed = euc_dist(map[i][j], data[t])
            if ed < small_dist:
                small_dist = ed
            result = (i, j)
    return result
def euc_dist(v1, v2):
    return np.linalg.norm(v1 - v2)
def manhattan_dist(r1, c1, r2, c2):
    return np.abs(r1-r2) + np.abs(c1-c2)
def most_common(lst, n):
# lst is a list of values 0 . . n
    if len(lst) == 0: return -1
    counts = np.zeros(shape=n, dtype=np.int)
    for i in range(len(lst)):
        counts[lst[i]] += 1
    return np.argmax(counts)
# ==================================================================
def main():
# 0. get started
    np.random.seed(1)
    Dim = 4
    Rows = 30; Cols = 30
    RangeMax = Rows + Cols
    LearnMax = 0.5
    StepsMax = 5000
# 1. load data
    print("\nLoading Iris data into memory \n")
    data_file = "C:/Users/exam/Documents/iris_data_012.txt"
    data_x = np.loadtxt(data_file, delimiter=",", usecols=range(0,4),dtype=np.float64)
    data_y = np.loadtxt(data_file, delimiter=",", usecols=[4],dtype=np.int)
# option: normalize data
# 2. construct the SOM
    print("Constructing a 30x30 SOM from the iris data")
    map = np.random.random_sample(size=(Rows,Cols,Dim))
    for s in range(StepsMax):
        if s % (StepsMax/10) == 0: print("step = ", str(s))
        pct_left = 1.0 - ((s * 1.0) / StepsMax)
        curr_range = (int)(pct_left * RangeMax)
        curr_rate = pct_left * LearnMax
        t = np.random.randint(len(data_x))
        (bmu_row, bmu_col) = closest_node(data_x, t, map, Rows, Cols)
        for i in range(Rows):
            for j in range(Cols):
                if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
                    map[i][j] = map[i][j] + curr_rate * \
                    (data_x[t] - map[i][j])
    print("SOM construction complete \n")
# 3. construct U-Matrix
    print("Constructing U-Matrix from SOM")
    u_matrix = np.zeros(shape=(Rows,Cols), dtype=np.float64)
    for i in range(Rows):
        for j in range(Cols):
            v = map[i][j] # a vector
            sum_dists = 0.0; ct = 0
            if i-1 >= 0: # above
                sum_dists += euc_dist(v, map[i-1][j]); ct += 1
            if i+1 <= Rows-1: # below
                sum_dists += euc_dist(v, map[i+1][j]); ct += 1
            if j-1 >= 0: # left
                sum_dists += euc_dist(v, map[i][j-1]); ct += 1
            if j+1 <= Cols-1: # right
                sum_dists += euc_dist(v, map[i][j+1]); ct += 1
            u_matrix[i][j] = sum_dists / ct
    print("U-Matrix constructed \n")
# display U-Matrix
    plt.imshow(u_matrix, cmap='gray') # black = close = clusters
    plt.show()
#4. because the data has labels, another possible visualization:
# associate each data label with a map node
    print("Associating each data label to one map node ")
    mapping = np.empty(shape=(Rows,Cols), dtype=object)
    for i in range(Rows):
        for j in range(Cols):
            mapping[i][j] = []
    for t in range(len(data_x)):
        (m_row, m_col) = closest_node(data_x, t, map, Rows, Cols)
        mapping[m_row][m_col].append(data_y[t])
    label_map = np.zeros(shape=(Rows,Cols), dtype=np.int)
    for i in range(Rows):
        for j in range(Cols):
            label_map[i][j] = most_common(mapping[i][j], 3)
    plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4))
    plt.colorbar()
    plt.show()
if __name__=="__main__":
    main()
----------------------------------------------------------------------------------
A) Aim: - Implement Membership and Identity Operators | in, not in.
Code: -
def overlapping(list1, list2):
    c=0
    d=0
    for i in list1:
        c+=1
    for i in list2:
        d+=1
    for i in range (0,c):
            for j in range(0,d):
                if(list1[i]==list2[j]):
                    return 1
    return 0

list1=[1,2,3,4,5]
list2=[4,6,7,8,9]
list3=[6,7,8,9,10]
if(overlapping(list1,list2)):
    print("overlapping")
else:
    print("not overlapping")
if(overlapping(list1,list3)):
    print("overlapping")
else:
    print("not overlapping")

-------------------------------------------------------------------------------
B) Aim: - Implement Membership and Identity Operators is, is not.
Code: -
x=5.2
if(type(x)is int):
    print("true")
else:
    print("false")
-----------------------------------------------------------------------------
Practical No: - 9
A) Aim: - Find ratios using fuzzy logic
Code: -
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process 

s1 = "I love fuzzysforfuzzys"
s2 = "I am loving fuzzysforfuzzys"
print ("FuzzyWuzzy Ratio:", fuzz.ratio(s1, s2)) 
print ("FuzzyWuzzy PartialRatio: ", fuzz.partial_ratio(s1, s2)) 
print ("FuzzyWuzzy TokenSortRatio: ", fuzz.token_sort_ratio(s1, s2)) 
print ("FuzzyWuzzy TokenSetRatio: ", fuzz.token_set_ratio(s1, s2)) 
print ("FuzzyWuzzy WRatio: ", fuzz.WRatio(s1, s2),'\n\n')

# for process library, 
query = 'fuzzys for fuzzys'
choices = ['fuzzy for fuzzy', 'fuzzy fuzzy', 'g. for fuzzys'] 
print ("List of ratios: ")
print (process.extract(query, choices), '\n')
print ("Best among the above list: ",process.extractOne(query, choices))

-------------------------------------------------------------------------------------
B) Aim: - Solve Tipping problem using fuzzy logic.
Code: -
import skfuzzy as fuzz
from skfuzzy import control as ctrl

quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

quality.automf(3)
service.automf(3)

tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

quality['average'].view()
service.view()
tip.view()

rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

rule1.view()

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8

tipping.compute()
print (tipping.output['tip'])
tip.view(sim=tipping)
------------------------------------------------------------------------------------
Practical No: - 10
A) Aim: - Implementation of Simple genetic algorithm.
Code: -
import random 
  
# Number of individuals in each generation 
POPULATION_SIZE = 100
  
# Valid genes 
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP 
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''
  
# Target string to be generated 
TARGET = "I love GeeksforGeeks"
  
class Individual(object): 
    ''' 
    Class representing individual in population 
    '''
    def __init__(self, chromosome): 
        self.chromosome = chromosome  
        self.fitness = self.cal_fitness() 
  
    @classmethod
    def mutated_genes(self): 
        ''' 
        create random genes for mutation 
        '''
        global GENES 
        gene = random.choice(GENES) 
        return gene 
  
    @classmethod
    def create_gnome(self): 
        ''' 
        create chromosome or string of genes 
        '''
        global TARGET 
        gnome_len = len(TARGET) 
        return [self.mutated_genes() for _ in range(gnome_len)] 
  
    def mate(self, par2): 
        ''' 
        Perform mating and produce new offspring 
        '''
  
        # chromosome for offspring 
        child_chromosome = [] 
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):     
  
            # random probability   
            prob = random.random() 
  
            # if prob is less than 0.45, insert gene 
            # from parent 1  
            if prob < 0.45: 
                child_chromosome.append(gp1) 
  
            # if prob is between 0.45 and 0.90, insert 
            # gene from parent 2 
            elif prob < 0.90: 
                child_chromosome.append(gp2) 
  
            # otherwise insert random gene(mutate),  
            # for maintaining diversity 
            else: 
                child_chromosome.append(self.mutated_genes()) 
  
        # create new Individual(offspring) using  
        # generated chromosome for offspring 
        return Individual(child_chromosome) 
  
    def cal_fitness(self): 
        ''' 
        Calculate fittness score, it is the number of 
        characters in string which differ from target 
        string. 
        '''
        global TARGET 
        fitness = 0
        for gs, gt in zip(self.chromosome, TARGET): 
            if gs != gt: fitness+= 1
        return fitness 
  
# Driver code 
def main(): 
    global POPULATION_SIZE 
  
    #current generation 
    generation = 1
  
    found = False
    population = [] 
  
    # create initial population 
    for _ in range(POPULATION_SIZE): 
                gnome = Individual.create_gnome() 
                population.append(Individual(gnome)) 
  
    while not found: 
  
        # sort the population in increasing order of fitness score 
        population = sorted(population, key = lambda x:x.fitness) 
  
        # if the individual having lowest fitness score ie.  
        # 0 then we know that we have reached to the target 
        # and break the loop 
        if population[0].fitness <= 0: 
            found = True
            break
  
        # Otherwise generate new offsprings for new generation 
        new_generation = [] 
  
        # Perform Elitism, that mean 10% of fittest population 
        # goes to the next generation 
        s = int((10*POPULATION_SIZE)/100) 
        new_generation.extend(population[:s]) 
  
        # From 50% of fittest population, Individuals  
        # will mate to produce offspring 
        s = int((90*POPULATION_SIZE)/100) 
        for _ in range(s): 
            parent1 = random.choice(population[:50]) 
            parent2 = random.choice(population[:50]) 
            child = parent1.mate(parent2) 
            new_generation.append(child) 
  
        population = new_generation 
  
        print("Generation: {}\tString: {}\tFitness: {}".\
              format(generation, 
              "".join(population[0].chromosome), 
              population[0].fitness)) 
  
        generation += 1
  
      
    print("Generation: {}\tString: {}\tFitness: {}".\
          format(generation, 
          "".join(population[0].chromosome), 
          population[0].fitness)) 
  
if __name__ == '__main__': 
    main()
Output
-----------------------------------------------------------------------------------
B) Aim: - Create two classes: City and Fitness using Genetic algorithm.
Code: -
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from tkinter import Tk, Canvas, Frame, BOTH, Text
import math
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

def main():
    cityList = []

    for i in range(0,25):
        cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
    geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
main()
------------------------------------------------------------------------------------























