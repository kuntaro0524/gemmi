import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

file_id = 'sample_data.csv'
#file_path = '/Users/kentaro/work_space/python/'
file_path = './'

rfile = file_path + file_id
data = np.loadtxt(rfile, comments='#' ,delimiter=',')

x_csv = data[:,0]
y_csv = data[:,1]

#Least squares method with scipy.optimize
def fit_func(parameter,x,y):
    a = parameter[0]
    b = parameter[1]
    c = parameter[2]
    residual = y-(a*x**2+b*x+c)
    return residual

parameter0 = [0.,0.,0.]
result = optimize.leastsq(fit_func,parameter0,args=(x_csv,y_csv))
a_fit=result[0][0]
b_fit=result[0][1]
c_fit=result[0][2]
print(a_fit,b_fit,c_fit)

#PLot
plt.figure(figsize=(8,5))
plt.plot(x_csv,y_csv,'bo', label='Exp.')
plt.plot(x_csv,a_fit*x_csv**2+b_fit*x_csv+c_fit,'k-', label='fitted parabora', linewidth=10, alpha=0.3)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()