
LTspiceExec = "D:\\DATA\\Software\\LTspice\\LTspice.exe"
# 设计值
my_Length_nm = [720 , 720 , 720  , 720  , 720 , 720  , 720 , 720  , 720 , 540 , 540  , 720 , 720 , 720, 720, 720 , 720 , 720, 720]
my_Width_nm  = [3600, 3600, 10800, 10800, 2880, 42350, 5500, 80000, 5200, 1683, 20197, 1683, 1683, 720, 720, 5120, 5120, 720, 720]
my_R0 = 30500
my_Cc = 1.3E-12


# 变量上下界    x0     x1     x2     x3      x4     x5      x6     x7    x8     x9    x10
lower_bound = [1000 , 5000 , 720  , 10000 , 1000 , 3000,   3000,  720,  5000,  720,  200*1E-15]
upper_bound = [10000, 50000, 10000, 100000, 50000, 100000, 10000, 3000, 50000, 3000, 5000*1E-15]

def CircuitParameter_to_Variable(a_Width_nm=my_Width_nm, R=my_R0, C=my_Cc):
    x = []
    x.append(a_Width_nm[0])
    x.append(a_Width_nm[2])
    x.append(a_Width_nm[4])
    x.append(a_Width_nm[5])
    x.append(a_Width_nm[6])
    x.append(a_Width_nm[7])
    x.append(a_Width_nm[8])
    x.append(a_Width_nm[13])
    x.append(a_Width_nm[15])
    x.append(a_Width_nm[18])
    x.append(C)
    return x
    

def Variable_to_CircuitParameter(Variable):
    a_Width_nm = my_Width_nm.copy()
    R = my_R0
    Cc = my_Cc
    a_Width_nm[0] = a_Width_nm[1] = Variable[0]
    a_Width_nm[2] = a_Width_nm[3] = Variable[1]
    a_Width_nm[4] = Variable[2]
    a_Width_nm[5] = Variable[3]
    a_Width_nm[6] = Variable[4]
    a_Width_nm[7] = Variable[5]
    a_Width_nm[8] = Variable[6]
    a_Width_nm[13] = a_Width_nm[14] = a_Width_nm[17] = Variable[7]
    a_Width_nm[15] = a_Width_nm[16] = Variable[8]
    a_Width_nm[18] = Variable[9]
    Cc = Variable[10]
    return a_Width_nm, R, Cc
