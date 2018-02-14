import math

T2 = 105702.0
T1 = 47525.0

M2 = 15565.0
M1 = 9361.0

b = math.log(M2/M1)/math.log(T2/T1)
print(b)

k = (M2 - M1)/(T2**b - T1**b)
print(k)