import sys

a=float(sys.argv[1])
b=float(sys.argv[2])

r1 = a / (a+b)
r2 = b / (a+b)

print("%5.3f %5.3f"%(r1,r2))

