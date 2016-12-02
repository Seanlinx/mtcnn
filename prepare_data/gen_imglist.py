import numpy as np
import numpy.random as npr

size = 12

if size == 12:
    net = "pnet"
elif size == 24:
    net = "rnet"
elif size == 48:
    net = "onet"

with open('%s/pos_%s.txt'%(net, size), 'r') as f:
    pos = f.readlines()

with open('%s/neg_%s.txt'%(net, size), 'r') as f:
    neg = f.readlines()

with open('%s/part_%s.txt'%(net, size), 'r') as f:
    part = f.readlines()


with open("%s/train_%s.txt"%(net, size), "w") as f:
    f.writelines(pos)
    neg_keep = npr.choice(len(neg), size=600000, replace=False)
    part_keep = npr.choice(len(part), size=300000, replace=False)
    for i in neg_keep:
        f.write(neg[i])
    for i in part_keep:
        f.write(part[i])
