__author__ = 'deepak'

base13Dict = dict((key,str(val)) for (key,val) in zip(range(10),range(10)))
base13Dict[10] = 'A'
base13Dict[11] = 'B'
base13Dict[12] = 'C'

def base13rep(tup):
    return base13Dict[tup[0]]+base13Dict[tup[1]]+base13Dict[tup[2]]

print base13rep((12,0,1))