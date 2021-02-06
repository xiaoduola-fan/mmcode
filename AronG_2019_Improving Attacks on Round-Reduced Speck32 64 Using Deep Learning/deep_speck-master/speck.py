import numpy as np
from os import urandom

word_size = 16
alpha = 7
beta = 2

MASK_VAL = 2 ** word_size - 1
#循环左移
def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (word_size - k)))
#循环右移
def ror(x,k):
    return((x >> k) | ((x << (word_size - k)) & MASK_VAL))
#***********字符串拼接实现循环左右移*************改进
def ro(x,k,wise='left'):
    f = format(x,'0{}b'.format(word_size))
    if wise in ['r','R','right','Right']:
        k = -k
    return int(f[k:]+f[:k],2)
#一轮加密
def enc_one_round1(p, k):
    c0, c1 = p
    c0 = ror(c0, alpha)
    c0 = (c0 + c1) & MASK_VAL#限制范围
    c0 = c0 ^ k
    c1 = rol(c1, beta)
    c1 = c1 ^ c0
    return(c0,c1)
#一轮加密---------------更新版
def enc_one_round(p, k):
    c0, c1 = p
    c0 = (ro(c0, alpha,'right') + c1)& MASK_VAL^k
    c1 = ro(c1, beta) ^ c0#左移
    return c0,c1 
#一轮解密
def dec_one_round(c,k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, beta)
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, alpha)
    return(c0, c1)
#一轮解密-------------------更新版
def dec_one_round1(c,k):
    c0, c1 = c
    c1 = ro(c1^c0, beta，'right')#右移
    c0 = ro((c0^k - c1) & MASK_VAL, alpha)#左移
    return c0, c1
#扩展密钥
def expand_key1(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k)-1]
    l = list(reversed(k[:len(k)-1]))
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    return(ks)
#扩展密钥--------------------更新版
def expand_key(k, t):
    ks = [k[-1]]+[0]*(t-1)
    l = list(reversed(k[:-1]))
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    return(ks)
def encrypt(p, ks):
    x, y = p
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return(x, y)

def decrypt(c, ks):
    x, y = c
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)

def check_testvector():
  key = (0x1918,0x1110,0x0908,0x0100)
  pt = (0x6574, 0x694c)
  ks = expand_key(key, 22)
  ct = encrypt(pt, ks)
  if (ct == (0xa868, 0x42f2)):
    print("Testvector verified.")
    return(True)
  else:
    print("Testvector not verified.")
    return(False)

#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data

#数组转为二进制
def convert_to_binary(arr):
  X = np.zeros((4 * word_size,len(arr[0])),dtype=np.uint8)
  for i in range(4 * word_size):
    index = i // word_size
    offset = word_size - (i % word_size) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose()
  return(X)

#takes a text file that contains encrypted block0, block1, true diff prob, real or random
#data samples are line separated, the above items whitespace-separated
#returns train data, ground truth, optimal ddt prediction
def readcsv(datei):
    data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s,16) for x in range(2)})
    X0 = [data[i][0] for i in range(len(data))]
    X1 = [data[i][1] for i in range(len(data))]
    Y = [data[i][3] for i in range(len(data))]
    Z = [data[i][2] for i in range(len(data))]
    ct0a = [X0[i] >> 16 for i in range(len(data))]
    ct1a = [X0[i] & MASK_VAL for i in range(len(data))]
    ct0b = [X1[i] >> 16 for i in range(len(data))]
    ct1b = [X1[i] & MASK_VAL for i in range(len(data))]
    ct0a = np.array(ct0a, dtype=np.uint16)
    ct1a = np.array(ct1a,dtype=np.uint16)
    ct0b = np.array(ct0b, dtype=np.uint16) 
    ct1b = np.array(ct1b, dtype=np.uint16)
    
    #X = [[X0[i] >> 16, X0[i] & 0xffff, X1[i] >> 16, X1[i] & 0xffff] for i in range(len(data))]
    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b]) 
    Y = np.array(Y, dtype=np.uint8) 
    Z = np.array(Z)
    return(X,Y,Z)

#baseline training data generator
def make_train_data(n, nr, diff=(0x0040,0)):
  Y = np.frombuffer(urandom(n), dtype=np.uint8) 
  Y = Y & 1
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
  plain1l = plain0l ^ diff[0] 
  plain1r = plain0r ^ diff[1]
  num_rand_samples = np.sum(Y==0)
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
  ks = expand_key(keys, nr)
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
  return(X,Y)

#real differences data generator
def real_differences_data(n, nr, diff=(0x0040,0)):
  #generate labels
  Y = np.frombuffer(urandom(n), dtype=np.uint8) 
  Y = Y & 1
  #generate keys
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
  #generate plaintexts
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
  #apply input difference
  plain1l = plain0l ^ diff[0] 
  plain1r = plain0r ^ diff[1]
  num_rand_samples = np.sum(Y==0)
  #expand keys and encrypt
  ks = expand_key(keys, nr)
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
  #generate blinding values
  k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
  k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
  #apply blinding to the samples labelled as random
  ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0 
  ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1
  ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0 
  ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1
  #convert to input data for neural networks
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
  return(X,Y)
  
  
if __name__== '__main__':
    check_testvector()
