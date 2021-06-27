import numpy as np
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from keras import backend as K
def data_partition(datax, datay, fraction=1/2, random=False):
    """
    data: len* \n
    return x1,x2,y1,y2
    @TODO part2 no random
    """
    if fraction<0 or fraction>1:
        return 0
    allnum = int(datax.shape[0])
    part1num = int(allnum*fraction)
    part2num = allnum - part1num

   # print(part1num,part2num,allnum)
   # assert part1num + part2num == allnum

    print(part1num,part2num)
    if random == False:
        partx1 = datax[:part1num]
        partx2 = datax[part1num:]
        party1 = datay[:part1num]
        party2 = datay[part1num:]
    else:
        part1index = np.random.choice(allnum, part1num, replace=False) 
        part2index = np.delete(np.arange(allnum),part1index)
        part2index = np.random.choice(part2index, part2num, replace=False)
        #print(part1index[:100])
       # print(part2inedx[:100])
        assert len(part1index)+len(part2index) == allnum
        partx1 = datax[part1index]
        partx2 = datax[part2index]
        party1 = datay[part1index]
        party2 = datay[part2index]

    return partx1, partx2, party1, party2 



def load_part_mnist(part, partnum):
    """
    divide mnist (include train and test) into 'partnum' part
    return  'part'-th part of mnist 
    """
    num_classes = 10
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_data, y_data = np.append(x_train,x_test,axis=0), np.append(y_train,y_test,axis=0)

    assert x_data.shape[0] == 70000

    i = part-1
    partsize = int(70000/partnum)
    s = partsize
    data_x = x_data[i*s:(i+1)*s]
    data_y = y_data[i*s:(i+1)*s]

    x_data = data_x
    y_data = data_y

    if K.image_data_format() == 'channels_first':
        x_data = x_data.reshape(x_data.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_data = x_data.reshape(x_data.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_data = x_data.astype('float32')
    x_data /= 255
    print('x_data shape:', x_data.shape)
    # convert class vectors to binary class matrices
    y_data = to_categorical(y_data, num_classes)
    return x_data, y_data, input_shape, num_classes

def add_noise(datax, k=0.1):
    shape = datax.shape
#    print(datax[0])
    datax = datax.reshape(-1)
    allnum = datax.shape[0]
    partnum = int(allnum*k)
    randomindex = np.random.choice(allnum, partnum, replace=False) 
    randoms = np.random.random((partnum))
 #   print(shape,randoms.shape)
    datax[randomindex] = randoms
    datax = datax.reshape(shape)
  #  print(datax.shape,datax[0])
    return datax


def shadowload(fraction = 1/2, database='mnist', trainfraction=1/4):
    """ 
    input fraction
    load data for shadow models(50 or more)
    return size: 17500 from fraction of the mnist   
    """
    if database == 'mnist':
        x, y, _, _ = load_part_mnist(1,1)
        total = 70000
    elif database == 'cifar10':
        x, y, _, _ = load_part_cifar10(1,1)
        total=60000
    else:
        raise Exception('no such db')
    
   # assert False
    totalnum = int(x.shape[0]*fraction)
    x, y = x[:totalnum], y[:totalnum]
    assert x.shape[0]==int(total*fraction)
    x,_, y,_ = data_partition(x,y,2*trainfraction,random=True)
    x_train, x_test, y_train, y_test = data_partition(x,y,1/2,random=True)
   # x_train = add_noise(x_train, 0.1)
   # x_test = add_noise(x_test,0.1)
   # tr_index = np.random.choice(x.shape[0], int(x.shape[0]/2), replace=False)
    # x_train = x[tr_index]
    # y_train = y[tr_index]

    # index_all = np.arange(x.shape[0])

    # te_index = np.delete(index_all, tr_index)
    # x_test = x[te_index]
    # y_test = y[te_index]

    # assert x_test.shape[0] == 17500

    return x_train, x_test, y_train, y_test


def targetload(fraction=1/2,database='mnist', trainfraction=1/12):
    """
    load mnist for target
    """
  #  print ('aaaa')
    if database == 'mnist':
        x, y, _, _ = load_part_mnist(1,1)
        total = 70000
    elif database == 'cifar10':
        x, y, _, _ = load_part_cifar10(1,1)
        total=60000
    else:
        raise Exception('no such db')
    totalnum = int(x.shape[0]*fraction)
    x, y = x[totalnum:], y[totalnum:]

    assert x.shape[0]==int(total*fraction)

#    x,_, y,_ = data_partition(x,y,2*trainfraction,random=True)
    x_train, x_test, y_train, y_test = data_partition(x,y,1/2,random=False)

    print("target train shape2:",x_train.shape)
    print("target test shape:",x_test.shape)

    return x_train, x_test, y_train, y_test


def load_part_cifar10(part, partnum):
    """
    divide - (include train and test) into 'partnum' part
    return  'part'-th part of cifar10 
    """
    num_classes = 10
    img_rows, img_cols = 32, 32
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255


    x_data, y_data = np.append(x_train,x_test,axis=0), np.append(y_train,y_test,axis=0)

    assert x_data.shape[0] == 60000

    i = part-1
    partsize = int(60000/partnum)
    s = partsize
    data_x = x_data[i*s:(i+1)*s]
    data_y = y_data[i*s:(i+1)*s]

    x_data = data_x
    y_data = data_y
    # Convert class vectors to binary class matrices.
    y_data = to_categorical(y_data, num_classes)
    print('x_data shape:', x_data.shape,y_data.shape)
    if K.image_data_format() == 'channels_first':
        x_data = x_data.reshape(x_data.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_data = x_data.reshape(x_data.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
    return x_data, y_data, input_shape, num_classes

def load_cifar10_class(classs=0):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

   # print(x_test.shape,y_test.shape)
   # print(x_test[-1],y_test[-1])
   # deltest = np.where(y_test==classs)
    #y_test = np.delete(y_test,deltest[0],axis=0)
   # x_test = np.delete(x_test,deltest[0],axis=0)
    #print(x_test.shape,y_test.shape)
    deltrain = np.where(y_train==classs)
    tmpx = np.take(x_train,deltrain[0],axis=0)
    tmpy = np.take(y_train,deltrain[0],axis=0)

    x_train = np.delete(x_train, deltrain[0], axis=0)
    y_train = np.delete(y_train, deltrain[0], axis=0)
    
    x_test = np.append(x_test,tmpx,axis=0)
    y_test = np.append(y_test,tmpy,axis=0)
   # print(x_test[-1],y_test[-1])
   # print(y_test)
   # assert False
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255
    return x_train, x_test, y_train, y_test




def load_cifar10_batch(batch_num=1):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

   # print(x_test.shape,y_test.shape)
   # print(x_test[-1],y_test[-1])
   # deltest = np.where(y_test==classs)
    #y_test = np.delete(y_test,deltest[0],axis=0)
   # x_test = np.delete(x_test,deltest[0],axis=0)
    #print(x_test.shape,y_test.shape)
    allnum = x_train.shape[0]
    randomindex = np.random.choice(allnum, batch_num, replace=False) 
 #   print(randomindex)
    deltrain = randomindex,1
    tmpx = np.take(x_train,deltrain[0],axis=0)
    tmpy = np.take(y_train,deltrain[0],axis=0)

    x_train = np.delete(x_train, deltrain[0], axis=0)
    y_train = np.delete(y_train, deltrain[0], axis=0)
    
    x_test = np.append(x_test,tmpx,axis=0)
    y_test = np.append(y_test,tmpy,axis=0)
   # print(x_test[-1],y_test[-1])
   # print(y_test).
   # assert False
    print('x_train shape:', x_train.shape)

   # assert False
    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    tmpy = to_categorical(tmpy,10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    tmpx = tmpx.astype('float32')
    tmpx /= 255
    x_train /= 255
    x_test /= 255
    return x_train, x_test, y_train, y_test,tmpx,tmpy