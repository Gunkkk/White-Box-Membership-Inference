import os
import time 
from multiprocessing import Pool
import numpy as np
import keras
import keras.backend as tf
from keras.datasets import mnist, cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Dropout, Activation, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.models import model_from_json
import time
import json

import membership_attackload_mnist22
from tensorflow import config
gpus = config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
   config.experimental.set_memory_growth(gpu, True)

PN=20
import tensorflow
tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)

def build_cifar10model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32,32,3),activation='relu'))
  #  model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3),activation='relu'))
   # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
   # model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3),activation='relu'))
  #  model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    #model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
   # model.add(Activation('softmax'))


#     json_string = model.to_json()
#     with open('cifar10.json', 'w') as outfile:
#         json.dump(json_string, outfile)
    return model


def trainproxyandshadows(splits):
    x_train = [None for _ in range(splits)]
    x_test = [None for _ in range(splits)]
    y_train = [None for _ in range(splits)]
    y_test = [None for _ in range(splits)]
    modelshadow = [None for _ in range(splits)]
    modelproxy = [None for _ in range(splits)]
    for i in range(splits):
        print(i,'start')
        x_train[i], x_test[i], y_train[i], y_test[i] = membership_attackload_mnist22.shadowload(database='cifar10')
        modelshadow[i] = build_cifar10model()
        modelproxy[i] = build_cifar10model()
        
        modelshadow[i].compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])
        modelproxy[i].compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])
        
        modelshadow[i].fit(x_train[i], y_train[i],
                  batch_size=64,
                  epochs=60,
                  verbose=0,
                  validation_data=(x_test[i], y_test[i]),
                  )

        
        modelproxy[i].fit(x_test[i], y_test[i],
                  batch_size=64,
                  epochs=60,
                  verbose=0,
                  validation_data=(x_train[i], y_train[i]),
                  )   
        modelshadow[i].save_weights('shadow_{}.h5'.format(i))
        modelproxy[i].save_weights('proxy_{}.h5'.format(i))
        
    return x_train, y_train, x_test, y_test, modelshadow, modelproxy



def calwbz(xinput_v,y_input_v,layernum,model):
    steps = 5

    y_index= np.argmax(y_input_v,axis=1)[0]
   # print(x_vs[50],x_v)
    #print(time.time())
    y_true = model.layers[-1].output
    z = model.layers[layernum].output
    xinput = model.layers[0].input
   # z_ph = tf.placeholder(shape=z.shape)
    
    if z == xinput:
        zz = [xinput_v]
    else:    
        zz = tf.function([xinput],[z])([xinput_v])
        
        
    
    if model.layers[layernum].output == y_true:
        #return model.layers[layernum].get_weights()[0], model.layers[layernum].get_weights()[1], zz[0]
        g0 = [zz[0]]
        
    else:
        g0 = tf.function([z],[y_true])([0*zz[0]])
 #   y_tt = tf.function([z],[y_true])(zz[0])
  #  print(g0)
    
    z_vs = [ step/steps * zz[0] for step in range(steps+1) ]
    
   # print(y_tt[0],y_index)
    
    influences = None

    for z_v in z_vs:
       # print(z_ph.shape,z_v.shape)
        
        influence = tf.function([z],[tf.gradients(y_true[0][y_index],z)])([z_v])
      #  print(influence[0][1].shape,len(influence[0][0]))
     #   print(time.time())
        
        if influences is None:
            influences = influence[0][0]
        else:
            influences += influence[0][0]

  #  avg_i = (influences[:-1] + influences[1:])/2.0

    influences /= (steps+1)
    
    return influences,g0[0][0][y_index],zz[0]#influences*zz[0],g0[0][0][y_index],zz[0]


def calwbz2(zz,y_input_v,layernum,model):
    steps = 5

    y_index= np.argmax(y_input_v,axis=1)[0]
   # print(x_vs[50],x_v)
    #print(time.time())
    y_true = model.layers[-1].output
    z = model.layers[layernum].output
    xinput = model.layers[0].input
   # z_ph = tf.placeholder(shape=z.shape)


    if model.layers[layernum].output == y_true:
        g0 = [zz]
   # zz = tf.function([xinput],[z])([xinput_v])
    else:
        g0 = tf.function([z],[y_true])([0*zz])
   # y_tt = tf.function([z],[y_true])(zz)
   # print(g0)
    z_vs = [ step/steps * zz for step in range(steps+1) ]
    
   # print(y_tt[0],y_index)
    
    influences = None

    for z_v in z_vs:
       # print(z_ph.shape,z_v.shape)
        
        influence = tf.function([z],[tf.gradients(y_true[0][y_index],z)])([z_v])
      #  print(influence[0][1].shape,len(influence[0][0]))
     #   print(time.time())
        
        if influences is None:
            influences = influence[0][0]
        else:
            influences += influence[0][0]

  #  avg_i = (influences[:-1] + influences[1:])/2.0

    influences /= (steps+1)
   # print(influences.shape,zz.shape)
    return influences,g0[0][0][y_index]#influences*zz,g0[0][0][y_index]


def getattackdatamp(x,y,index,layernum):
#     import keras.backend as tf
#     #print('111')
#     import tensorflow as tt
#     #print('222')
#     from keras.models import model_from_json
#     #print('333')
#     tf.set_session(tt.compat.v1.Session())
    
   # print('aaa')
    if x.shape[0]==0:
        return None, None
    model1 = build_cifar10model() #model_from_json(json_string)
 #   print('qqq')
    #model1 = build_cifar10model()
    model1.load_weights('shadow_{}.h5'.format(index))
    model1.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])
    model2 = build_cifar10model()#model_from_json(json_string)\
  #  print('zzz')
    model2.load_weights('proxy_{}.h5'.format(index))
    model2.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])
    #model1.summary()
    wbb = None
    zz = None
    #print(tf.eval(model1.weights[0][0,0,0,0]))
    #print(tf.eval(model2.weights[0][0,0,0,0]))
    for j in range(x.shape[0]):
        #print('eee')
        w1,b1,z1 = calwbz(x[j:j+1],y[j:j+1],layernum,model1)
        w2,b2 = calwbz2(z1,y[j:j+1],layernum,model2)
        #w2,b2,z2 = calwbz(x[j:j+1],y[j:j+1],layernum,model2)
#         print(w1.shape,b1.shape,z1.shape)
#         print(w2.shape,b2.shape)
     #   print(w1,w2,b1,b2)
        wb1 = tf.concatenate([tf.reshape(w1,(1,-1)),tf.reshape(b1,(1,-1))],axis=1)
        wb2 = tf.concatenate([tf.reshape(w2,(1,-1)),tf.reshape(b2,(1,-1))],axis=1)


        wb = tf.concatenate([wb1,wb2],axis=0)

  #  wb = tf.expand_dims(wb,-1)
        wb = tf.transpose(wb)#n*2
        wb = tf.reshape(wb,shape=(1,-1,1))
#         wb = tf.eval(wb)
#         z1 = tf.eval(z1)
        
        if wbb is None:
            wbb = wb
        else:
           # print(wbb.shape)
            wbb = tf.concatenate([wbb,wb],axis=0)

        if zz is None:
            zz = z1
        else:
            zz = tf.concatenate([zz,z1],axis=0)
    
    wbb = np.array(tf.eval(wbb))
    zz = np.array(tf.eval(zz))
    #print(wbb,zz)
    return wbb,zz

# def buildattackdatabatchmpoo(x,y,index):
#     #mdoel1 target model2 proxy
#     model1 = build_cifar10model()
#     #model1.load_weights('shadow_{}.h5'.format(index))
    
#     layersnum = len(model1.layers)
#     print(layersnum)
#   #  inputs = Input(shape=x.shape)
#   #  inputz = Input(shape=())
#   #  confidence = []#[None for _ in range(layersnum-1)]#tf.zeros(shape=(layersnum,1,10))
#     wbs = []
#     zs = []
    
#     bs = x.shape[0]

    
    
#     for i in range(layersnum-1):
#         print(i)
#         if model1.layers[i].weights == []:
#             continue
#         wbb = None
#         zz = None
        
        
# #         for j in range(bs):
# #             xinput = x[j:j+1]
# #             yinput = y[j:j+1]
        
#         #print('qqq')
#         #with Pool(10) as pool:
#     #params = [(1, ), (2, ), (3, ), (4, )]
#         pool = Pool(processes = PN)
#         num_per = bs//PN
        
#         results = [pool.apply_async(getattackdatamp, (x[j*num_per:(j+1)*num_per],y[j*num_per:(j+1)*num_per],index,i)) for j in range(PN)]
#         print('www')
#         wbb = None
#         zz = None
#         for ii in results:
#             wtt,ztt = ii.get()

#             if wbb is None:
#                 wbb = wtt
#             else:
#                 wbb = tf.concatenate([wbb,wtt],axis=0)
                
#             if zz is None:
#                 zz = ztt
#             else:
#                 zz = tf.concatenate([zz,ztt],axis=0)

#         pool.close()
#         pool.join()

#         zs.append(zz)
#         wbs.append(wbb)
        

#        # assert False
# #         t1_1 = Conv1D(1, kernel_size=2, strides=2,activation='relu')(wb)
# #         t1_2 = Conv1D(1, kernel_size=1, strides=1,activation='relu')(t1_1)
#  #   print(wbs.shape,zs.shape)

#     return wbs, zs


def buildattackdatabatchmp2(x,y,index):
    import math
    #mdoel1 target model2 proxy
    model1 = build_cifar10model()
    #model1.load_weights('shadow_{}.h5'.format(index))
    
    layersnum = len(model1.layers)
    print(layersnum)
  #  inputs = Input(shape=x.shape)
  #  inputz = Input(shape=())
  #  confidence = []#[None for _ in range(layersnum-1)]#tf.zeros(shape=(layersnum,1,10))
    wbs = []
    zs = []
    
    bs = x.shape[0]

    
    
    for i in range(0,layersnum):
       # print(i)
        if model1.layers[i].weights == []:
            continue
        wbb = None
        zz = None
        
        
#         for j in range(bs):
#             xinput = x[j:j+1]
#             yinput = y[j:j+1]
        
        #print('qqq')
        #with Pool(10) as pool:
    #params = [(1, ), (2, ), (3, ), (4, )]
        
        num_per = bs//PN
        PER_TIME = 200
        PER_P = PER_TIME//PN
        times = math.ceil(bs/(PER_TIME))
        wbb = None
        zz = None
        for q in range(times):
            print(q*PER_TIME)
            ext = q*PER_TIME
            pool = Pool(processes = PN)
            results = [pool.apply_async(getattackdatamp, (x[ext+j*PER_P:ext+(j+1)*PER_P], y[ext+j*PER_P:ext+(j+1)*PER_P],index,i)) for j in range(PN)]
        
        #results = [pool.apply_async(getattackdatamp, (x[j*num_per:(j+1)*num_per],y[j*num_per:(j+1)*num_per],index,i)) for j in range(PN)]
       # print('www')
   
            for ii in results:
                wtt,ztt = ii.get()
                if wtt is None:
                    break
                #ztt = tf.eval(ztt)
                #wtt = tf.eval(wtt)
                if wbb is None:
                    wbb = wtt
                else:
                    wbb = np.concatenate([wbb,wtt],axis=0)

                if zz is None:
                    zz = ztt
                else:
                    zz = np.concatenate([zz,ztt],axis=0)
            
            pool.close()
            pool.join()
        
            
        zs.append(zz)
        wbs.append(wbb)
    print('outtt')

    return wbs, zs




def traindata():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    indexnums = 5
    x_trains, y_trains, x_tests, y_tests, modelshadow, modelproxy = trainproxyandshadows(indexnums)
   # x_train, x_test, y_train, y_test = membership_attackload_mnist22.shadowload(database='cifar10')
    #print(x_train[:2])
    
    os.environ['CUDA_VISIBLE_DEVICES']=''
    
    for index in range(indexnums):
        www1,zzz1 = buildattackdatabatchmp2(x_trains[index][:],y_trains[index][:],index)
      #  with open('www1_{}.pkl'.format(index),'wb') as f:
      #      pickle.dump([tf.eval(i) for i in www1], f, protocol=pickle.HIGHEST_PROTOCOL)
        print('dumppp',index)
      #  print('dump11',www1_0[0].shape)
    #    joblib.dump(www1_0,'www1_0.pkl',compress=5)
        hickle.dump(www1,'1www1_{}.hkl'.format(index),mode='w',compression='gzip')
        
        #assert False
        hickle.dump(zzz1,'1zzz1_{}.hkl'.format(index),mode='w',compression='gzip')
        
       #joblib.dump(www1_1,'www1_1_{}.pkl'.format(index),compress=3)
       # joblib.dump([tf.eval(i) for i in www1_2],'www1_2_{}.pkl'.format(index),compress=3)
#         #with open('zzz1_{}.pkl'.format(index),'wb') as f:
#         #    pickle.dump([tf.eval(i) for i in zzz1], f, protocol=pickle.HIGHEST_PROTOCOL)
        
        #joblib.dump([tf.eval(i) for i in zzz1],'zzz1_{}.pkl'.format(index),compress=3)
        #joblib.dump(zzz1,'zzz1_{}.pkl'.format(index),compress=3)
      #  assert False
        www0,zzz0 = buildattackdatabatchmp2(x_tests[index][:],y_tests[index][:],index)
#         with open('www0_{}.pkl'.format(index),'wb') as f:
#             pickle.dump([tf.eval(i) for i in www0], f, protocol=pickle.HIGHEST_PROTOCOL)

#         with open('zzz0_{}.pkl'.format(index),'wb') as f:
#             pickle.dump([tf.eval(i) for i in zzz0], f, protocol=pickle.HIGHEST_PROTOCOL)
        hickle.dump(www0,'1www0_{}.hkl'.format(index),mode='w',compression='gzip')
        
        #assert False
        hickle.dump(zzz0,'1zzz0_{}.hkl'.format(index),mode='w',compression='gzip')


def databylabel(x,y):
   
    y_index = np.argmax(y,axis=1)
    data = {}
    for i in range(10):
        data[i] = None
        data[str(i)+'_y'] = None
        
    for index,i in enumerate(y_index):
        if data[i] is None:
            data[i] = x[index:index+1] 
            data[str(i)+'_y'] = i
        else:
            data[i] = np.concatenate([data[i],x[index:index+1]],axis=0)
            data[str(i)+'_y'] = np.append(data[str(i)+'_y'],i)
    for i in range(10):
        data[str(i)+'_y'] = to_categorical(data[str(i)+'_y'],10)
    
    return data
    
    
def calibratingdata():
    x_train, x_test, y_train, y_test = membership_attackload_mnist22.shadowload(database='cifar10',trainfraction=1/2)
    
    x = np.concatenate([x_train,x_test],axis=0)
    y = np.concatenate([y_train,y_test],axis=0)
    os.environ['CUDA_VISIBLE_DEVICES']=''
 #   print(x.shape[0])
    data = databylabel(x,y)
    
    for i in range(10):
        print('label:',i)
        x_data = data[i]
        y_data = data[str(i)+'_y']
       # print(x_data.shape,y_data.shape)
       # assert False
        w,z = buildattackdatabatchmp2(x_data,y_data,999)
       # assert False
        hickle.dump(w,'calw_{}.hkl'.format(i),mode='w',compression='gzip')
        hickle.dump(z,'calz_{}.hkl'.format(i),mode='w',compression='gzip')
        
        
        
        
        
            
def testdata():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    targetmodel = build_cifar10model()

    targetmodel.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])

    x_train1, x_test1, y_train1, y_test1 = membership_attackload_mnist22.targetload(database='cifar10')
    #print(x_train1.shape)
    targetmodel.fit(x_train1, y_train1,
                  batch_size=64,
                  epochs=70,
                  verbose=2,
                  validation_data=(x_test1, y_test1),
                  )
    targetmodel.save_weights('shadow_999.h5')
    proxymodel = build_cifar10model()
    proxymodel.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])
    x_train, x_test, y_train, y_test = membership_attackload_mnist22.shadowload(database='cifar10',trainfraction=1/2)
    x_proxy = np.concatenate([x_train,x_test],axis=0)
    y_proxy = np.concatenate([y_train,y_test],axis=0)
    
    proxymodel.fit(x_proxy, y_proxy,
                  batch_size=64,
                  epochs=70,
                  verbose=2,
                  #validation_data=(x_test, y_test),
                  )
     
    proxymodel.save_weights('proxy_999.h5')
    
    os.environ['CUDA_VISIBLE_DEVICES']=''
    
    
    data_train = databylabel(x_train1,y_train1)
    data_test = databylabel(x_test1,y_test1)
    
    for i in range(10):
        
        
        testw1, testz1 = buildattackdatabatchmp2(data_train[i],data_train[str(i)+'_y'],999)
        
        hickle.dump(testw1,'testw1_{}.hkl'.format(i),mode='w',compression='gzip')

            #assert False
        hickle.dump(testz1,'testz1_{}.hkl'.format(i),mode='w',compression='gzip')
      #  assert False
        
        testw0, testz0 = buildattackdatabatchmp2(data_test[i],data_test[str(i)+'_y'],999)



        hickle.dump(testw0,'testw0_{}.hkl'.format(i),mode='w',compression='gzip')

            #assert False
        hickle.dump(testz0,'testz0_{}.hkl'.format(i),mode='w',compression='gzip')
    
    
    #calibratingdata(x_train, x_test, y_train, y_test)

    
import multiprocessing
import pickle 
import joblib
import hickle
multiprocessing.set_start_method('spawn', force=True)

if __name__ == '__main__':
    traindata()
    testdata()
    calibratingdata()
        #joblib.dump([tf.eval(i) for i in www0],'www0_{}.pkl'.format(index))
        
       # joblib.dump([tf.eval(i) for i in zzz0],'zzz0_{}.pkl'.format(index))
# from keras.models import model_from_json
# import json
# with open('{}.json'.format('cifar10')) as data_file:
#     # with open('cifar10_zero_no_dropout.json') as data_file:
#     json_string = json.load(data_file)
# print(json_string)
# model1 =  model_from_json(json_string)
# model1.summary()
# model1.load_weights('shadow_0.h5')
# print(model1)
# model2 =  model_from_json(json_string)
# model2.summary()
# model2.load_weights('shadow_1.h5')
# print(model2)
# assert model1==model2
