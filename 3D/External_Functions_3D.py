"""The present code is the Version 1.0 of the RCNN approach to perform MPS 
in 3D for categorical variables. It has been developed by S. Avalos and J. Ortiz in the
Geometallurygical Group at Queen's University as part of a PhD program.
The code is not free of bugs but running end-to-end. 
Any comments and further improvements are well recevied to: 17saa6@queensu.ca
April 16, 2019.
Geomet Group - Queen's University - Canada"""

# Do not display the AVX message about using GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import time
import gc
import csv
from mayavi import mlab



class Grid():
    
    def __init__(self, HyperPar, DBname, Lvl, Training=True, Padding=False):
        self.SGsizex = int(HyperPar[0]) # Simulation grid X
        self.SGsizey = int(HyperPar[1]) # Simulation grid Y
        self.SGsizez = int(HyperPar[2]) # Simulation grid z
        
        self.Search_x = int(HyperPar[3]) # odd number search grid X
        self.Search_y = int(HyperPar[4]) # odd number search grid Y
        self.Search_z = int(HyperPar[5]) # odd number search grid Z
        
        self.IPsizex = int(HyperPar[6]) # odd inner pattern search grid X   
        self.IPsizey = int(HyperPar[7]) # odd inner pattern search grid Y   
        self.IPsizez = int(HyperPar[8]) # odd inner pattern search grid Y   
        
        self.PercDC = np.around(HyperPar[9]/1000, decimals=3) # data conditioning percentage
        self.MinDC = int(HyperPar[10]) # Minimum data conditioning 
        
        self.delta_x = int((HyperPar[3]-1)/2)
        self.delta_y = int((HyperPar[4]-1)/2)
        self.delta_z = int((HyperPar[5]-1)/2)
        
        self.d_Sim_x = int((HyperPar[6]-1)/2)
        self.d_Sim_y = int((HyperPar[7]-1)/2)
        self.d_Sim_z = int((HyperPar[8]-1)/2)
        
        self.Level = Lvl
        self.NumCategs = int(HyperPar[14]) 
        self.Pad = Padding
        
        self.TrainingImage = np.loadtxt(DBname).reshape(self.SGsizex,self.SGsizey,self.SGsizez)
        count_0 = 0
                    
        if self.Pad==True:
            self.Values = np.zeros([self.SGsizex+2*self.delta_x,self.SGsizey+2*self.delta_y,self.SGsizez+2*self.delta_z,self.Level])
            self.DataConditioning = np.ones([self.SGsizex+2*self.delta_x,self.SGsizey+2*self.delta_y,self.SGsizez+2*self.delta_z,self.Level])
            self.TrainingImage_Padding = np.zeros([self.SGsizex+2*self.delta_x,self.SGsizey+2*self.delta_y,self.SGsizez+2*self.delta_z])
            self.TrainingImage_Padding[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey,self.delta_z:self.delta_z+self.SGsizez] = self.TrainingImage
            self.largo = int((HyperPar[0])*(HyperPar[1])*(HyperPar[2])) 
                    
            for ind0 in range(self.Level):
                self.Values[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey,self.delta_z:self.delta_z+self.SGsizez,ind0] = np.copy(self.TrainingImage)   #np.around(((np.copy(self.TrainingImage)-1)/(self.NumCategs-1) + 1) ,decimals=2)        

            self.TI_Matrix = np.zeros((self.largo,3))
            for indz in range(self.SGsizez):
                for indy in range(self.SGsizey):
                    for indx in range(self.SGsizex):
                        self.TI_Matrix[count_0,0],self.TI_Matrix[count_0,1],self.TI_Matrix[count_0,2] = indx, indy, indz
                        count_0 += 1
        if self.Pad==False:
            self.Values = np.zeros([self.SGsizex,self.SGsizey,self.SGsizez,self.Level])
            self.DataConditioning = np.ones([self.SGsizex,self.SGsizey,self.SGsizez,self.Level])
            self.largo = int((HyperPar[0]-HyperPar[3]+1)*(HyperPar[1]-HyperPar[4]+1)*(HyperPar[2]-HyperPar[5]+1)) 
    
            for ind0 in range(self.Level):
                self.Values[:,:,:,ind0] = np.copy(self.TrainingImage)      
            
            self.TI_Matrix = np.zeros((self.largo,3))
            for indz in range(self.SGsizez-self.Search_z+1):
                for indy in range(self.SGsizey-self.Search_y+1):
                    for indx in range(self.SGsizex-self.Search_x+1):
                        self.TI_Matrix[count_0,0],self.TI_Matrix[count_0,1],self.TI_Matrix[count_0,2] = indx, indy, indz
                        count_0 += 1     
        if Training == True:
            G = np.random.random((self.Values.shape[0],self.Values.shape[1],self.Values.shape[2]))
            for indx in range(self.Values.shape[0]):
                for indy in range(self.Values.shape[1]):
                    for indz in range(self.Values.shape[2]):
                        if G[indx,indy,indz] < self.PercDC:
                            G[indx,indy,indz] = 1
                        else:
                            G[indx,indy,indz] = 0       
            # Random Drill Holes
            #G = np.random.random((self.Values.shape[0],self.Values.shape[1],self.Values.shape[2]))
            #for indy in range(self.Values.shape[1]):
                #for indz in range(self.Values.shape[2]):
                    #if G[0,indy,indz] < self.PercDC:
                        #G[:,indy,indz] = 1
                    #else:
                        #G[:,indy,indz] = 0
            for ind0 in range(self.Values.shape[3]):
                self.Values[:,:,:,ind0] = np.multiply(self.Values[:,:,:,ind0],G)
    
        for indx in range(self.DataConditioning.shape[0]):
            for indy in range(self.DataConditioning.shape[1]):
                for indz in range(self.DataConditioning.shape[2]):
                    if self.Values[indx][indy][indz][0] != 0:
                        self.DataConditioning[indx,indy,indz] = 0 # 1 are NON conditional, 0  ARE conditionals

        # ######
    def Target(self, ti, tj):
        Total_len = int(tj-ti)
        TargetMatrix = np.zeros((Total_len,self.IPsizex,self.IPsizey,self.IPsizez,1))  
        ind0 = 0
        ind1 = 0
        if self.Pad == True:
            for indx in range(self.SGsizex):
                for indy in range(self.SGsizey):
                    for indz in range(self.SGsizez):
                        if ind0 >= ti and ind0 < tj:
                            TargetMatrix[ind1,:,:,:,0] = self.TrainingImage_Padding[indx:indx + self.IPsizex, indy:indy + self.IPsizey, indz:indz + self.IPsizez]
                            ind1 += 1
                        ind0 += 1    
        if self.Pad == False:
            for indx in range(self.SGsizex-self.Search_x+1):
                for indy in range(self.SGsizey-self.Search_y+1):
                    for indz in range(self.SGsizez-self.Search_z+1):
                        if ind0 >= ti and ind0 < tj:
                            posx = indx + self.delta_x - self.d_Sim_x
                            posy = indy + self.delta_y - self.d_Sim_y
                            posz = indz + self.delta_z - self.d_Sim_z
                            TargetMatrix[ind1,:,:,:,0] = self.TrainingImage[posx:posx + self.IPsizex, posy:posy + self.IPsizey, posz:posz + self.IPsizez]
                            ind1 += 1
                        ind0 += 1           
        return TargetMatrix[:,:,:,:,0]    
    

    
    def InputFirstGrid(self, ti, tj):
        Total_len = int(tj-ti)
        RawInput = np.zeros((Total_len,self.Search_x,self.Search_y,self.Search_z,1))  
        ind0 = 0
        ind1 = 0
        if self.Pad == True:
            for indx in range(self.SGsizex):
                for indy in range(self.SGsizey):
                    for indz in range(self.SGsizez):
                        if ind0 >= ti and ind0 < tj:
                            RawInput[ind1,:,:,:,0] = self.Values[indx:indx + self.Search_x, indy:indy + self.Search_y, indz:indz + self.Search_z,0]
                            ind1 += 1
                        ind0 += 1        
        if self.Pad == False:
            for indx in range(self.SGsizex-self.Search_x+1):
                for indy in range(self.SGsizey-self.Search_y+1):
                    for indz in range(self.SGsizez-self.Search_z+1):
                        if ind0 >= ti and ind0 < tj:
                            RawInput[ind1,:,:,:,0] = self.Values[indx:indx + self.Search_x, indy:indy + self.Search_y, indz:indz + self.Search_z,0]
                            ind1 += 1
                        ind0 += 1   
        return RawInput     
       

    
    def InputLevelGrid(self, ti, tj, Level):
        Total_len = int(tj-ti)
        RawInput = np.zeros((Total_len,self.Search_x,self.Search_y,self.Search_z,Level))  
        ind0 = 0
        ind1 = 0
        if self.Pad == True:
            for indx in range(self.SGsizex):
                for indy in range(self.SGsizey):
                    for indz in range(self.SGsizez):
                        if ind0 >= ti and ind0 < tj:
                            for nivel in range(Level):
                                RawInput[ind1,:,:,:,nivel] = self.Values[indx:indx + self.Search_x, indy:indy + self.Search_y, indz:indz + self.Search_z,nivel]
                            ind1 += 1
                        ind0 += 1        
        if self.Pad == False:
            for indx in range(self.SGsizex-self.Search_x+1):
                for indy in range(self.SGsizey-self.Search_y+1):
                    for indz in range(self.SGsizez-self.Search_z+1):
                        if ind0 >= ti and ind0 < tj:
                            for nivel in range(Level):
                                RawInput[ind1,:,:,:,nivel] = self.Values[indx:indx + self.Search_x, indy:indy + self.Search_y, indz:indz + self.Search_z,nivel]
                            ind1 += 1
                        ind0 += 1          
        return RawInput      
    
    def HotspotCenterValue0n1(self, CenterValueInnerGrid):
        largo = CenterValueInnerGrid.shape[0]
        A = np.zeros([largo, self.IPsizex, self.IPsizey, self.IPsizez,self.NumCategs])
    
        for ind0 in range(largo):
            for indx in range(self.IPsizex):
                for indy in range(self.IPsizey):
                    for indz in range(self.IPsizez):
                        A[ind0][indx][indy][indz][int(CenterValueInnerGrid[ind0][indx][indy][indz]-1)] = 1
        return A    
    

    def DC_Matrix(self):
        Matrix = np.copy(self.DataConditioning[:,:,:,0])
        return Matrix
    def RandomPath(self):
        A = np.ones((self.largo,3))#,dtype=int32)
        count = 0                    
        if self.Pad == True:
            for indx in range(self.SGsizex):
                for indy in range(self.SGsizey):
                    for indz in range(self.SGsizez):
                        A[count,0], A[count,1], A[count,2] = indx + self.delta_x, indy + self.delta_y, indz + self.delta_z
                        count += 1
        if self.Pad == False:
            for indx in range(self.SGsizex-self.Search_x+1):
                for indy in range(self.SGsizey-self.Search_y+1):
                    for indz in range(self.SGsizez-self.Search_z+1):
                        A[count,0], A[count,1], A[count,2] = indx + self.delta_x, indy + self.delta_y, indz + self.delta_z
                        count += 1        
        np.random.shuffle(A)
        return A  
    
    def PDS_Path(self):
        A = np.ones((self.largo,3))
        count = 0                    
        if self.Pad == False:
            for indx in range(self.SGsizex-self.Search_x+1):
                for indy in range(self.SGsizey-self.Search_y+1):
                    for indz in range(self.SGsizez-self.Search_z+1):
                        A[count,0], A[count,1], A[count,2] = indx + self.delta_x, indy + self.delta_y, indz + self.delta_z
                        count += 1     
        return A      
    
    def PredPattern_Categories(self, VolSoft):
        A = np.zeros((self.IPsizex, self.IPsizey,self.IPsizez))
        B = np.random.rand(self.IPsizex, self.IPsizey,self.IPsizez)
        for ind_x in range(self.IPsizex):
            for ind_y in range(self.IPsizey):
                for ind_z in range(self.IPsizez):
                    A[ind_x,ind_y,ind_z] = 0
                    cat = 0
                    while np.sum(VolSoft[ind_x,ind_y,ind_z,0:(cat+1)]) < B[ind_x,ind_y,ind_z]:
                        cat += 1
                        A[ind_x,ind_y,ind_z] = cat
        return A
    
    def PredPattern(self, VolSoft):
        A = np.zeros((self.IPsizex, self.IPsizey,self.IPsizez))
        B = np.random.rand(self.IPsizex, self.IPsizey,self.IPsizez)
        for ind_x in range(self.IPsizex):
            for ind_y in range(self.IPsizey):
                for ind_z in range(self.IPsizez):
                    A[ind_x,ind_y,ind_z] = np.argmax(VolSoft[ind_x,ind_y,ind_z,:])
        return A 
    
    def SavePlot(self,name,Level):
        mlab.options.offscreen = True
        A = str(name) + ".png"
        mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(800, 700))
        if self.Pad == True:
            pts = mlab.points3d(self.TI_Matrix[:,0],self.TI_Matrix[:,1],self.TI_Matrix[:,2],self.Values[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey,self.delta_z:self.delta_z+self.SGsizez,Level].reshape(self.largo), scale_factor=1,  scale_mode='none', mode='cube', vmin=0, vmax=2)
        if self.Pad == False:
            pts = mlab.points3d(self.TI_Matrix[:,0],self.TI_Matrix[:,1],self.TI_Matrix[:,2],self.Values[self.delta_x:self.SGsizex-self.delta_x,self.delta_y:self.SGsizey-self.delta_y,self.delta_z:self.SGsizez-self.delta_z,Level].reshape(self.largo), scale_factor=1,  scale_mode='none', mode='cube', vmin=0, vmax=2)
        mesh = mlab.pipeline.delaunay2d(pts)
        mlab.colorbar(orientation='vertical', nb_labels=(self.NumCategs+1), label_fmt='%1.0f')
        mlab.savefig(A)
        return mlab.close()    
 
 
    def SaveGrid(self,file="results.txt"):
        with open(file, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter = '\t')
            for indx in range(self.SGsizex):
                for indy in range(self.SGsizey):
                    for indz in range(self.SGsizez):
                        b = []
                        for level in range(self.Level):
                            if self.Pad == True:
                                b += [int(self.Values[self.delta_x+indx][self.delta_y+indy][self.delta_z+indz][level])]
                            else:
                                b += [int(self.Values[indx][indy][indz][level])]
                        spamwriter.writerow(b)
    def BP_Calc(self, Matrix):
        return np.around(np.average(np.abs(Matrix-2)), decimals=3)
    def BP_CalcNewRep(self, Matrix):
        return np.around(np.average(Matrix-1), decimals=3)    
    # 
    def PDS(self,LocModel):
        print("Percentage:", self.PercDC)
        tf.reset_default_graph() # A brand new graph each run
        with tf.device('/gpu:0'):
            config = tf.ConfigProto(allow_soft_placement = True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.8 # Testing
            sess = tf.InteractiveSession(config = config) 
            saver = tf.train.import_meta_graph(LocModel+'.meta')
            saver.restore(sess, save_path=LocModel)
            graph = tf.get_default_graph()
    
            x_image_a = graph.get_tensor_by_name("x_image_a:0")
            x_image_b = graph.get_tensor_by_name("x_image_b:0")
            x_image_c = graph.get_tensor_by_name("x_image_c:0")
            x_image_d = graph.get_tensor_by_name("x_image_d:0")
            InTrain = graph.get_tensor_by_name("InTrain:0")   
            softmax_y_conv_a = graph.get_tensor_by_name("softmax_y_conv_a:0")
            softmax_y_conv_b = graph.get_tensor_by_name("softmax_y_conv_b:0")
            softmax_y_conv_c = graph.get_tensor_by_name("softmax_y_conv_c:0")
            softmax_y_conv_d = graph.get_tensor_by_name("softmax_y_conv_d:0")
            
            argmax_y_conv_a = graph.get_tensor_by_name("argmax_y_conv_a:0")
            argmax_y_conv_b = graph.get_tensor_by_name("argmax_y_conv_b:0")
            argmax_y_conv_c = graph.get_tensor_by_name("argmax_y_conv_c:0")
            argmax_y_conv_d = graph.get_tensor_by_name("argmax_y_conv_d:0")            
    
            # ###############################################################
            # Running over Simulation grids
            # ###############################################################     
            Level_T = np.ones(((1, self.Search_x, self.Search_y, self.Search_z, 4)))
            Path = self.PDS_Path()
            Prop_Matrix = np.zeros((Path.shape[0],4))
            for ind0 in range(Path.shape[0]):   
                xi = int(Path[ind0,0])
                xj = int(Path[ind0,1])
                xk = int(Path[ind0,2])
                Level_T[0,:,:,:,0] = self.Values[xi - self.delta_x : xi + self.delta_x +1, xj - self.delta_y : xj + self.delta_y + 1, xk - self.delta_z : xk + self.delta_z + 1,0]
                IP_a = argmax_y_conv_a.eval(feed_dict={x_image_a: Level_T[:,:,:,:,0:1], InTrain:False})[0] + 1  
                Level_T[0,:,:,:,1] = IP_a
                IP_b = argmax_y_conv_b.eval(feed_dict={x_image_b: Level_T[:,:,:,:,0:2], InTrain:False})[0]  + 1 
                Level_T[0,:,:,:,2] = IP_b
                IP_c = argmax_y_conv_c.eval(feed_dict={x_image_c: Level_T[:,:,:,:,0:3], InTrain:False})[0] + 1 
                Level_T[0,:,:,:,3] = IP_c
                IP_d = argmax_y_conv_d.eval(feed_dict={x_image_d: Level_T[:,:,:,:,0:4], InTrain:False})[0] + 1 

                Prop_Matrix[ind0,0], Prop_Matrix[ind0,1], Prop_Matrix[ind0,2], Prop_Matrix[ind0,3] = self.BP_Calc(IP_a), self.BP_Calc(IP_b), self.BP_Calc(IP_c), self.BP_Calc(IP_d)
        tf.get_default_graph().finalize()
        gc.collect()
        sess.close()            
            
        return np.around(np.average(Prop_Matrix, axis=0), decimals=3)
    def PDS_NewRep(self,LocModel):
        print("Percentage:", self.PercDC)
        tf.reset_default_graph() # A brand new graph each run
        with tf.device('/gpu:0'):
            config = tf.ConfigProto(allow_soft_placement = True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.8 # Testing
            sess = tf.InteractiveSession(config = config) 
            saver = tf.train.import_meta_graph(LocModel+'.meta')
            saver.restore(sess, save_path=LocModel)
            graph = tf.get_default_graph()
    
            x_image_a = graph.get_tensor_by_name("x_image_a:0")
            x_image_b = graph.get_tensor_by_name("x_image_b:0")
            x_image_c = graph.get_tensor_by_name("x_image_c:0")
            x_image_d = graph.get_tensor_by_name("x_image_d:0")
            InTrain = graph.get_tensor_by_name("InTrain:0")   
            softmax_y_conv_a = graph.get_tensor_by_name("softmax_y_conv_a:0")
            softmax_y_conv_b = graph.get_tensor_by_name("softmax_y_conv_b:0")
            softmax_y_conv_c = graph.get_tensor_by_name("softmax_y_conv_c:0")
            softmax_y_conv_d = graph.get_tensor_by_name("softmax_y_conv_d:0")
            
            argmax_y_conv_a = graph.get_tensor_by_name("argmax_y_conv_a:0")
            argmax_y_conv_b = graph.get_tensor_by_name("argmax_y_conv_b:0")
            argmax_y_conv_c = graph.get_tensor_by_name("argmax_y_conv_c:0")
            argmax_y_conv_d = graph.get_tensor_by_name("argmax_y_conv_d:0")            
    
            # ###############################################################
            # Running over Simulation grids
            # ###############################################################     
            Level_T = np.ones(((1, self.Search_x, self.Search_y, self.Search_z, 4)))
            Path = self.PDS_Path()
            Prop_Matrix = np.zeros((Path.shape[0],4))
            for ind0 in range(Path.shape[0]):   
                xi = int(Path[ind0,0])
                xj = int(Path[ind0,1])
                xk = int(Path[ind0,2])
                Level_T[0,:,:,:,0] = self.Values[xi - self.delta_x : xi + self.delta_x +1, xj - self.delta_y : xj + self.delta_y + 1, xk - self.delta_z : xk + self.delta_z + 1,0]
                IP_a = argmax_y_conv_a.eval(feed_dict={x_image_a: Level_T[:,:,:,:,0:1], InTrain:False})[0] + 1 
                Level_T[0,:,:,:,1] = IP_a
                IP_b = argmax_y_conv_b.eval(feed_dict={x_image_b: Level_T[:,:,:,:,0:2], InTrain:False})[0]  + 1 
                Level_T[0,:,:,:,2] = IP_b
                IP_c = argmax_y_conv_c.eval(feed_dict={x_image_c: Level_T[:,:,:,:,0:3], InTrain:False})[0] + 1 
                Level_T[0,:,:,:,3] = IP_c
                IP_d = argmax_y_conv_d.eval(feed_dict={x_image_d: Level_T[:,:,:,:,0:4], InTrain:False})[0] + 1 

                Prop_Matrix[ind0,0], Prop_Matrix[ind0,1], Prop_Matrix[ind0,2], Prop_Matrix[ind0,3] = self.BP_CalcNewRep(IP_a), self.BP_CalcNewRep(IP_b), self.BP_CalcNewRep(IP_c), self.BP_CalcNewRep(IP_d)
            #print(np.around(np.average(Prop_Matrix, axis=0), decimals=3))
        tf.get_default_graph().finalize()
        gc.collect()
        sess.close()            
            
        return np.around(np.average(Prop_Matrix, axis=0), decimals=3)    
    
    def Simulate_4ConvNets_BN_3D(self,LocModel,Cicle,Plot=False):
        tf.reset_default_graph() # A brand new graph each run
        with tf.device('/gpu:0'):
            
            config = tf.ConfigProto(allow_soft_placement = True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 
            #config.gpu_options.allow_growth=True
            sess = tf.InteractiveSession(config = config) 
            saver = tf.train.import_meta_graph(LocModel+'.meta')
            saver.restore(sess, save_path=LocModel)
            graph = tf.get_default_graph()
    
            x_image_a = graph.get_tensor_by_name("x_image_a:0")
            x_image_b = graph.get_tensor_by_name("x_image_b:0")
            x_image_c = graph.get_tensor_by_name("x_image_c:0")
            x_image_d = graph.get_tensor_by_name("x_image_d:0")
            InTrain = graph.get_tensor_by_name("InTrain:0")   
            softmax_y_conv_a = graph.get_tensor_by_name("softmax_y_conv_a:0")
            softmax_y_conv_b = graph.get_tensor_by_name("softmax_y_conv_b:0")
            softmax_y_conv_c = graph.get_tensor_by_name("softmax_y_conv_c:0")
            softmax_y_conv_d = graph.get_tensor_by_name("softmax_y_conv_d:0")
            
            argmax_y_conv_a = graph.get_tensor_by_name("argmax_y_conv_a:0")
            argmax_y_conv_b = graph.get_tensor_by_name("argmax_y_conv_b:0")
            argmax_y_conv_c = graph.get_tensor_by_name("argmax_y_conv_c:0")
            argmax_y_conv_d = graph.get_tensor_by_name("argmax_y_conv_d:0")            
            
    
            # ###############################################################
            # Simulate grids
            # ###############################################################  
            Path = self.RandomPath()
            DC_Position_a = self.DC_Matrix()
            DC_Position_b = self.DC_Matrix()
            DC_Position_c = self.DC_Matrix()
            DC_Position_d = self.DC_Matrix()
            
            if Plot==True:
                self.SavePlot(name=LocModel+"_{}_DC".format(Cicle),Level=0)
            print("First Sim Grid..")
            time_firstGird = time.time()
            for ind0 in range(Path.shape[0]):   
                xi = int(Path[ind0,0])
                xj = int(Path[ind0,1])
                xk = int(Path[ind0,2])
                
                if DC_Position_a[xi][xj][xk] == 1:
                    if np.count_nonzero(self.Values[xi - self.delta_x : xi + self.delta_x +1, xj - self.delta_y : xj + self.delta_y + 1, xk - self.delta_z : xk + self.delta_z + 1, 0]) < self.MinDC:
                        temp_point = np.random.uniform()
                        if temp_point < 0.236:
                            self.Values[xi,xj,xk, 0] = 2
                        else: 
                            self.Values[xi,xj,xk, 0] = 1
                    SimInPut_a =  np.expand_dims(np.expand_dims(self.Values[xi - self.delta_x : xi + self.delta_x +1, xj - self.delta_y : xj + self.delta_y + 1, xk - self.delta_z : xk + self.delta_z + 1,0],axis=0),axis=4)
                    IP_final = argmax_y_conv_a.eval(feed_dict={x_image_a: SimInPut_a, InTrain:False})[0] + 1  
                    self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,1] = np.multiply(self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,1], (1 - DC_Position_a[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1])) # maintain only DC
                    TempDC = np.copy(np.multiply(DC_Position_a[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1],np.random.randint(2, size=(self.IPsizex, self.IPsizey,self.IPsizez))))
                    TempDC[self.d_Sim_x][self.d_Sim_y][self.d_Sim_z] = 1
                    self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,1] += np.multiply(IP_final, TempDC) # Resimulate non conditional points 
                    DC_Position_a[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1] = np.multiply(DC_Position_a[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1],(1-TempDC)) # convert ins                    

            print(".............. Done --%s seconds of simulation-" % (np.around((time.time() - time_firstGird), decimals=2)))   
            if Plot==True:
                self.SavePlot(name=LocModel+"_{}_FirstGrid".format(Cicle),Level=1)
            
            print("Second Sim Grid..")
            time_secondGird = time.time()
            for ind0 in range(Path.shape[0]):   
                xi = int(Path[ind0,0])
                xj = int(Path[ind0,1])
                xk = int(Path[ind0,2])             
    
                if DC_Position_b[xi][xj][xk] == 1:  
                    SimInPut_b =  np.expand_dims(self.Values[xi - self.delta_x : xi + self.delta_x +1, xj - self.delta_y : xj + self.delta_y + 1, xk - self.delta_z : xk + self.delta_z + 1,[0,1]],axis=0)                        
                    IP_final = argmax_y_conv_b.eval(feed_dict={x_image_b: SimInPut_b, InTrain:False})[0]  + 1 
                    self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,2] = np.multiply(self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,2], (1 - DC_Position_b[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1])) # maintain only DC
                    TempDC = np.copy(np.multiply(DC_Position_b[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1],np.random.randint(2, size=(self.IPsizex, self.IPsizey,self.IPsizez))))
                    TempDC[self.d_Sim_x][self.d_Sim_y][self.d_Sim_z] = 1
                    self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,2] += np.multiply(IP_final, TempDC) # Resimulate non conditional points 
                    DC_Position_b[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1] = np.multiply(DC_Position_b[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1],(1-TempDC)) # convert ins                    
            print(".............. Done --%s seconds of simulation-" % (np.around((time.time() - time_secondGird), decimals=2)))   
            if Plot==True:
                self.SavePlot(name=LocModel+"_{}_SecondGrid".format(Cicle),Level=2)
            
            print("Third Sim Grid..")
            time_thirdGird = time.time()
            for ind0 in range(Path.shape[0]):   
                xi = int(Path[ind0,0])
                xj = int(Path[ind0,1])
                xk = int(Path[ind0,2])   
                
                if DC_Position_c[xi][xj][xk] == 1:   
                    SimInPut_c =  np.expand_dims(self.Values[xi - self.delta_x : xi + self.delta_x +1, xj - self.delta_y : xj + self.delta_y + 1, xk - self.delta_z : xk + self.delta_z + 1,[0,1,2]],axis=0)                        
                    IP_final = argmax_y_conv_c.eval(feed_dict={x_image_c: SimInPut_c, InTrain:False})[0] + 1 
                    self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,3] = np.multiply(self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,3], (1 - DC_Position_c[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1])) # maintain only DC
                    TempDC = np.copy(np.multiply(DC_Position_c[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1],np.random.randint(2, size=(self.IPsizex, self.IPsizey,self.IPsizez))))
                    TempDC[self.d_Sim_x][self.d_Sim_y][self.d_Sim_z] = 1
                    self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,3] += np.multiply(IP_final, TempDC) # Resimulate non conditional points 
                    DC_Position_c[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1] = np.multiply(DC_Position_c[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1],(1-TempDC)) # convert ins               
            print(".............. Done --%s seconds of simulation-" % (np.around((time.time() - time_thirdGird), decimals=2)))   
            if Plot==True:
                self.SavePlot(name=LocModel+"_{}_ThirdGrid".format(Cicle),Level=3)
            
            print("Fourth Sim Grid..")
            time_fourthGird = time.time()
            for ind0 in range(Path.shape[0]):   
                xi = int(Path[ind0,0])
                xj = int(Path[ind0,1])
                xk = int(Path[ind0,2])                
    
                if DC_Position_d[xi][xj][xk] == 1: 
                    SimInPut_d =  np.expand_dims(self.Values[xi - self.delta_x : xi + self.delta_x +1, xj - self.delta_y : xj + self.delta_y + 1, xk - self.delta_z : xk + self.delta_z + 1,[0,1,2,3]],axis=0)                        
                    IP_final = argmax_y_conv_d.eval(feed_dict={x_image_d: SimInPut_d, InTrain:False})[0] + 1 
                    self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,4] = np.multiply(self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,4], (1 - DC_Position_d[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1])) # maintain only DC
                    TempDC = np.copy(np.multiply(DC_Position_d[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1],np.random.randint(2, size=(self.IPsizex, self.IPsizey,self.IPsizez))))
                    TempDC[self.d_Sim_x][self.d_Sim_y][self.d_Sim_z] = 1
                    self.Values[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1,4] += np.multiply(IP_final, TempDC) # Resimulate non conditional points 
                    DC_Position_d[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1] = np.multiply(DC_Position_d[xi-self.d_Sim_x:xi+self.d_Sim_x+1,xj-self.d_Sim_y:xj+self.d_Sim_y+1,xk-self.d_Sim_z:xk+self.d_Sim_z+1],(1-TempDC)) # convert ins               
            print(".............. Done --%s seconds of simulation-" % (np.around((time.time() - time_fourthGird), decimals=2)))   
            if Plot==True:
                self.SavePlot(name=LocModel+"_{}_FourthGrid".format(Cicle),Level=4)   
        
        self.Values[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey,self.delta_z:self.delta_z+self.SGsizez,0] = np.copy(self.TrainingImage)
        tf.get_default_graph().finalize()
        gc.collect()
        sess.close()
    
        
    def Train_4ConvNets_BN_3D(self, Epochs, Num_samples, LocModel, LR):
        tf.reset_default_graph() # A brand new graph each run
        with tf.device('/gpu:0'):
            print("----Training Process--")
            start_time = time.time()
            Num_batch = int(self.largo/Num_samples)
            for ind_epoch in range(Epochs): 
                tf.reset_default_graph() 
                config = tf.ConfigProto(allow_soft_placement = True)
                config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 
                #config.gpu_options.allow_growth=True
                with tf.Session(config = config) as sess:
                    start_time_train = time.time()
                    
                    saver = tf.train.import_meta_graph(LocModel+'.meta')
                    saver.restore(sess, save_path=LocModel)
                    graph = tf.get_default_graph()
                    
                    x_image_a = graph.get_tensor_by_name("x_image_a:0")
                    x_image_b = graph.get_tensor_by_name("x_image_b:0")
                    x_image_c = graph.get_tensor_by_name("x_image_c:0")
                    x_image_d = graph.get_tensor_by_name("x_image_d:0")
                    InTrain = graph.get_tensor_by_name("InTrain:0")
                    y_a = graph.get_tensor_by_name("y_a:0") 
                    y_b = graph.get_tensor_by_name("y_b:0") 
                    y_c = graph.get_tensor_by_name("y_c:0") 
                    y_d = graph.get_tensor_by_name("y_d:0") 
                    learning_rate = graph.get_tensor_by_name("learning_rate:0")
           
        
                    cross_entropy_a = tf.get_collection("cross_entropy_a")[0]
                    cross_entropy_b = tf.get_collection("cross_entropy_b")[0]
                    cross_entropy_c = tf.get_collection("cross_entropy_c")[0]
                    cross_entropy_d = tf.get_collection("cross_entropy_d")[0]
                    #train_step = tf.get_collection("train_step")[0]
                    train_step_a = tf.get_collection("train_step_a")[0]
                    train_step_b = tf.get_collection("train_step_b")[0]
                    train_step_c = tf.get_collection("train_step_c")[0]
                    train_step_d = tf.get_collection("train_step_d")[0]                    
                    accuracy_a = tf.get_collection("accuracy_a")[0]
                    accuracy_b = tf.get_collection("accuracy_b")[0]
                    accuracy_c = tf.get_collection("accuracy_c")[0]
                    accuracy_d = tf.get_collection("accuracy_d")[0]
                     
                
                    # ###############################################################
                    # Data Training setting
                    # ###############################################################    
                    ind_ti, ind_tj = 2*Num_samples, 3*Num_samples
                    Train_Acc_Samples_a = self.InputFirstGrid(ti=ind_ti, tj=ind_tj)
                    Train_Acc_Samples_b = self.InputLevelGrid(ti=ind_ti, tj=ind_tj, Level=2)
                    Train_Acc_Samples_c = self.InputLevelGrid(ti=ind_ti, tj=ind_tj, Level=3)
                    Train_Acc_Samples_d = self.InputLevelGrid(ti=ind_ti, tj=ind_tj, Level=4)
                    Train_Acc_Target = self.HotspotCenterValue0n1(self.Target(ti=ind_ti, tj=ind_tj))
   
                    feed_acc_training_a = {x_image_a: Train_Acc_Samples_a, y_a: Train_Acc_Target, InTrain:False}  
                    feed_acc_training_b = {x_image_b: Train_Acc_Samples_b, y_b: Train_Acc_Target, InTrain:False} 
                    feed_acc_training_c = {x_image_c: Train_Acc_Samples_c, y_c: Train_Acc_Target, InTrain:False}  
                    feed_acc_training_d = {x_image_d: Train_Acc_Samples_d, y_d: Train_Acc_Target, InTrain:False}  
                    
                    print("Model size:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
                    for ind_batch in range(Num_batch): 
                        ind_bi, ind_bj = (ind_batch)*Num_samples, (ind_batch+1)*Num_samples
                        Samples_in_batch_a = self.InputFirstGrid(ti=ind_bi, tj=ind_bj)
                        Samples_in_batch_b = self.InputLevelGrid(ti=ind_bi, tj=ind_bj, Level=2)
                        Samples_in_batch_c = self.InputLevelGrid(ti=ind_bi, tj=ind_bj, Level=3)
                        Samples_in_batch_d = self.InputLevelGrid(ti=ind_bi, tj=ind_bj, Level=4)
                        Target_in_batch =  self.HotspotCenterValue0n1(self.Target(ti=ind_bi, tj=ind_bj))
                        sess.run(train_step_a, feed_dict={x_image_a: Samples_in_batch_a, x_image_b: Samples_in_batch_b, x_image_c: Samples_in_batch_c, x_image_d: Samples_in_batch_d, y_a: Target_in_batch, y_b: Target_in_batch, y_c: Target_in_batch, y_d: Target_in_batch, learning_rate: LR, InTrain:True})
                        sess.run(train_step_b, feed_dict={x_image_a: Samples_in_batch_a, x_image_b: Samples_in_batch_b, x_image_c: Samples_in_batch_c, x_image_d: Samples_in_batch_d, y_a: Target_in_batch, y_b: Target_in_batch, y_c: Target_in_batch, y_d: Target_in_batch, learning_rate: LR, InTrain:True})
                        sess.run(train_step_c, feed_dict={x_image_a: Samples_in_batch_a, x_image_b: Samples_in_batch_b, x_image_c: Samples_in_batch_c, x_image_d: Samples_in_batch_d, y_a: Target_in_batch, y_b: Target_in_batch, y_c: Target_in_batch, y_d: Target_in_batch, learning_rate: LR, InTrain:True})
                        sess.run(train_step_d, feed_dict={x_image_a: Samples_in_batch_a, x_image_b: Samples_in_batch_b, x_image_c: Samples_in_batch_c, x_image_d: Samples_in_batch_d, y_a: Target_in_batch, y_b: Target_in_batch, y_c: Target_in_batch, y_d: Target_in_batch, learning_rate: LR, InTrain:True})
                  
                    # Calculating errors and entropy
                    Cross_entro_a = sess.run(cross_entropy_a, feed_dict = feed_acc_training_a)
                    Cross_entro_b = sess.run(cross_entropy_b, feed_dict = feed_acc_training_b)
                    Cross_entro_c = sess.run(cross_entropy_c, feed_dict = feed_acc_training_c)
                    Cross_entro_d = sess.run(cross_entropy_d, feed_dict = feed_acc_training_d)
                    train_accuracy_a = sess.run(accuracy_a, feed_dict = feed_acc_training_a) 
                    train_accuracy_b = sess.run(accuracy_b, feed_dict = feed_acc_training_b) 
                    train_accuracy_c = sess.run(accuracy_c, feed_dict = feed_acc_training_c) 
                    train_accuracy_d = sess.run(accuracy_d, feed_dict = feed_acc_training_d) 
                    print("Epoch {} of {}.".format((ind_epoch+1), Epochs))                
                    print("1st grid: {} % training error. {} cross entropy.".format(np.around(100*(1-train_accuracy_a), decimals=2), np.around(Cross_entro_a*100, decimals=2)))                
                    print("2nd grid: {} % training error. {} cross entropy.".format(np.around(100*(1-train_accuracy_b), decimals=2), np.around(Cross_entro_b*100, decimals=2)))                
                    print("3rd grid: {} % training error. {} cross entropy.".format(np.around(100*(1-train_accuracy_c), decimals=2), np.around(Cross_entro_c*100, decimals=2)))                
                    print("4th grid: {} % training error. {} cross entropy.".format(np.around(100*(1-train_accuracy_d), decimals=2), np.around(Cross_entro_d*100, decimals=2)))                
                    print("Total cross entropy = {}".format(np.around((Cross_entro_a + Cross_entro_b + Cross_entro_c + Cross_entro_d)*100, decimals=2)))                
                 
                    print("")
                    saver.save(sess, LocModel)   
                    tf.get_default_graph().finalize()
        
            print("--%s seconds of training-" % (np.around((time.time() - start_time_train), decimals=2)))      
        gc.collect()
        return print("--%s minutes of training-" % ((time.time() - start_time)/60))  
        
    # ######### Final
# ### No LBP With BN

def CreateGraph_4ConvNets_4HL_NFeaConv_wdnhxwdnh_BN_3D(HyperPar, LocModel='Models/Default/FeatMaps'):
    with tf.device('/gpu:0'):
        
        # Global Environment
        Search_x, Search_y, Search_z = int(HyperPar[3]), int(HyperPar[4]), int(HyperPar[5]) #odd number Search Grid
        IPsizex, IPsizey, IPsizez = int(HyperPar[6]), int(HyperPar[7]), int(HyperPar[8])    #odd number Inner Pattern        
        NFeaConv1 = int(HyperPar[13]) 
        NFeaConv2 = int(HyperPar[13]) 
        NFeaConv3 = int(HyperPar[13]) 
        NFeaConv4 = int(HyperPar[13]) 
    
        NumCategories = int(HyperPar[14])  # (2+1)
        InputLvl_a = 1
        InputLvl_b = 2
        InputLvl_c = 3
        InputLvl_d = 4
        
        NumNodesFC = int(HyperPar[11])
        
        wdnh = int(HyperPar[12])
    
        # ###############################################################
        # Defining the architecture
        # ###############################################################
        tf.reset_default_graph() 
        x_image_a = tf.placeholder(tf.float32, shape=[None, Search_x, Search_y, Search_z, InputLvl_a], name="x_image_a") # Input
        y_a = tf.placeholder(tf.float32, shape=[None, IPsizex, IPsizey,IPsizez,  NumCategories], name="y_a") # Output
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        InTrain = tf.placeholder(tf.bool, name="InTrain")
    
        W1a = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,InputLvl_a,NFeaConv1], stddev=0.1), dtype=tf.float32, name="W1a")
        b1a = tf.Variable(tf.constant(0.1, shape=[Search_x, Search_y,Search_z,NFeaConv1]), dtype=tf.float32, name="b1a")
        conv1a = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(x_image_a, W1a, strides=[1,1,1,1,1], padding='SAME')+b1a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv1a")
        conv1a_pool = tf.nn.max_pool3d(conv1a, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv1a_pool")
        
        W2a = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv1,NFeaConv2], stddev=0.1), dtype=tf.float32, name="W2a")
        b2a = tf.Variable(tf.constant(0.1, shape=[conv1a_pool.shape[1],conv1a_pool.shape[2],conv1a_pool.shape[3],NFeaConv2]), dtype=tf.float32, name="b2a")
        conv2a = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv1a_pool, W2a, strides=[1,1,1,1,1], padding='SAME')+b2a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv2a")
        conv2a_pool = tf.nn.max_pool3d(conv2a, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv2a_pool")
        
        W3a = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv2,NFeaConv3], stddev=0.1), dtype=tf.float32, name="W3a")
        b3a = tf.Variable(tf.constant(0.1, shape=[conv2a_pool.shape[1],conv2a_pool.shape[2],conv2a_pool.shape[3],NFeaConv3]), dtype=tf.float32, name="b3a")
        conv3a = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv2a_pool, W3a, strides=[1,1,1,1,1], padding='SAME')+b3a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv3a")
        conv3a_pool = tf.nn.max_pool3d(conv3a, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv3a_pool")    
        
        W4a = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv3,NFeaConv4], stddev=0.1), dtype=tf.float32, name="W4a")
        b4a = tf.Variable(tf.constant(0.1, shape=[conv3a_pool.shape[1],conv3a_pool.shape[2],conv3a_pool.shape[3],NFeaConv4]), dtype=tf.float32, name="b4a")
        conv4a = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv3a_pool, W4a, strides=[1,1,1,1,1], padding='SAME')+b4a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv4a")
        conv4a_pool = tf.nn.max_pool3d(conv4a, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv4a_pool")         
    
        conv4a_flat = tf.reshape(conv4a_pool, [-1,int(conv4a_pool.shape[1])*int(conv4a_pool.shape[2])*int(conv4a_pool.shape[3])*int(conv4a_pool.shape[4])], name="conv4a_flat")
        WF1a = tf.Variable(tf.truncated_normal(shape=[int(conv4a_flat.get_shape()[1]), NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF1a")    
        bFC1a = tf.Variable(tf.constant(0.1, shape=[NumNodesFC]), dtype=tf.float32, name="bFC1a")  
        full_1a = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(conv4a_flat, WF1a)+bFC1a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_1a")
        
        WF2a = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF2a")
        bFC2a = tf.Variable(tf.constant(0.1, shape=[NumNodesFC]), dtype=tf.float32, name="bFC2a")                
        full_2a = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(full_1a, WF2a)+bFC2a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_2a")   
        
        WF3a = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, IPsizez*IPsizex*IPsizey*NumCategories], stddev=0.1), dtype=tf.float32, name="WF3a")
        bFC3a = tf.Variable(tf.constant(0.1, shape=[WF3a.shape[1]]), dtype=tf.float32, name="bFC3a")                        
        full_3a = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(full_2a, WF3a)+bFC3a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_3a")          
    
        y_conv_a = tf.reshape(full_3a, [-1,int(IPsizex), int(IPsizey),int(IPsizez), NumCategories], name="y_conv_a")
        
        # ##########################################################################  
        
        x_image_b = tf.placeholder(tf.float32, shape=[None,Search_x, Search_y,  Search_z, InputLvl_b], name="x_image_b") # Input
        y_b = tf.placeholder(tf.float32, shape=[None, IPsizex, IPsizey,IPsizez, NumCategories], name="y_b") # Output
    
    
        W1b = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,InputLvl_b,NFeaConv1], stddev=0.1), dtype=tf.float32, name="W1b")
        b1b = tf.Variable(tf.constant(0.1, shape=[Search_x, Search_y,Search_z,NFeaConv1]), dtype=tf.float32, name="b1b")
        conv1b = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(x_image_b, W1b, strides=[1,1,1,1,1], padding='SAME')+b1b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv1b")        
        conv1b_pool = tf.nn.max_pool3d(conv1b, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv1b_pool")
        
        W2b = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv1,NFeaConv2], stddev=0.1), dtype=tf.float32, name="W2b")
        b2b = tf.Variable(tf.constant(0.1, shape=[conv1b_pool.shape[1],conv1b_pool.shape[2],conv1b_pool.shape[3],NFeaConv2]), dtype=tf.float32, name="b2b")
        conv2b = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv1b_pool, W2b, strides=[1,1,1,1,1], padding='SAME')+b2b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv2b")        
        conv2b_pool = tf.nn.max_pool3d(conv2b, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv2b_pool")  
        
        W3b = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv2,NFeaConv3], stddev=0.1), dtype=tf.float32, name="W3b")
        b3b = tf.Variable(tf.constant(0.1, shape=[conv2b_pool.shape[1],conv2b_pool.shape[2],conv2b_pool.shape[3],NFeaConv3]), dtype=tf.float32, name="b3b")
        conv3b = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv2b_pool, W3b, strides=[1,1,1,1,1], padding='SAME')+b3b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv3b")
        conv3b_pool = tf.nn.max_pool3d(conv3b, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv3b_pool")           
    
        W4b = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv3,NFeaConv4], stddev=0.1), dtype=tf.float32, name="W4b")
        b4b = tf.Variable(tf.constant(0.1, shape=[conv3b_pool.shape[1],conv3b_pool.shape[2],conv3b_pool.shape[3],NFeaConv4]), dtype=tf.float32, name="b4b")
        conv4b = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv3b_pool, W4b, strides=[1,1,1,1,1], padding='SAME')+b4b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv4b")
        conv4b_pool = tf.nn.max_pool3d(conv4b, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv4b_pool")         
    
        conv4b_flat = tf.reshape(conv4b_pool, [-1,int(conv4b_pool.shape[1])*int(conv4b_pool.shape[2])*int(conv4b_pool.shape[3])*int(conv4b_pool.shape[4])], name="conv4b_flat")
        WF1b = tf.Variable(tf.truncated_normal(shape=[int(conv4b_flat.get_shape()[1]), NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF1b")
        bFC1b = tf.Variable(tf.constant(0.1, shape=[NumNodesFC]), dtype=tf.float32, name="bFC1b")        
        full_1b = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(conv4b_flat, WF1b)+bFC1b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_1b")
        
        WF2b = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF2b")
        bFC2b = tf.Variable(tf.constant(0.1, shape=[NumNodesFC]), dtype=tf.float32, name="bFC2b")        
        full_2b = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(full_1b, WF2b)+bFC2b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_2b")   
        
        WF3b = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, IPsizez*IPsizex*IPsizey*NumCategories], stddev=0.1), dtype=tf.float32, name="WF3b")
        bFC3b = tf.Variable(tf.constant(0.1, shape=[WF3b.shape[1]]), dtype=tf.float32, name="bFC3b")        
        full_3b = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(full_2b, WF3b)+bFC3b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_3b")          
    
        y_conv_b = tf.reshape(full_3b, [-1,int(IPsizex), int(IPsizey),int(IPsizez), NumCategories], name="y_conv_b")   
        
        # ##########################################################################  
    
        x_image_c = tf.placeholder(tf.float32, shape=[None, Search_x, Search_y, Search_z, InputLvl_c], name="x_image_c") # Input
        y_c = tf.placeholder(tf.float32, shape=[None, IPsizex, IPsizey, IPsizez, NumCategories], name="y_c") # Output
    
    
        W1c = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,InputLvl_c,NFeaConv1], stddev=0.1), dtype=tf.float32, name="W1c")
        b1c = tf.Variable(tf.constant(0.1, shape=[Search_x, Search_y,Search_z,NFeaConv1]), dtype=tf.float32, name="b1c")
        conv1c = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(x_image_c, W1c, strides=[1,1,1,1,1], padding='SAME')+b1c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv1c")           
        conv1c_pool = tf.nn.max_pool3d(conv1c, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv1c_pool")
        
        W2c = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv1,NFeaConv2], stddev=0.1), dtype=tf.float32, name="W2c")
        b2c = tf.Variable(tf.constant(0.1, shape=[conv1c_pool.shape[1],conv1c_pool.shape[2],conv1c_pool.shape[3],NFeaConv2]), dtype=tf.float32, name="b2c")
        conv2c = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv1c_pool, W2c, strides=[1,1,1,1,1], padding='SAME')+b2c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv2c")        
        conv2c_pool = tf.nn.max_pool3d(conv2c, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv2c_pool")        
    
        W3c = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv2,NFeaConv3], stddev=0.1), dtype=tf.float32, name="W3c")
        b3c = tf.Variable(tf.constant(0.1, shape=[conv2c_pool.shape[1],conv2c_pool.shape[2],conv2c_pool.shape[3],NFeaConv3]), dtype=tf.float32, name="b3c")
        conv3c = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv2c_pool, W3c, strides=[1,1,1,1,1], padding='SAME')+b3c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv3c")
        conv3c_pool = tf.nn.max_pool3d(conv3c, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv3c_pool")           
    
        W4c = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv3,NFeaConv4], stddev=0.1), dtype=tf.float32, name="W4c")
        b4c = tf.Variable(tf.constant(0.1, shape=[conv3c_pool.shape[1],conv3c_pool.shape[2],conv3c_pool.shape[3],NFeaConv4]), dtype=tf.float32, name="b4c")
        conv4c = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv3c_pool, W4c, strides=[1,1,1,1,1], padding='SAME')+b4c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv4c")
        conv4c_pool = tf.nn.max_pool3d(conv4c, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv4c_pool")         
    
        conv4c_flat = tf.reshape(conv4c_pool, [-1,int(conv4c_pool.shape[1])*int(conv4c_pool.shape[2])*int(conv4c_pool.shape[3])*int(conv4c_pool.shape[4])], name="conv4c_flat")
        WF1c = tf.Variable(tf.truncated_normal(shape=[int(conv4c_flat.get_shape()[1]), NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF1c")
        bFC1c = tf.Variable(tf.constant(0.1, shape=[NumNodesFC]), dtype=tf.float32, name="bFC1c")        
        full_1c = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(conv4c_flat, WF1c)+bFC1c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_1c")
        
        WF2c = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF2c")
        bFC2c = tf.Variable(tf.constant(0.1, shape=[NumNodesFC]), dtype=tf.float32, name="bFC2c")        
        full_2c = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(full_1c, WF2c)+bFC2c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_2c")   
        
        WF3c = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, IPsizez*IPsizex*IPsizey*NumCategories], stddev=0.1), dtype=tf.float32, name="WF3c")  
        bFC3c = tf.Variable(tf.constant(0.1, shape=[WF3c.shape[1]]), dtype=tf.float32, name="bFC3c")        
        full_3c = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(full_2c, WF3c)+bFC3c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_3c")           
        
    
        y_conv_c = tf.reshape(full_3c, [-1,int(IPsizex), int(IPsizey), int(IPsizez), NumCategories], name="y_conv_c")  
        
        # ##########################################################################     
        
        x_image_d = tf.placeholder(tf.float32, shape=[None, Search_x, Search_y, Search_z,  InputLvl_d], name="x_image_d") # Input
        y_d = tf.placeholder(tf.float32, shape=[None, IPsizex, IPsizey, IPsizez, NumCategories], name="y_d") # Output
    
    
        W1d = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,InputLvl_d,NFeaConv1], stddev=0.1), dtype=tf.float32, name="W1d")
        b1d = tf.Variable(tf.constant(0.1, shape=[Search_x, Search_y,Search_z,NFeaConv1]), dtype=tf.float32, name="b1d")
        conv1d = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(x_image_d, W1d, strides=[1,1,1,1,1], padding='SAME')+b1d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv1d")           
        conv1d_pool = tf.nn.max_pool3d(conv1d, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv1d_pool")
        
        W2d = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv1,NFeaConv2], stddev=0.1), dtype=tf.float32, name="W2d")
        b2d = tf.Variable(tf.constant(0.1, shape=[conv1d_pool.shape[1],conv1d_pool.shape[2],conv1d_pool.shape[3],NFeaConv2]), dtype=tf.float32, name="b2d")
        conv2d = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv1d_pool, W2d, strides=[1,1,1,1,1], padding='SAME')+b2d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv2d")        
        conv2d_pool = tf.nn.max_pool3d(conv2d, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv2d_pool")      
    
        W3d = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv2,NFeaConv3], stddev=0.1), dtype=tf.float32, name="W3d")
        b3d = tf.Variable(tf.constant(0.1, shape=[conv2d_pool.shape[1],conv2d_pool.shape[2],conv2d_pool.shape[3],NFeaConv3]), dtype=tf.float32, name="b3d")
        conv3d = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv2d_pool, W3d, strides=[1,1,1,1,1], padding='SAME')+b3d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv3d")
        conv3d_pool = tf.nn.max_pool3d(conv3d, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv3d_pool")           
    
        W4d = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,wdnh,NFeaConv3,NFeaConv4], stddev=0.1), dtype=tf.float32, name="W4d")
        b4d = tf.Variable(tf.constant(0.1, shape=[conv3d_pool.shape[1],conv3d_pool.shape[2],conv3d_pool.shape[3],NFeaConv4]), dtype=tf.float32, name="b4d")
        conv4d = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv3d(conv3d_pool, W4d, strides=[1,1,1,1,1], padding='SAME')+b4d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv4d")
        conv4d_pool = tf.nn.max_pool3d(conv4d, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name="conv4d_pool")         
    
        conv4d_flat = tf.reshape(conv4d_pool, [-1,int(conv4d_pool.shape[1])*int(conv4d_pool.shape[2])*int(conv4d_pool.shape[3])*int(conv4d_pool.shape[4])], name="conv4d_flat")
        WF1d = tf.Variable(tf.truncated_normal(shape=[int(conv4d_flat.get_shape()[1]), NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF1d")
        bFC1d = tf.Variable(tf.constant(0.1, shape=[NumNodesFC]), dtype=tf.float32, name="bFC1d")        
        full_1d = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(conv4d_flat, WF1d)+bFC1d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_1d")   
           
        WF2d = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF2d")
        bFC2d = tf.Variable(tf.constant(0.1, shape=[NumNodesFC]), dtype=tf.float32, name="bFC2d")        
        full_2d = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(full_1d, WF2d)+bFC2d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_2d")   
        
        WF3d = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, IPsizex*IPsizey*IPsizez*NumCategories], stddev=0.1), dtype=tf.float32, name="WF3d")
        bFC3d = tf.Variable(tf.constant(0.1, shape=[WF3d.shape[1]]), dtype=tf.float32, name="bFC3d")        
        full_3d = tf.nn.relu(tf.layers.batch_normalization((tf.matmul(full_2d, WF3d)+bFC3d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_3d")           
    
        y_conv_d = tf.reshape(full_3d, [-1,int(IPsizex), int(IPsizey), int(IPsizez), NumCategories], name="y_conv_d")           
        # ##########################################################################     
        softmax_y_conv_a = tf.nn.softmax(y_conv_a,axis=4,name="softmax_y_conv_a")[0]
        softmax_y_conv_b = tf.nn.softmax(y_conv_b,axis=4,name="softmax_y_conv_b")[0]
        softmax_y_conv_c = tf.nn.softmax(y_conv_c,axis=4,name="softmax_y_conv_c")[0]
        softmax_y_conv_d = tf.nn.softmax(y_conv_d,axis=4,name="softmax_y_conv_d")[0]
    
        tf.add_to_collection("softmax_y_conv_a", softmax_y_conv_a)
        tf.add_to_collection("softmax_y_conv_b", softmax_y_conv_b)
        tf.add_to_collection("softmax_y_conv_c", softmax_y_conv_c)    
        tf.add_to_collection("softmax_y_conv_d", softmax_y_conv_d)    
        
        argmax_y_conv_a = tf.argmax(y_conv_a,axis=4,name="argmax_y_conv_a")[0]
        argmax_y_conv_b = tf.argmax(y_conv_b,axis=4,name="argmax_y_conv_b")[0]
        argmax_y_conv_c = tf.argmax(y_conv_c,axis=4,name="argmax_y_conv_c")[0]
        argmax_y_conv_d = tf.argmax(y_conv_d,axis=4,name="argmax_y_conv_d")[0]
    
        tf.add_to_collection("argmax_y_conv_a", argmax_y_conv_a)
        tf.add_to_collection("argmax_y_conv_b", argmax_y_conv_b)
        tf.add_to_collection("argmax_y_conv_c", argmax_y_conv_c)    
        tf.add_to_collection("argmax_y_conv_d", argmax_y_conv_d)         
    
        cross_entropy_a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_conv_a,labels = y_a), name="cross_entropy_a")   
        cross_entropy_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_conv_b,labels = y_b), name="cross_entropy_b")  
        cross_entropy_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_conv_c,labels = y_c), name="cross_entropy_c") 
        cross_entropy_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_conv_d,labels = y_d), name="cross_entropy_d") 
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step_a = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_a)
            train_step_b = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_b)
            train_step_c = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_c)
            train_step_d = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_d)
        tf.add_to_collection("train_step_a", train_step_a)
        tf.add_to_collection("train_step_b", train_step_b) 
        tf.add_to_collection("train_step_c", train_step_c) 
        tf.add_to_collection("train_step_d", train_step_d)             
        
        correct_prediction_a = tf.equal(tf.argmax(y_conv_a,4), tf.argmax(y_a,4), name="correct_prediction_a")
        correct_prediction_b = tf.equal(tf.argmax(y_conv_b,4), tf.argmax(y_b,4), name="correct_prediction_b")
        correct_prediction_c = tf.equal(tf.argmax(y_conv_c,4), tf.argmax(y_c,4), name="correct_prediction_c")
        correct_prediction_d = tf.equal(tf.argmax(y_conv_d,4), tf.argmax(y_d,4), name="correct_prediction_d")
        accuracy_a = tf.reduce_mean(tf.cast(correct_prediction_a, tf.float32), name="accuracy_a")
        accuracy_b = tf.reduce_mean(tf.cast(correct_prediction_b, tf.float32), name="accuracy_b")
        accuracy_c = tf.reduce_mean(tf.cast(correct_prediction_c, tf.float32), name="accuracy_c")
        accuracy_d = tf.reduce_mean(tf.cast(correct_prediction_d, tf.float32), name="accuracy_d")
        tf.add_to_collection("cross_entropy_a", cross_entropy_a)
        tf.add_to_collection("cross_entropy_b", cross_entropy_b)
        tf.add_to_collection("cross_entropy_c", cross_entropy_c)
        tf.add_to_collection("cross_entropy_d", cross_entropy_d)
        tf.add_to_collection("accuracy_a", accuracy_a)
        tf.add_to_collection("accuracy_b", accuracy_b)
        tf.add_to_collection("accuracy_c", accuracy_c)
        tf.add_to_collection("accuracy_d", accuracy_d)
        print("Model size:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
     
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, LocModel)
    return print("--Graph created--") 



