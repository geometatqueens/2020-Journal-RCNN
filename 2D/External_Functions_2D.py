"""The present code is the Version 1.0 of the RCNN approach to perform MPS 
in 2D for categorical variables. It has been developed by S. Avalos and J. Ortiz in the
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


class Grid():
    
    def __init__(self, HyperPar, DBname, Lvl, Training=True):
        self.SGsizex = int(HyperPar[0]) # Simulation grid X
        self.SGsizey = int(HyperPar[1]) # Simulation grid Y
        self.Search_x = int(HyperPar[2]) # odd number search grid X
        self.Search_y = int(HyperPar[3]) # odd number search grid Y
        self.IPsizex = int(HyperPar[4]) # odd inner pattern search grid X   
        self.IPsizey = int(HyperPar[5]) # odd inner pattern search grid Y   
        self.PercDC = np.around(HyperPar[6]/1000, decimals=3) # odd inner pattern search grid Y
        self.MinDC = int(HyperPar[7]) # Minimum data conditioning 
        self.NumGrid = int(HyperPar[11]) # Number of multi grids
        self.NumCateg = int(HyperPar[12]) # Number of categories
        
        self.largo = int((HyperPar[0]*(HyperPar[1]))) 
        self.delta_x = int((HyperPar[2]-1)/2)
        self.delta_y = int((HyperPar[3]-1)/2)
        self.d_Sim_x = int((HyperPar[4]-1)/2) 
        self.d_Sim_y = int((HyperPar[5]-1)/2)   
        self.Level = Lvl
        self.Values = np.zeros([self.SGsizex+2*self.delta_x,self.SGsizey+2*self.delta_y,self.Level])
        self.DataConditioning = np.ones([self.SGsizex+2*self.delta_x,self.SGsizey+2*self.delta_y,self.Level])
        self.TrainingImage = np.loadtxt(DBname).reshape(self.SGsizex,self.SGsizey)
        self.TrainingImage_Padding = np.zeros([self.SGsizex+2*self.delta_x,self.SGsizey+2*self.delta_y])
        self.TrainingImage_Padding[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey] = self.TrainingImage
        for ind0 in range(self.Level):
            self.Values[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey,ind0] = np.copy(self.TrainingImage)          
        
        if Training == True:
            G = np.random.random((self.Values.shape[0],self.Values.shape[1]))
            for indx in range(self.Values.shape[0]):
                for indy in range(self.Values.shape[1]):
                    if G[indx][indy] < self.PercDC:
                        G[indx][indy] = 1
                    else:
                        G[indx][indy] = 0
            for ind0 in range(self.Values.shape[2]):
                self.Values[:,:,ind0] = np.multiply(self.Values[:,:,ind0],G)

        for indx in range(self.DataConditioning.shape[0]):
            for indy in range(self.DataConditioning.shape[1]):
                if self.Values[indx][indy][0] != 0:
                    self.DataConditioning[indx,indy] = 0 # 1 are NON conditional, 0  ARE conditionals
    
    def Target(self):
        TargetMatrix = np.zeros(((self.SGsizex*self.SGsizey),self.IPsizex,self.IPsizey,1))  
        ind0 = 0
        for indx in range(self.SGsizex):
            for indy in range(self.SGsizey):
                TargetMatrix[ind0,:,:,0] = self.TrainingImage_Padding[indx:indx + self.IPsizex, indy:indy + self.IPsizey]
                ind0 += 1        
        return TargetMatrix
    
    def InputFirstGrid(self):
        RawInput = np.zeros(((self.SGsizex*self.SGsizey),self.Search_x,self.Search_y,1))  
        ind0 = 0
        for indx in range(self.SGsizex):
            for indy in range(self.SGsizey):
                RawInput[ind0,:,:,0] = self.Values[indx:indx + self.Search_x, indy:indy + self.Search_y,0]
                ind0 += 1        
        return RawInput   
       
    def InputLevelGrid(self,Level):
        RawInput = np.zeros(((self.SGsizex*self.SGsizey),self.Search_x,self.Search_y,Level))  
        ind0 = 0
        for indx in range(self.SGsizex):
            for indy in range(self.SGsizey):
                for nivel in range(Level):
                    RawInput[ind0,:,:,nivel] = self.Values[indx:indx + self.Search_x, indy:indy + self.Search_y,nivel]
                ind0 += 1        
        return RawInput  
    
    def HotspotCenterValue0n1(self, CenterValueInnerGrid):
        Lar = CenterValueInnerGrid.shape[0]
        A = np.zeros([Lar, self.IPsizex, self.IPsizey,self.NumCateg])
        for ind0 in range(Lar):
            for indx in range(self.IPsizex):
                for indy in range(self.IPsizey):
                    A[ind0][indx][indy][int(CenterValueInnerGrid[ind0][indx][indy]-1)] = 1
        return A    
    

    def DC_Matrix(self):
        Matrix = np.copy(self.DataConditioning[:,:,0])
        return Matrix
    
    def RandomPathOLD(self):
        A = np.ones((self.SGsizex)*(self.SGsizey))
        count = 0
        for indx in range(self.SGsizex):
            for indy in range(self.SGsizey):
                A[count] = indx*10000 + indy    
                count += 1
        np.random.shuffle(A)
        return A  
    
    def RandomPath(self):
        d = dict()
        for ind_g in range(self.NumGrid):
            amount_x = int((self.SGsizex+2*self.delta_x-self.Search_x*2**ind_g)/(2**ind_g)+1)
            amount_y = int((self.SGsizey+2*self.delta_y-self.Search_y*2**ind_g)/(2**ind_g)+1)            
            Lar = amount_x*amount_y
            A = np.ones((Lar,2))#,dtype=int32)
            count = 0  
            for indx in range(amount_x):
                for indy in range(amount_y):
                    A[count,0], A[count,1] = (indx + self.delta_x)*2**ind_g, (indy + self.delta_y)*2**ind_g
                    count += 1
            np.random.shuffle(A)
            d[str(ind_g)] = A
        return d          
    
    def SavePlot(self,name,Level):
        A = str(name) + ".png"
        fig,ax = plt.subplots()
        colors1 = plt.cm.Paired(np.linspace(0, 1, 128))
        colors2 = plt.cm.binary(np.linspace(0, 1, 128))
        
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)    
        pa = ax.imshow(self.Values[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey,Level], cmap=plt.cm.rainbow_r,  vmin=1, vmax=self.NumCateg)
        cba = plt.colorbar(pa,shrink=0.5, ticks=[1, self.NumCateg]) 
        cba.set_clim(0, self.NumCateg)
        fig.savefig(A)
        return plt.close(fig)     
    
    def SaveGrid(self,file="results.txt"):
        with open(file, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter = '\t')
            for indx in range(self.SGsizex):
                for indy in range(self.SGsizey):
                    b = []
                    for level in range(self.Level):
                        b += [int(self.Values[self.delta_x+indx][self.delta_y+indy][level])]
                    spamwriter.writerow(b)
    # #### Metrics
    def PixelError(self, lvl=0):
        num_zeros = ((self.Values[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey,lvl] - self.TrainingImage) == 0).sum()
        prop = np.around(((self.SGsizex*self.SGsizey - num_zeros)/(self.SGsizex*self.SGsizey))*100, decimals=2)           
        return prop
    
    def BinaryProportion(self, lvl=0, TI=False):
        if TI==False:
            num_ones = (self.Values[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey,lvl] == 1).sum()
            num_two = (self.Values[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey,lvl] == 2).sum()  
            prop = np.around((num_two/(num_ones + num_two))*100, decimals=2)
        else:
            num_ones = (self.TrainingImage == 1).sum()
            num_two = (self.TrainingImage == 2).sum()  
            prop = np.around((num_two/(num_ones + num_two))*100, decimals=2)            
        return prop
    
    def TI_LBP_2x2(self):
        LBP_TI = 0
        LHBPH_TI = np.zeros(16)
        for indx in range(self.SGsizex-1):
            for indy in range(self.SGsizey-1):
                TI_v_i = np.reshape(self.TrainingImage[indx:indx+2, indy:indy+2], 4) - 1  # transf to 0 - 1
                for j in range(4):
                    LBP_TI += TI_v_i[j]*(2**j) 
                #print(LBP_TI)
                LHBPH_TI[int(LBP_TI)] += 1
                LBP_TI = 0 
        return LHBPH_TI 
    
    def TI_LBP_3x3(self):
        LBP_TI = 0
        LHBPH_TI = np.zeros(512)
        for indx in range(self.SGsizex-2):
            for indy in range(self.SGsizey-2):
                TI_v_i = np.reshape(self.TrainingImage[indx:indx+3, indy:indy+3], 9) - 1  # transf to 0 - 1
                for j in range(9):
                    LBP_TI += TI_v_i[j]*(2**j) 
                #print(LBP_TI)
                LHBPH_TI[int(LBP_TI)] += 1
                LBP_TI = 0 
        return LHBPH_TI      
    
    def MSE_LBP_2x2(self, TI_LBPH, lvl=0, WithExtrems=True):
        LBP_SimGrid = 0
        LHBPH_SimGrid = np.zeros(16)        
        for indx in range(self.SGsizex-1):
            for indy in range(self.SGsizey-1):
                Value_v_i = np.reshape(self.Values[self.delta_x+indx:self.delta_x + indx +2, self.delta_y+indy:self.delta_y + indy +2, lvl], 4) - 1
                for j in range(4):
                    LBP_SimGrid += Value_v_i[j]*(2**j) 
                LHBPH_SimGrid[int(LBP_SimGrid)] += 1    
                LBP_SimGrid = 0
        if WithExtrems == True:
            Results = np.around(np.linalg.norm(TI_LBPH - LHBPH_SimGrid), decimals=2)
        if WithExtrems == False:
            Results = np.around(np.linalg.norm(TI_LBPH[1:15] - LHBPH_SimGrid[1:15]), decimals=2)
        return Results 
    
    def MSE_LBP_3x3(self, TI_LBPH, lvl=0, WithExtrems=True):
        LBP_SimGrid = 0
        LHBPH_SimGrid = np.zeros(512)        
        for indx in range(self.SGsizex-2):
            for indy in range(self.SGsizey-2):
                Value_v_i = np.reshape(self.Values[self.delta_x+indx:self.delta_x + indx + 3, self.delta_y+indy:self.delta_y + indy + 3, lvl], 9) - 1
                for j in range(9):
                    LBP_SimGrid += Value_v_i[j]*(2**j) 
                LHBPH_SimGrid[int(LBP_SimGrid)] += 1    
                LBP_SimGrid = 0
        if WithExtrems == True:
            Results = np.around(np.linalg.norm(TI_LBPH - LHBPH_SimGrid), decimals=2) 
        if WithExtrems == False:
            Results = np.around(np.linalg.norm(TI_LBPH[1:511] - LHBPH_SimGrid[1:511]), decimals=2)             
        return Results
    
    def Max_x(self, lvl=0):
        Max_X_final = 0
        Max_X = 0
        Value_v_0 = 0
        for indx in range(self.SGsizex): # indx goes vertical in the 2D array
            for indy in range(self.SGsizey): # indy goes horizontal in the 2D array
                Value_v_i = self.Values[self.delta_x+indx][self.delta_y+indy][lvl] - 1
                if Value_v_i == 1 and Value_v_0 == 0:
                    Max_X = 1
                    Value_v_0 = 1
                elif Value_v_i == 1 and Value_v_0 == 1:
                    Max_X += 1  
                    Value_v_0 = 1
                elif Value_v_i == 0 and Value_v_0 == 1:
                    if Max_X_final < Max_X:
                        Max_X_final = Max_X
                    Value_v_0 = 0
                    Max_X = 0
                else:
                    Value_v_0 = 0
                    Max_X = 0
            Max_X = 0
            Value_v_0 = 0
        return int(Max_X_final)  

        
    def Norm_1to2(self, Level):
        return (1 + (1/(self.NumCateg-1))*(self.Values[self.delta_x:self.SGsizex+self.delta_x,self.delta_y:self.delta_y+self.SGsizey,Level]-1))

    
    def Max_y(self, lvl=0):
        Max_Y_final = 0
        Max_Y = 0
        Value_v_0 = 0
        for indy in range(self.SGsizey): # indy goes horizontal in the 2D array
            for indx in range(self.SGsizex): # indx goes vertical in the 2D array
                Value_v_i = self.Values[self.delta_x+indx][self.delta_y+indy][lvl] - 1
                if Value_v_i == 1 and Value_v_0 == 0:
                    Max_Y = 1
                    Value_v_0 = 1
                elif Value_v_i == 1 and Value_v_0 == 1:
                    Max_Y += 1  
                    Value_v_0 = 1
                elif Value_v_i == 0 and Value_v_0 == 1:
                    if Max_Y_final < Max_Y:
                        Max_Y_final = Max_Y
                    Value_v_0 = 0
                    Max_Y = 0
                else:
                    Value_v_0 = 0
                    Max_Y = 0
            Max_Y = 0
            Value_v_0 = 0
        return int(Max_Y_final)       
        
    # 
    def Simulate_4ConvNets_BN(self,LocModel,Cicle,Plot=True):
        tf.reset_default_graph() # A brand new graph each run
        with tf.device('/gpu:0'):
               
            config = tf.ConfigProto(allow_soft_placement = True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 
            sess = tf.InteractiveSession(config = config) 
            saver = tf.train.import_meta_graph(LocModel+'.meta')
            saver.restore(sess, save_path=LocModel)
            graph = tf.get_default_graph()
    
            x_image_a = graph.get_tensor_by_name("x_image_a:0")
            x_image_b = graph.get_tensor_by_name("x_image_b:0")
            x_image_c = graph.get_tensor_by_name("x_image_c:0")
            x_image_d = graph.get_tensor_by_name("x_image_d:0")
            InTrain = graph.get_tensor_by_name("InTrain:0")   
            argmax_y_conv_a = graph.get_tensor_by_name("argmax_y_conv_a:0")
            argmax_y_conv_b = graph.get_tensor_by_name("argmax_y_conv_b:0")
            argmax_y_conv_c = graph.get_tensor_by_name("argmax_y_conv_c:0")
            argmax_y_conv_d = graph.get_tensor_by_name("argmax_y_conv_d:0")
            
    
            # ###############################################################
            # Simulate grids
            # ###############################################################  
            Path = self.RandomPath()
            if Plot==True:
                self.SavePlot(name=LocModel+"{}_0DC".format(Cicle),Level=0)
            print("First Sim Grid..")
            time_firstGird = time.time()
            for ind_g in range(self.NumGrid):
                ig = self.NumGrid - ind_g - 1
                DC_Position_a = self.DC_Matrix()
                for ind0 in range(Path[str(ig)].shape[0]):   
                    xi = int(Path[str(ig)][ind0,0])
                    xj = int(Path[str(ig)][ind0,1])
        
                    if DC_Position_a[xi][xj] == 1: # 
                        dd = 2**ig
                        ax, bx = xi-self.delta_x*2**ig, xi+self.delta_x*2**ig+1
                        ay, by = xj-self.delta_y*2**ig, xj+self.delta_y*2**ig+1
                        
                        cx, dx = xi-self.d_Sim_x*2**ig, xi+self.d_Sim_x*2**ig+1
                        cy, dy = xj-self.d_Sim_y*2**ig, xj+self.d_Sim_y*2**ig+1                           
                        if np.count_nonzero(self.Values[ax:bx:dd,ay:by:dd,0]) < self.MinDC:
                            temp_point = np.random.uniform()
                            if temp_point < 0.267:
                                self.Values[xi,xj,0] = 2
                            else: 
                                self.Values[xi,xj,0] = 1                        
                        SimInPut_a =  np.expand_dims(np.expand_dims(self.Values[ax:bx:dd,ay:by:dd,0],axis=0),axis=3)
                        IP_final = argmax_y_conv_a.eval(feed_dict={x_image_a: SimInPut_a, InTrain:False})[0] + 1 
    
                        # Assigning simulated data within random points inside inner pattern
                        self.Values[cx:dx:dd,cy:dy:dd,1] = np.multiply(self.Values[cx:dx:dd,cy:dy:dd,1], (1 - DC_Position_a[cx:dx:dd,cy:dy:dd])) # maintain only DC
                        TempDC = np.copy(np.multiply(DC_Position_a[cx:dx:dd,cy:dy:dd],np.random.randint(2, size=(self.IPsizex, self.IPsizey))))
                        TempDC[self.d_Sim_x][self.d_Sim_y] = 1
                        self.Values[cx:dx:dd,cy:dy:dd,1] += np.multiply(IP_final, TempDC) # Resimulate non conditional points 
                        DC_Position_a[cx:dx:dd,cy:dy:dd] = np.multiply(DC_Position_a[cx:dx:dd,cy:dy:dd],(1-TempDC)) # convert ins                    
                                      
            print(".............. Done --%s seconds of simulation-" % (np.around((time.time() - time_firstGird), decimals=2)))   
            if Plot==True:
                self.SavePlot(name=LocModel+"{}_1Grid".format(Cicle),Level=1)
            
            print("Second Sim Grid..")
            time_secondGird = time.time()
            for ind_g in range(self.NumGrid):
                ig = self.NumGrid - ind_g - 1            
                DC_Position_b = self.DC_Matrix()
                for ind0 in range(Path[str(ig)].shape[0]):   
                    xi = int(Path[str(ig)][ind0,0])
                    xj = int(Path[str(ig)][ind0,1])
        
                    if DC_Position_b[xi][xj] == 1:    
                        dd = 2**ig
                        ax, bx = xi-self.delta_x*2**ig, xi+self.delta_x*2**ig+1
                        ay, by = xj-self.delta_y*2**ig, xj+self.delta_y*2**ig+1
                        
                        cx, dx = xi-self.d_Sim_x*2**ig, xi+self.d_Sim_x*2**ig+1
                        cy, dy = xj-self.d_Sim_y*2**ig, xj+self.d_Sim_y*2**ig+1                           
                        
                        SimInPut_b =  np.expand_dims(self.Values[ax:bx:dd,ay:by:dd,[0,1]],axis=0)                        
                        IP_final = argmax_y_conv_b.eval(feed_dict={x_image_b: SimInPut_b, InTrain:False})[0] + 1 
                        
                        # Assigning simulated data within random points inside inner pattern
                        self.Values[cx:dx:dd,cy:dy:dd,2] = np.multiply(self.Values[cx:dx:dd,cy:dy:dd,2], (1 - DC_Position_b[cx:dx:dd,cy:dy:dd])) # maintain only DC
                        TempDC = np.copy(np.multiply(DC_Position_b[cx:dx:dd,cy:dy:dd],np.random.randint(2, size=(self.IPsizex, self.IPsizey))))
                        TempDC[self.d_Sim_x][self.d_Sim_y] = 1
                        self.Values[cx:dx:dd,cy:dy:dd,2] += np.multiply(IP_final, TempDC) # Resimulate non conditional points 
                        DC_Position_b[cx:dx:dd,cy:dy:dd] = np.multiply(DC_Position_b[cx:dx:dd,cy:dy:dd],(1-TempDC)) # convert ins  
                if Plot==True:
                    self.SavePlot(name=LocModel+"{}_2Grid_{}".format(Cicle,ind_g),Level=2)            
            print(".............. Done --%s seconds of simulation-" % (np.around((time.time() - time_secondGird), decimals=2)))   
            if Plot==True:
                self.SavePlot(name=LocModel+"{}_SecondGrid".format(Cicle),Level=2)
            
            print("Third Sim Grid..")
            time_thirdGird = time.time()
            for ind_g in range(self.NumGrid):
                ig = self.NumGrid - ind_g - 1    
                DC_Position_c = self.DC_Matrix()                
                for ind0 in range(Path[str(ig)].shape[0]):   
                    xi = int(Path[str(ig)][ind0,0])
                    xj = int(Path[str(ig)][ind0,1])
        
                    if DC_Position_c[xi][xj] == 1: # 
                        dd = 2**ig
                        ax, bx = xi-self.delta_x*2**ig, xi+self.delta_x*2**ig+1
                        ay, by = xj-self.delta_y*2**ig, xj+self.delta_y*2**ig+1
                        
                        cx, dx = xi-self.d_Sim_x*2**ig, xi+self.d_Sim_x*2**ig+1
                        cy, dy = xj-self.d_Sim_y*2**ig, xj+self.d_Sim_y*2**ig+1                           
                        
                        SimInPut_c =  np.expand_dims(self.Values[ax:bx:dd,ay:by:dd,[0,1,2]],axis=0)                        
                        IP_final = argmax_y_conv_c.eval(feed_dict={x_image_c: SimInPut_c, InTrain:False})[0] + 1 # 
                        
                        # Assigning simulated data within random points inside inner pattern
                        self.Values[cx:dx:dd,cy:dy:dd,3] = np.multiply(self.Values[cx:dx:dd,cy:dy:dd,3], (1 - DC_Position_c[cx:dx:dd,cy:dy:dd])) # maintain only DC
                        TempDC = np.copy(np.multiply(DC_Position_c[cx:dx:dd,cy:dy:dd],np.random.randint(2, size=(self.IPsizex, self.IPsizey))))
                        TempDC[self.d_Sim_x][self.d_Sim_y] = 1
                        self.Values[cx:dx:dd,cy:dy:dd,3] += np.multiply(IP_final, TempDC) # Resimulate non conditional points 
                        DC_Position_c[cx:dx:dd,cy:dy:dd] = np.multiply(DC_Position_c[cx:dx:dd,cy:dy:dd],(1-TempDC)) # convert ins               
            print(".............. Done --%s seconds of simulation-" % (np.around((time.time() - time_thirdGird), decimals=2)))   
            if Plot==True:
                self.SavePlot(name=LocModel+"{}_3Grid".format(Cicle),Level=3)
            
            print("Fourth Sim Grid..")
            time_fourthGird = time.time()
            for ind_g in range(self.NumGrid):
                ig = self.NumGrid - ind_g - 1            
                DC_Position_d = self.DC_Matrix()   
                for ind0 in range(Path[str(ig)].shape[0]):   
                    xi = int(Path[str(ig)][ind0,0])
                    xj = int(Path[str(ig)][ind0,1])
        
                    if DC_Position_d[xi][xj] == 1:  
                        dd = 2**ig
                        ax, bx = xi-self.delta_x*2**ig, xi+self.delta_x*2**ig+1
                        ay, by = xj-self.delta_y*2**ig, xj+self.delta_y*2**ig+1
                        
                        cx, dx = xi-self.d_Sim_x*2**ig, xi+self.d_Sim_x*2**ig+1
                        cy, dy = xj-self.d_Sim_y*2**ig, xj+self.d_Sim_y*2**ig+1                           

                        SimInPut_d =  np.expand_dims(self.Values[ax:bx:dd,ay:by:dd,[0,1,2,3]],axis=0)                        
                        IP_final = argmax_y_conv_d.eval(feed_dict={x_image_d: SimInPut_d, InTrain:False})[0] + 1 
                        
                        # Assigning simulated data within random points inside inner pattern
                        
                        self.Values[cx:dx:dd,cy:dy:dd,4] = np.multiply(self.Values[cx:dx:dd,cy:dy:dd,4], (1 - DC_Position_d[cx:dx:dd,cy:dy:dd])) # maintain only DC
                        TempDC = np.copy(np.multiply(DC_Position_d[cx:dx:dd,cy:dy:dd],np.random.randint(2, size=(self.IPsizex, self.IPsizey))))
                        TempDC[self.d_Sim_x][self.d_Sim_y] = 1
                        self.Values[cx:dx:dd,cy:dy:dd,4] += np.multiply(IP_final, TempDC) # Resimulate non conditional points 
                        DC_Position_d[cx:dx:dd,cy:dy:dd] = np.multiply(DC_Position_d[cx:dx:dd,cy:dy:dd],(1-TempDC)) # convert ins               
            print(".............. Done --%s seconds of simulation-" % (np.around((time.time() - time_fourthGird), decimals=2))) 
            if Plot==True:
                self.SavePlot(name=LocModel+"{}_4Grid".format(Cicle),Level=4)   
            
        tf.get_default_graph().finalize()
        gc.collect()
        sess.close()
    
    def Train_4ConvNets_BN(self, Epochs, Num_batch, LocModel, LR):
        tf.reset_default_graph() # A brand new graph each run
        with tf.device('/gpu:0'):
            print("----Training Process--")
            start_time = time.time()
            Num_samples = int(self.largo/Num_batch)
            for ind_epoch in range(Epochs): 
                tf.reset_default_graph() 
                config = tf.ConfigProto(allow_soft_placement = True)
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
                    # Train Data import
                    # ###############################################################  
                    TargetTraining = self.Target()
                    InputTraining_a = self.InputFirstGrid()
                    InputTraining_b = self.InputLevelGrid(Level=2)
                    InputTraining_c = self.InputLevelGrid(Level=3)
                    InputTraining_d = self.InputLevelGrid(Level=4)
                    
                    # ###############################################################
                    # Data Training setting
                    # ###############################################################    
                    Train_Acc_Samples_a = InputTraining_a[2*Num_samples:2*Num_samples+1500,:,:,:]
                    Train_Acc_Samples_b = InputTraining_b[2*Num_samples:2*Num_samples+1500,:,:,:]
                    Train_Acc_Samples_c = InputTraining_c[2*Num_samples:2*Num_samples+1500,:,:,:]
                    Train_Acc_Samples_d = InputTraining_d[2*Num_samples:2*Num_samples+1500,:,:,:]
                    Train_Acc_Target = self.HotspotCenterValue0n1(TargetTraining[2*Num_samples:2*Num_samples+1500,:,:,0])
   
                    feed_acc_training_a = {x_image_a: Train_Acc_Samples_a, y_a: Train_Acc_Target, InTrain:False}  
                    feed_acc_training_b = {x_image_b: Train_Acc_Samples_b, y_b: Train_Acc_Target, InTrain:False} 
                    feed_acc_training_c = {x_image_c: Train_Acc_Samples_c, y_c: Train_Acc_Target, InTrain:False}  
                    feed_acc_training_d = {x_image_d: Train_Acc_Samples_d, y_d: Train_Acc_Target, InTrain:False}  
                    
                    print("Model size:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
                    for ind_batch in range(Num_batch): 
                        Samples_in_batch_a = InputTraining_a[(ind_batch)*Num_samples:(ind_batch+1)*Num_samples,:,:,:]
                        Samples_in_batch_b = InputTraining_b[(ind_batch)*Num_samples:(ind_batch+1)*Num_samples,:,:,:]
                        Samples_in_batch_c = InputTraining_c[(ind_batch)*Num_samples:(ind_batch+1)*Num_samples,:,:,:]
                        Samples_in_batch_d = InputTraining_d[(ind_batch)*Num_samples:(ind_batch+1)*Num_samples,:,:,:]
                        Target_in_batch =  self.HotspotCenterValue0n1(TargetTraining[(ind_batch)*Num_samples:(ind_batch+1)*Num_samples,:,:,0])
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
     

def CreateGraph_4ConvNets_4HL_NFeaConv_wdnhxwdnh_BN(HyperPar, LocModel='Models/Default/FeatMaps'):
    with tf.device('/gpu:0'):
        
        # Global Environment
        Search_x, Search_y = int(HyperPar[2]), int(HyperPar[3]) #odd number Search Grid
        IPsizex, IPsizey =int(HyperPar[4]), int(HyperPar[5])    #odd number Inner Pattern        
        NFeaConv1 = int(HyperPar[10]) 
        NFeaConv2 = int(HyperPar[10]) 
        NFeaConv3 = int(HyperPar[10]) 
        NFeaConv4 = int(HyperPar[10]) 
    
        NumCategories = int(HyperPar[12]) # (2+1)
        InputLvl_a = 1
        InputLvl_b = 2
        InputLvl_c = 3
        InputLvl_d = 4
        
        NumNodesFC = int(HyperPar[8])
        
        wdnh = int(HyperPar[9])
    
        # ###############################################################
        # Defining the architecture
        # ###############################################################
        tf.reset_default_graph() 
        x_image_a = tf.placeholder(tf.float32, shape=[None, Search_x, Search_y, InputLvl_a], name="x_image_a") # Input
        y_a = tf.placeholder(tf.float32, shape=[None, IPsizex, IPsizey,NumCategories], name="y_a") # Output
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        InTrain = tf.placeholder(tf.bool, name="InTrain")
    
        W1a = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,InputLvl_a,NFeaConv1], stddev=0.1), dtype=tf.float32, name="W1a")
        b1a = tf.Variable(tf.constant(0.1, shape=[Search_x, Search_y,NFeaConv1]), dtype=tf.float32, name="b1a")
        conv1a = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(x_image_a, W1a, strides=[1,1,1,1], padding='SAME')+b1a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv1a")
        conv1a_pool = tf.nn.max_pool(conv1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv1a_pool")
        
        W2a = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv1,NFeaConv2], stddev=0.1), dtype=tf.float32, name="W2a")
        b2a = tf.Variable(tf.constant(0.1, shape=[conv1a_pool.shape[1],conv1a_pool.shape[2],NFeaConv2]), dtype=tf.float32, name="b2a")
        conv2a = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv1a_pool, W2a, strides=[1,1,1,1], padding='SAME')+b2a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv2a")
        conv2a_pool = tf.nn.max_pool(conv2a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv2a_pool")
        
        W3a = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv2,NFeaConv3], stddev=0.1), dtype=tf.float32, name="W3a")
        b3a = tf.Variable(tf.constant(0.1, shape=[conv2a_pool.shape[1],conv2a_pool.shape[2],NFeaConv3]), dtype=tf.float32, name="b3a")
        conv3a = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv2a_pool, W3a, strides=[1,1,1,1], padding='SAME')+b3a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv3a")
        conv3a_pool = tf.nn.max_pool(conv3a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv3a_pool")    
        
        W4a = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv3,NFeaConv4], stddev=0.1), dtype=tf.float32, name="W4a")
        b4a = tf.Variable(tf.constant(0.1, shape=[conv3a_pool.shape[1],conv3a_pool.shape[2],NFeaConv4]), dtype=tf.float32, name="b4a")
        conv4a = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv3a_pool, W4a, strides=[1,1,1,1], padding='SAME')+b4a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv4a")
        conv4a_pool = tf.nn.max_pool(conv4a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv4a_pool")         
    
        conv4a_flat = tf.reshape(conv4a_pool, [-1,int(conv4a_pool.shape[1])*int(conv4a_pool.shape[2])*int(conv4a_pool.shape[3])], name="conv4a_flat")
        WF1a = tf.Variable(tf.truncated_normal(shape=[int(conv4a_flat.get_shape()[1]), NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF1a")        
        full_1a = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(conv4a_flat, WF1a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_1a")
        
        WF2a = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF2a")
        full_2a = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(full_1a, WF2a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_2a")   
        
        WF3a = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, IPsizex*IPsizey*NumCategories], stddev=0.1), dtype=tf.float32, name="WF3a")
        full_3a = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(full_2a, WF3a), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_3a")          
    
        y_conv_a = tf.reshape(full_3a, [-1,int(IPsizex), int(IPsizey), NumCategories], name="y_conv_a")
        
        # ##########################################################################  
        
        x_image_b = tf.placeholder(tf.float32, shape=[None, Search_x, Search_y, InputLvl_b], name="x_image_b") # Input
        y_b = tf.placeholder(tf.float32, shape=[None, IPsizex, IPsizey,NumCategories], name="y_b") # Output
    
    
        W1b = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,InputLvl_b,NFeaConv1], stddev=0.1), dtype=tf.float32, name="W1b")
        b1b = tf.Variable(tf.constant(0.1, shape=[Search_x, Search_y,NFeaConv1]), dtype=tf.float32, name="b1b")
        conv1b = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(x_image_b, W1b, strides=[1,1,1,1], padding='SAME')+b1b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv1b")        
        conv1b_pool = tf.nn.max_pool(conv1b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv1b_pool")
        
        W2b = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv1,NFeaConv2], stddev=0.1), dtype=tf.float32, name="W2b")
        b2b = tf.Variable(tf.constant(0.1, shape=[conv1b_pool.shape[1],conv1b_pool.shape[2],NFeaConv2]), dtype=tf.float32, name="b2b")
        conv2b = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv1b_pool, W2b, strides=[1,1,1,1], padding='SAME')+b2b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv2b")        
        conv2b_pool = tf.nn.max_pool(conv2b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv2b_pool")  
        
        W3b = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv2,NFeaConv3], stddev=0.1), dtype=tf.float32, name="W3b")
        b3b = tf.Variable(tf.constant(0.1, shape=[conv2b_pool.shape[1],conv2b_pool.shape[2],NFeaConv3]), dtype=tf.float32, name="b3b")
        conv3b = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv2b_pool, W3b, strides=[1,1,1,1], padding='SAME')+b3b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv3b")
        conv3b_pool = tf.nn.max_pool(conv3b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv3b_pool")           
    
        W4b = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv3,NFeaConv4], stddev=0.1), dtype=tf.float32, name="W4b")
        b4b = tf.Variable(tf.constant(0.1, shape=[conv3b_pool.shape[1],conv3b_pool.shape[2],NFeaConv4]), dtype=tf.float32, name="b4b")
        conv4b = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv3b_pool, W4b, strides=[1,1,1,1], padding='SAME')+b4b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv4b")
        conv4b_pool = tf.nn.max_pool(conv4b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv4b_pool")         
    
        conv4b_flat = tf.reshape(conv4b_pool, [-1,int(conv4b_pool.shape[1])*int(conv4b_pool.shape[2])*int(conv4b_pool.shape[3])], name="conv4b_flat")
        WF1b = tf.Variable(tf.truncated_normal(shape=[int(conv4b_flat.get_shape()[1]), NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF1b")
        full_1b = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(conv4b_flat, WF1b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_1b")
        
        WF2b = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF2b")
        full_2b = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(full_1b, WF2b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_2b")   
        
        WF3b = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, IPsizex*IPsizey*NumCategories], stddev=0.1), dtype=tf.float32, name="WF3b")
        full_3b = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(full_2b, WF3b), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_3b")          
    
        y_conv_b = tf.reshape(full_3b, [-1,int(IPsizex), int(IPsizey), NumCategories], name="y_conv_b")   
        
        # ##########################################################################  
    
        x_image_c = tf.placeholder(tf.float32, shape=[None, Search_x, Search_y, InputLvl_c], name="x_image_c") # Input
        y_c = tf.placeholder(tf.float32, shape=[None, IPsizex, IPsizey, NumCategories], name="y_c") # Output
    
    
        W1c = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,InputLvl_c,NFeaConv1], stddev=0.1), dtype=tf.float32, name="W1c")
        b1c = tf.Variable(tf.constant(0.1, shape=[Search_x, Search_y,NFeaConv1]), dtype=tf.float32, name="b1c")
        conv1c = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(x_image_c, W1c, strides=[1,1,1,1], padding='SAME')+b1c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv1c")           
        conv1c_pool = tf.nn.max_pool(conv1c, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv1c_pool")
        
        W2c = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv1,NFeaConv2], stddev=0.1), dtype=tf.float32, name="W2c")
        b2c = tf.Variable(tf.constant(0.1, shape=[conv1c_pool.shape[1],conv1c_pool.shape[2],NFeaConv2]), dtype=tf.float32, name="b2c")
        conv2c = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv1c_pool, W2c, strides=[1,1,1,1], padding='SAME')+b2c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv2c")        
        conv2c_pool = tf.nn.max_pool(conv2c, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv2c_pool")        
    
        W3c = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv2,NFeaConv3], stddev=0.1), dtype=tf.float32, name="W3c")
        b3c = tf.Variable(tf.constant(0.1, shape=[conv2c_pool.shape[1],conv2c_pool.shape[2],NFeaConv3]), dtype=tf.float32, name="b3c")
        conv3c = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv2c_pool, W3c, strides=[1,1,1,1], padding='SAME')+b3c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv3c")
        conv3c_pool = tf.nn.max_pool(conv3c, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv3c_pool")           
    
        W4c = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv3,NFeaConv4], stddev=0.1), dtype=tf.float32, name="W4c")
        b4c = tf.Variable(tf.constant(0.1, shape=[conv3c_pool.shape[1],conv3c_pool.shape[2],NFeaConv4]), dtype=tf.float32, name="b4c")
        conv4c = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv3c_pool, W4c, strides=[1,1,1,1], padding='SAME')+b4c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv4c")
        conv4c_pool = tf.nn.max_pool(conv4c, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv4c_pool")         
    
        conv4c_flat = tf.reshape(conv4c_pool, [-1,int(conv4c_pool.shape[1])*int(conv4c_pool.shape[2])*int(conv4c_pool.shape[3])], name="conv4c_flat")
        WF1c = tf.Variable(tf.truncated_normal(shape=[int(conv4c_flat.get_shape()[1]), NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF1c")
        full_1c = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(conv4c_flat, WF1c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_1c")
        
        WF2c = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF2c")
        full_2c = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(full_1c, WF2c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_2c")   
        
        WF3c = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, IPsizex*IPsizey*NumCategories], stddev=0.1), dtype=tf.float32, name="WF3c")  
        full_3c = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(full_2c, WF3c), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_3c")           
        
    
        y_conv_c = tf.reshape(full_3c, [-1,int(IPsizex), int(IPsizey), NumCategories], name="y_conv_c")  
        
        # ##########################################################################     
        
        x_image_d = tf.placeholder(tf.float32, shape=[None, Search_x, Search_y, InputLvl_d], name="x_image_d") # Input
        y_d = tf.placeholder(tf.float32, shape=[None, IPsizex, IPsizey, NumCategories], name="y_d") # Output
    
    
        W1d = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,InputLvl_d,NFeaConv1], stddev=0.1), dtype=tf.float32, name="W1d")
        b1d = tf.Variable(tf.constant(0.1, shape=[Search_x, Search_y,NFeaConv1]), dtype=tf.float32, name="b1d")
        conv1d = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(x_image_d, W1d, strides=[1,1,1,1], padding='SAME')+b1d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv1d")           
        conv1d_pool = tf.nn.max_pool(conv1d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv1d_pool")
        
        W2d = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv1,NFeaConv2], stddev=0.1), dtype=tf.float32, name="W2d")
        b2d = tf.Variable(tf.constant(0.1, shape=[conv1d_pool.shape[1],conv1d_pool.shape[2],NFeaConv2]), dtype=tf.float32, name="b2d")
        conv2d = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv1d_pool, W2d, strides=[1,1,1,1], padding='SAME')+b2d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv2d")        
        conv2d_pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv2d_pool")         
    
        W3d = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv2,NFeaConv3], stddev=0.1), dtype=tf.float32, name="W3d")
        b3d = tf.Variable(tf.constant(0.1, shape=[conv2d_pool.shape[1],conv2d_pool.shape[2],NFeaConv3]), dtype=tf.float32, name="b3d")
        conv3d = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv2d_pool, W3d, strides=[1,1,1,1], padding='SAME')+b3d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv3d")
        conv3d_pool = tf.nn.max_pool(conv3d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv3d_pool")           
    
        W4d = tf.Variable(tf.truncated_normal(shape=[wdnh,wdnh,NFeaConv3,NFeaConv4], stddev=0.1), dtype=tf.float32, name="W4d")
        b4d = tf.Variable(tf.constant(0.1, shape=[conv3d_pool.shape[1],conv3d_pool.shape[2],NFeaConv4]), dtype=tf.float32, name="b4d")
        conv4d = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(conv3d_pool, W4d, strides=[1,1,1,1], padding='SAME')+b4d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="conv4d")
        conv4d_pool = tf.nn.max_pool(conv4d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="conv4d_pool")         
    
        conv4d_flat = tf.reshape(conv4d_pool, [-1,int(conv4d_pool.shape[1])*int(conv4d_pool.shape[2])*int(conv4d_pool.shape[3])], name="conv4d_flat")
        WF1d = tf.Variable(tf.truncated_normal(shape=[int(conv4d_flat.get_shape()[1]), NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF1d")
        full_1d = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(conv4d_flat, WF1d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_1d")   
           
        WF2d = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, NumNodesFC], stddev=0.1), dtype=tf.float32, name="WF2d")
        full_2d = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(full_1d, WF2d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_2d")   
        
        WF3d = tf.Variable(tf.truncated_normal(shape=[NumNodesFC, IPsizex*IPsizey*NumCategories], stddev=0.1), dtype=tf.float32, name="WF3d")
        full_3d = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(full_2d, WF3d), axis=-1, momentum=0.99, epsilon=0.001, scale=False, training=InTrain), name="full_3d")           
    
        y_conv_d = tf.reshape(full_3d, [-1,int(IPsizex), int(IPsizey), NumCategories], name="y_conv_d")           
        # ##########################################################################     
        argmax_y_conv_a = tf.argmax(tf.nn.softmax(y_conv_a,axis=-1),3, name="argmax_y_conv_a")[0]
        argmax_y_conv_b = tf.argmax(tf.nn.softmax(y_conv_b,axis=-1),3, name="argmax_y_conv_b")[0]
        argmax_y_conv_c = tf.argmax(tf.nn.softmax(y_conv_c,axis=-1),3, name="argmax_y_conv_c")[0]
        argmax_y_conv_d = tf.argmax(tf.nn.softmax(y_conv_d,axis=-1),3, name="argmax_y_conv_d")[0]
    
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
        
        correct_prediction_a = tf.equal(tf.argmax(y_conv_a,3), tf.argmax(y_a,3), name="correct_prediction_a")
        correct_prediction_b = tf.equal(tf.argmax(y_conv_b,3), tf.argmax(y_b,3), name="correct_prediction_b")
        correct_prediction_c = tf.equal(tf.argmax(y_conv_c,3), tf.argmax(y_c,3), name="correct_prediction_c")
        correct_prediction_d = tf.equal(tf.argmax(y_conv_d,3), tf.argmax(y_d,3), name="correct_prediction_d")
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

