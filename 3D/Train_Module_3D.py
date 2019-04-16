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
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


## ######################### 

import numpy as np
import tensorflow as tf
import time
import External_Functions_3D as fns_nested
import gc

for ind0 in range(1):
	
	start_time_AllTrain = time.time() 
	HyperPar = []
	HyperPar.append(50) # SGsizex - Num 0
	HyperPar.append(50) # SGsizey - Num 1
	HyperPar.append(50) # SGsizez - Num 2
	
	HyperPar.append(int(7)) # Search_x - Num 3
	HyperPar.append(int(7)) # Search_y - Num 4
	HyperPar.append(int(7)) # Search_z - Num 5
	
	HyperPar.append(int(7)) # IPsizex - Num 6
	HyperPar.append(int(7)) # IPsizey - Num 7
	HyperPar.append(int(7)) # IPsizez - Num 8	
	
	HyperPar.append(50) # Percentage of Data Conditioning - Num 9 .. divided by 3 so 1% is 10 represents 1%
	HyperPar.append(1) # MinDC - Num 10
	HyperPar.append(1500) # Num Fully Connected - Num 11
	HyperPar.append(3) # wdnh - Num 12
	HyperPar.append(16) # convdepth - Num 13
	HyperPar.append(2) # num of categories - Num 14
	print("SG: ", int(HyperPar[3]),"x",int(HyperPar[4]),"x",int(HyperPar[5]), "IP: ", int(HyperPar[6]),"x",int(HyperPar[7]),"x",int(HyperPar[8]))
	
	Ncicles = 500
	Nepoch = 1
	#Nbatch = 250
	Nsamples = 512
	TrainingImage = "TI_Collaboration_1of4_50x50x50_newRepresentation.dat"
	LocModel = 'Models/3D_NewRepresentation/Allperc/%sx%sx%s_%sx%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%sx%s_%sconvdepth/FeatMaps'%(int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]), int(HyperPar[6]),int(HyperPar[7]),int(HyperPar[8]), int(HyperPar[11]), int(HyperPar[12]),int(HyperPar[12]),int(HyperPar[12]), int(HyperPar[13]))	
	#LocModel = 'Models/3D_NewRepresentation/New%sperc/%sx%sx%s_%sx%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%sx%s_%sconvdepth/FeatMaps'%(int(HyperPar[9]), int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]), int(HyperPar[6]),int(HyperPar[7]),int(HyperPar[8]), int(HyperPar[11]), int(HyperPar[12]),int(HyperPar[12]),int(HyperPar[12]), int(HyperPar[13]))	
	LocFile = 'Models/3D_NewRepresentation/Allperc/%sx%sx%s_%sx%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%sx%s_%sconvdepth'%(int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]), int(HyperPar[6]),int(HyperPar[7]),int(HyperPar[8]), int(HyperPar[11]), int(HyperPar[12]),int(HyperPar[12]),int(HyperPar[12]), int(HyperPar[13]))	
	#LocFile = 'Models/3D_NewRepresentation/New%sperc/%sx%sx%s_%sx%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%sx%s_%sconvdepth'%(int(HyperPar[9]), int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]), int(HyperPar[6]),int(HyperPar[7]),int(HyperPar[8]), int(HyperPar[11]), int(HyperPar[12]),int(HyperPar[12]),int(HyperPar[12]), int(HyperPar[13]))	
	print("[Graph]")
	#fns_nested.CreateGraph_4ConvNets_4HL_NFeaConv_wdnhxwdnh_BN_3D_NoBN(HyperPar=HyperPar, LocModel=LocModel)
	fns_nested.CreateGraph_4ConvNets_4HL_NFeaConv_wdnhxwdnh_BN_3D(HyperPar=HyperPar, LocModel=LocModel)
	
	
	# To save the TI
	TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=TrainingImage, Lvl=3,Training=False, Padding=True)
	TempSimGrid.SavePlot(name=LocModel+'_TI.png', Level=1)
	MaxLR, MinLR = 0.01, 0.001 
	StepLR = 10
	PointStart = 1
	for indTrain in range(Ncicles):
		#HyperPar[9] = np.random.randint(41)+10
		cuos = indTrain%(2*StepLR)
		if cuos < StepLR:
			LearningRate = np.around(((MaxLR - MinLR)/StepLR)*cuos + MinLR, decimals=7) 
		else:
			LearningRate = np.around(((MaxLR - MinLR)/StepLR)*(StepLR - cuos) + MaxLR, decimals=7) 
		start_time_1 = time.time()
		print ("Cicle: {}".format(indTrain+PointStart), "Learning Rate: ", LearningRate)
		TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=TrainingImage, Lvl=5, Training=True, Padding=True)
		print("[Sim]")
		TempSimGrid.Simulate_4ConvNets_BN_3D(LocModel=LocModel, Cicle=(indTrain+PointStart), Plot=True)
		print("[Saving Grid]")		
		TempSimGrid.SaveGrid(file="{}/TrainReas_{}.txt".format(LocFile, indTrain+PointStart))
		print("[Train]")
		TempSimGrid.Train_4ConvNets_BN_3D(Epochs=Nepoch, Num_samples=Nsamples, LocModel=LocModel, LR=LearningRate) 
		print("--%s seconds of whole training process-" % (np.around((time.time() - start_time_1), decimals=2)))  	
		gc.collect()
		print(" ")
	
	
	print("--%s minutes of ALL training-" % ((time.time() - start_time_AllTrain)/60)) 