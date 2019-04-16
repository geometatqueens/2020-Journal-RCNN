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

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
## ######################### 

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import time
import External_Functions_3D as fns_nested
import gc
from mayavi import mlab


def SavePlot(Pos_Matrix,SimRes,name,Level):
	mlab.options.offscreen = True
	A = str(name) + ".png"
	mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(800, 700))
	pts = mlab.points3d(Pos_Matrix[:,0],Pos_Matrix[:,1],Pos_Matrix[:,2],SimRes[:,:,:,Level].reshape((SimRes.shape[0]*SimRes.shape[1]*SimRes.shape[2])), scale_factor=1,  scale_mode='none', mode='cube', vmin=0, vmax=2)
	mesh = mlab.pipeline.delaunay2d(pts)
	#mlab.colorbar(orientation='horizontal', nb_labels=(self.NumCategs+1), label_fmt='%1.1f')
	mlab.savefig(A)
	return mlab.close()   


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

	ExtDelta_x = int((HyperPar[3]-1)/2)
	ExtDelta_y = int((HyperPar[4]-1)/2)
	ExtDelta_z = int((HyperPar[5]-1)/2)
	LargoOficial = HyperPar[0]*HyperPar[1]*HyperPar[2]
	
	DataCond = "DC_4of4_5perc_NewRepresentation.dat"
	GroundTruth = "TI_Collaboration_4of4_50x50x50_newRepresentation.dat"
	#LocModel = 'Models/3D_NewRepresentation/New%sperc/%sx%sx%s_%sx%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%sx%s_%sconvdepth/FeatMaps'%(int(HyperPar[9]), int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]), int(HyperPar[6]),int(HyperPar[7]),int(HyperPar[8]), int(HyperPar[11]), int(HyperPar[12]),int(HyperPar[12]),int(HyperPar[12]), int(HyperPar[13]))	
	#LocFile = 'Models/3D_NewRepresentation/New%sperc/%sx%sx%s_%sx%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%sx%s_%sconvdepth/ResSim_176epochs_S1'%(int(HyperPar[9]), int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]), int(HyperPar[6]),int(HyperPar[7]),int(HyperPar[8]), int(HyperPar[11]), int(HyperPar[12]),int(HyperPar[12]),int(HyperPar[12]), int(HyperPar[13]))	
	LocModel = 'Models/3D_NewRepresentation/Allperc/%sx%sx%s_%sx%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%sx%s_%sconvdepth/FeatMaps'%(int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]), int(HyperPar[6]),int(HyperPar[7]),int(HyperPar[8]), int(HyperPar[11]), int(HyperPar[12]),int(HyperPar[12]),int(HyperPar[12]), int(HyperPar[13]))	
	LocFile = 'Models/3D_NewRepresentation/Allperc/%sx%sx%s_%sx%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%sx%s_%sconvdepth/ResSim_5perDC_S3'%(int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]), int(HyperPar[6]),int(HyperPar[7]),int(HyperPar[8]), int(HyperPar[11]), int(HyperPar[12]),int(HyperPar[12]),int(HyperPar[12]), int(HyperPar[13]))	
	Nsim = 100
	WithPadd = True
	start_time_1 = time.time()
	TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=GroundTruth, Lvl=5, Training=False, Padding=WithPadd)
	TempSimGrid.SavePlot(name=LocFile+'/GroundTruth', Level=1)
	
	TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=DataCond, Lvl=5, Training=False, Padding=WithPadd)
	#TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=TrainingImage, Lvl=5, Training=True)
	#print("Training Image proportion: ", TempSimGrid.BinaryProportion(TI=True))
	#LHBPH_2x2_TI, LHBPH_3x3_TI = TempSimGrid.TI_LBP_2x2(), TempSimGrid.TI_LBP_3x3()
	#print("Max_X ", TempSimGrid.Max_x(lvl=0))
	#print("Max_Y ", TempSimGrid.Max_y(lvl=0))
	#print(TempSimGrid.Values.shape)
	
	Pos_Matrix = np.zeros((LargoOficial,3))
	count_0 = 0
	for indz in range(HyperPar[2]):
		for indy in range(HyperPar[1]):
			for indx in range(HyperPar[0]):
				Pos_Matrix[count_0,0],Pos_Matrix[count_0,1],Pos_Matrix[count_0,2] = indx, indy, indz
				count_0 += 1	
	
	SimRes = np.zeros((TempSimGrid.Values.shape[0],TempSimGrid.Values.shape[1],TempSimGrid.Values.shape[2],TempSimGrid.Values.shape[3],Nsim+1))
	MeanRes = np.zeros((TempSimGrid.Values.shape[0],TempSimGrid.Values.shape[1],TempSimGrid.Values.shape[2],TempSimGrid.Values.shape[3]))
	#print(SimRes[:,:,0,0:Nsim].shape)
	#MetricsMatrix = np.zeros((8,Nsim+1))
	Metrics = False
	print("[Simulation]")
	for ind_c in range(Nsim):
		print ("Cicle: {}".format(ind_c+1)) 
		TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=DataCond, Lvl=5, Training=False, Padding=WithPadd)
		TempSimGrid.Simulate_4ConvNets_BN_3D(LocModel=LocModel, Cicle=(ind_c+1), Plot=False)  	
		SimRes[:,:,:,:,ind_c] = TempSimGrid.Values
		TempSimGrid.SaveGrid(file="{}/Res_{}.txt".format(LocFile, ind_c+1))
		if ind_c > 0:
			for lvl in range(SimRes.shape[3]):
				if WithPadd == True:
					MeanRes[:,:,:,lvl] = np.average(SimRes[:,:,:,lvl,0:ind_c+1], axis=3)
					np.savetxt("{}/Mean_lvl_{}.txt".format(LocFile, lvl), MeanRes[:,:,:,lvl].reshape(-1), delimiter='\t',  fmt='%1.2f')
					SavePlot(Pos_Matrix=Pos_Matrix,SimRes=MeanRes[ExtDelta_x:ExtDelta_x+50,ExtDelta_y:ExtDelta_y+50,ExtDelta_z:ExtDelta_z+50,:],name='{}/Mean_lvl_{}'.format(LocFile, lvl),Level=lvl)				
				else:
					MeanRes[:,:,:,lvl] = np.average(SimRes[:,:,:,lvl,0:ind_c+1], axis=3)
					np.savetxt("{}/Mean_lvl_{}.txt".format(LocFile, lvl), MeanRes[:,:,:,lvl].reshape(-1), delimiter='\t',  fmt='%1.2f')					
					SavePlot(Pos_Matrix=Pos_Matrix,SimRes=MeanRes,name='{}/Mean_lvl_{}'.format(LocFile, lvl),Level=lvl)		
		if Metrics == True:
			print("[Metrics]")
			#print("PixelError ", TempSimGrid.PixelError(lvl=1), ", ", TempSimGrid.PixelError(lvl=2), ", ", TempSimGrid.PixelError(lvl=3), ", ", TempSimGrid.PixelError(lvl=4), ": 1st, 2nd, 3rd and 4th Pixel Error")		
			#print("BinaryProportion ", TempSimGrid.BinaryProportion(lvl=1,TI=False), ", ", TempSimGrid.BinaryProportion(lvl=2,TI=False), ", ", TempSimGrid.BinaryProportion(lvl=3,TI=False), ", ", TempSimGrid.BinaryProportion(lvl=4,TI=False), ": 1st, 2nd, 3rd and 4th proportion")	
			#print("Max_X ", TempSimGrid.Max_x(lvl=1), ", ", TempSimGrid.Max_x(lvl=2), ", ", TempSimGrid.Max_x(lvl=3), ", ", TempSimGrid.Max_x(lvl=4), ": 1st, 2nd, 3rd and 4th Max_X")
			#print("Max_Y ", TempSimGrid.Max_y(lvl=1), ", ", TempSimGrid.Max_y(lvl=2), ", ", TempSimGrid.Max_y(lvl=3), ", ", TempSimGrid.Max_y(lvl=4), ": 1st, 2nd, 3rd and 4th Max_Y")
			#print("MSE_LBP_2x2_WithExtrems ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=1), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=2), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=3), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=4), ": 1st, 2nd, 3rd and 4th MSE_LBP_2x2" )
			#print("MSE_LBP_2x2_NoExtrems ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=1, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=2, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=3, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=4, WithExtrems=False), ": 1st, 2nd, 3rd and 4th MSE_LBP_2x2 No Extrems" )
			#print("MSE_LBP_3x3_WithExtrems ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=1), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=2), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=3), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=4), ": 1st, 2nd, 3rd and 4th MSE_LBP_3x3" )
			#print("MSE_LBP_3x3_NoExtrems ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=1, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=2, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=3, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=4, WithExtrems=False), ": 1st, 2nd, 3rd and 4th MSE_LBP_3x3 No Extrems" )
			#print(" ")
			MetricsMatrix[0,ind_c] = TempSimGrid.PixelError(lvl=4)
			MetricsMatrix[1,ind_c] = TempSimGrid.BinaryProportion(lvl=4,TI=False)
			MetricsMatrix[2,ind_c] = TempSimGrid.Max_x(lvl=4)
			MetricsMatrix[3,ind_c] = TempSimGrid.Max_y(lvl=4)
			MetricsMatrix[4,ind_c] = TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=4)
			MetricsMatrix[5,ind_c] = TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=4, WithExtrems=False)
			MetricsMatrix[6,ind_c] = TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=4)
			MetricsMatrix[7,ind_c] = TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=4, WithExtrems=False)
			print("..Done")
		gc.collect()
	if Metrics == True:
		for ind1 in range(8):
			MetricsMatrix[ind1][-1] = np.mean(MetricsMatrix[ind1][0:Nsim])
		np.savetxt("{}/MetricsMatrix.txt".format(LocFile), MetricsMatrix, delimiter='\t',  fmt='%5.8e')	
	
	
	
	
	print("--%s seconds of whole simulation process-" % (np.around((time.time() - start_time_1), decimals=2)))  
	print(" ")
	print("End")