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


# ######################### 

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import time
import External_Functions_2D as fns_nested
import gc


def SavePlot(HyperPar,SimRes,name,Level):
	SGsizex = HyperPar[0]
	SGsizey = HyperPar[1]
	delta_x = int((HyperPar[2]-1)/2)
	delta_y = int((HyperPar[3]-1)/2)
	A = str(name) + ".png"
	fig,ax = plt.subplots()
	colors1 = plt.cm.Paired(np.linspace(0, 1, 128))
	colors2 = plt.cm.binary(np.linspace(0, 1, 128))

	colors = np.vstack((colors1, colors2))
	mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)    
	pa = ax.imshow(SimRes[delta_x:SGsizex+delta_x,delta_y:delta_y+SGsizey,Level], cmap=mymap,  vmin=1, vmax=2)
	cba = plt.colorbar(pa,shrink=0.5, ticks=[1, 2]) 
	cba.set_clim(0, 2)
	fig.savefig(A)
	return plt.close(fig)   

def MeanReas(SimRes):
	for lvl in range(SimRes.shape[2]):
		SimRes[:,:,lvl,(SimRes.shape[3]-1)] = np.average(SimRes[:,:,lvl,(SimRes.shape[3]-1)], axis=2)
	return SimRes





for ind0 in range(3):
	
	SG_IP = [19,19,19]
	FullySizeNum = [5000,5000,5000]
	ConvNetDepth = [16,16,16]
	PercentageDataCond = [10, 20, 50]
	DC_PercS = [1, 2, 5]
	
	
	start_time_AllTrain = time.time() 
	HyperPar = []
	HyperPar.append(250) # SGsizex
	HyperPar.append(250) # SGsizey
	HyperPar.append(int(SG_IP[ind0])) # Search_x
	HyperPar.append(int(SG_IP[ind0])) # Search_y
	HyperPar.append(int(SG_IP[ind0])) # IPsizex
	HyperPar.append(int(SG_IP[ind0])) # IPsizey
	HyperPar.append(int(PercentageDataCond[ind0])) # Percentage of Data Conditioning *1000
	HyperPar.append(1) # MinDC
	HyperPar.append(int(FullySizeNum[ind0])) # Num Fully Connected
	HyperPar.append(3) # wdnh
	HyperPar.append(int(ConvNetDepth[ind0])) # convdepth
	HyperPar.append(1) # MultiGrid
	HyperPar.append(2) # NumCateg
	print("Size: ", int(HyperPar[2]))

	ExtDelta_x = int((HyperPar[2]-1)/2)
	ExtDelta_y = int((HyperPar[3]-1)/2)
	
	DataCond = "Circle_%spercDC.txt"%(int(DC_PercS[ind0]))
	TrainingImage = "Circle_TI_250x250.txt"
	LocModel = 'NewSimModels/Circle/Allperc/%sx%s_%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%s_%sconvdepth/FeatMaps'%(int(HyperPar[2]),int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]),int(HyperPar[8]),int(HyperPar[9]),int(HyperPar[9]),int(HyperPar[10]))	
	LocFile = 'NewSimModels/Circle/Allperc/%sx%s_%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%s_%sconvdepth/ResSim_GT_%sperc/'%(int(HyperPar[2]),int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]),int(HyperPar[8]),int(HyperPar[9]),int(HyperPar[9]),int(HyperPar[10]),int(DC_PercS[ind0]))	
	Nsim = 100
	start_time_1 = time.time()
	TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=DataCond, Lvl=5, Training=False)
	
	SimRes = np.zeros((TempSimGrid.Values.shape[0],TempSimGrid.Values.shape[1],TempSimGrid.Values.shape[2],Nsim+1))
	MetricsMatrix = np.zeros((8,Nsim+1))
	Metrics = False
	print("[Simulation]")
	for ind_c in range(Nsim):
		print ("Cicle: {}".format(ind_c+1)) 
		TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=DataCond, Lvl=5, Training=False)
		TempSimGrid.Simulate_4ConvNets_BN(LocModel=LocModel, Cicle=(ind_c+1), Plot=False)  	
		SimRes[:,:,:,ind_c] = TempSimGrid.Values
		TempSimGrid.SaveGrid(file="{}Res_{}.txt".format(LocFile, ind_c+1))
		if Metrics == True:
			print("[Metrics]")
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
		np.savetxt("{}MetricsMatrix.txt".format(LocFile), MetricsMatrix, delimiter='\t',  fmt='%5.8e')	
	
	for lvl in range(SimRes.shape[2]):
		np.savetxt("{}Total_{}Grid.txt".format(LocFile, lvl), SimRes[ExtDelta_x:ExtDelta_x+250,ExtDelta_y:ExtDelta_y+250,lvl,0:Nsim].reshape(-1,Nsim), delimiter='\t',  fmt='%1.0f')
		SimRes[:,:,lvl,Nsim] = np.average(SimRes[:,:,lvl,0:Nsim], axis=2)
		SavePlot(HyperPar=HyperPar,SimRes=SimRes[:,:,:,Nsim],name='{}Mean_lvl_{}'.format(LocFile, lvl),Level=lvl)

	print("--%s seconds of whole simulation process-" % (np.around((time.time() - start_time_1), decimals=2)))  
	print(" ")
	print("End")