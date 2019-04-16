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

import numpy as np
import tensorflow as tf
import time
import External_Functions_2D as fns_nested
import gc

for ind0 in range(1):
	SG = [19]
	IP = [19]
	FullySizeNum = [5000]
	WeFilter = [3]
	ConvNetDepth = [16]
	
	start_time_AllTrain = time.time() 
	HyperPar = []
	HyperPar.append(250) # SGsizex
	HyperPar.append(250) # SGsizey
	HyperPar.append(int(SG[ind0])) # Search_x
	HyperPar.append(int(SG[ind0])) # Search_y
	HyperPar.append(int(IP[ind0])) # IPsizex
	HyperPar.append(int(IP[ind0])) # IPsizey
	HyperPar.append(10) # Percentage of Data Conditioning *1000
	HyperPar.append(1) # MinDC
	HyperPar.append(int(FullySizeNum[ind0])) # Num Fully Connected
	HyperPar.append(int(WeFilter[ind0])) # wdnh
	HyperPar.append(int(ConvNetDepth[ind0])) # convdepth
	HyperPar.append(1) # MultiGrid
	HyperPar.append(2) # NumCateg
	print("Size: ", int(HyperPar[2]))
	
	Ncicles = 400
	Nepoch = 1
	Nbatch = 100
	TrainingImage = "Strebelle_TI_250x250.txt" # Training image in GSLib format WITHOUT the header
	LocModel = 'FinalModels/Strebelle/Allperc/%sx%s_%sx%s_4ConvNets_4HL_BN_3FC%s_ws%sx%s_%sconvdepth/FeatMaps'%(int(HyperPar[2]),int(HyperPar[3]),int(HyperPar[4]),int(HyperPar[5]),int(HyperPar[8]),int(HyperPar[9]),int(HyperPar[9]),int(HyperPar[10]))	
	
	print("[Graph]")
	fns_nested.CreateGraph_4ConvNets_4HL_NFeaConv_wdnhxwdnh_BN(HyperPar=HyperPar, LocModel=LocModel)
	TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=TrainingImage, Lvl=5, Training=True)
	print("Training Image proportion: ", TempSimGrid.BinaryProportion(TI=True))
	LHBPH_2x2_TI, LHBPH_3x3_TI = TempSimGrid.TI_LBP_2x2(), TempSimGrid.TI_LBP_3x3()
	
	# To save the TI
	TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=TrainingImage, Lvl=5,Training=False)
	TempSimGrid.SavePlot(name=LocModel+'_TI.png', Level=1)		
	
	for indTrain in range(Ncicles):
		HyperPar[6] = np.random.randint(46)+5 # Percentage of Data Conditioning *1000
		
		start_time_1 = time.time()
		print ("Cicle: {}".format(indTrain+1))
		TempSimGrid = fns_nested.Grid(HyperPar=HyperPar, DBname=TrainingImage, Lvl=5, Training=True)
		print("[Sim]")
		TempSimGrid.Simulate_4ConvNets_BN(LocModel=LocModel, Cicle=(indTrain+1)) 
		print("[Train]")
		TempSimGrid.Train_4ConvNets_BN(Epochs=Nepoch, Num_batch=Nbatch, LocModel=LocModel, LR=3e-4) 
		print("--%s seconds of whole training process-" % (np.around((time.time() - start_time_1), decimals=2)))  	
		gc.collect()
		MetricsBollean = False
		if MetricsBollean == True:
			print("[Metrics]")
			print("BinaryProportion ", TempSimGrid.BinaryProportion(lvl=1,TI=False), ", ", TempSimGrid.BinaryProportion(lvl=2,TI=False), ", ", TempSimGrid.BinaryProportion(lvl=3,TI=False), ", ", TempSimGrid.BinaryProportion(lvl=4,TI=False), ": 1st, 2nd, 3rd and 4th proportion")	
			print("Max_X ", TempSimGrid.Max_x(lvl=1), ", ", TempSimGrid.Max_x(lvl=2), ", ", TempSimGrid.Max_x(lvl=3), ", ", TempSimGrid.Max_x(lvl=4), ": 1st, 2nd, 3rd and 4th Max_X")
			print("Max_Y ", TempSimGrid.Max_y(lvl=1), ", ", TempSimGrid.Max_y(lvl=2), ", ", TempSimGrid.Max_y(lvl=3), ", ", TempSimGrid.Max_y(lvl=4), ": 1st, 2nd, 3rd and 4th Max_Y")
			print("MSE_LBP_2x2_WithExtrems ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=1), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=2), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=3), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=4), ": 1st, 2nd, 3rd and 4th MSE_LBP_2x2" )
			print("MSE_LBP_2x2_NoExtrems ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=1, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=2, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=3, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_2x2(LHBPH_2x2_TI, lvl=4, WithExtrems=False), ": 1st, 2nd, 3rd and 4th MSE_LBP_2x2 No Extrems" )
			print("MSE_LBP_3x3_WithExtrems ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=1), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=2), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=3), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=4), ": 1st, 2nd, 3rd and 4th MSE_LBP_3x3" )
			print("MSE_LBP_3x3_NoExtrems ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=1, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=2, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=3, WithExtrems=False), ", ", TempSimGrid.MSE_LBP_3x3(LHBPH_3x3_TI, lvl=4, WithExtrems=False), ": 1st, 2nd, 3rd and 4th MSE_LBP_3x3 No Extrems" )
		print(" ")
	
	
	print("--%s minutes of ALL training-" % ((time.time() - start_time_AllTrain)/60)) 