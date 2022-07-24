#!/usr/bin/env python
import numpy as np
from numpy import genfromtxt
import sys

#sys.argv: 1: cut, 2: numFWsample, 3: nbranch, 4: pythonseeed

if __name__ == "__main__":
	# print(sys.argv[1:])
	''' Basic Data Inputs '''
	# read electricity price data
	lmp_data = genfromtxt('./electricity_price_data_21.csv', delimiter=',')
	SUBPERIOD = 24
	if lmp_data.shape[0] % SUBPERIOD == 0:
		HORIZON = int(lmp_data.shape[0] / SUBPERIOD)
		print("HORIZON = ", HORIZON)
		NominalPrice = np.zeros( ( HORIZON, SUBPERIOD ) )
		for t in range(HORIZON):
			for d in range(SUBPERIOD):
				NominalPrice[t][d] = lmp_data[ d + SUBPERIOD * t ]

	BREAKSTAGE = int( HORIZON / 3.0 )
	STATEVARIABLE_LIST = {0: 'Start-up', 1: 'Shut-down', 2: 'On-off', 3: 'Gen-amount'}
	CUPPER = 440
	CLOWER = 59
	VRAMP = 51
	VUPPER = CLOWER + VRAMP / 3
	MINUP = 4
	MINDOWN = 4
	SUCOST = 300
	SDCOST = 0
	EPLISON = 0.01			# used for binary expansion
	quadA = 0.02
	quadB = 22
	quadC = 100
	
	# seed number for generate random numbers
	seedNo = int(sys.argv[4])
	print("seedNo = ", seedNo)
	np.random.seed( seedNo )
	
	Q0 = SUBPERIOD * CUPPER / 3.0 # for generating supply

	GenInfo = [CUPPER, CLOWER, VRAMP, VUPPER, MINUP, MINDOWN, SUCOST, SDCOST, quadA, quadB, quadC, SUBPERIOD, Q0, seedNo]

	myFile = open("./data/GenInfo.dat", "w")
	myFile.write(str(GenInfo))
	myFile.close()
	
	numFWsample = int(sys.argv[2])
	print("numFWsample = ", numFWsample)
	
	numPieceSegment = 4		# for piecewise linearization of quadratic cost function
	nType = len(STATEVARIABLE_LIST) 	# the number of state variable types, 4
	
	numScenEach = int(sys.argv[3])
	print("numScenEach = ", numScenEach)	
	
	numScen = np.ones(HORIZON - 1, np.int64) * numScenEach 	# [2, 2], carefully check it needs np.int64 or np.int32
	numScen = np.insert(numScen, 0, 1) 				# [1, 2, 2]
	print("numScen = ", numScen)
	
	numBinExp = int( np.floor(np.log2(CUPPER / EPLISON)) + 1 )
	print("numBinExp = ", numBinExp)

	intX = (nType - 1) * SUBPERIOD		# the number of integer variables, double check later

	# parameters to indicate use cutting planes or not
	addCut = int(sys.argv[1])
	print("addCut = ", addCut)
	
	''' Parameters used by SDDiP '''
	# number of stages
	myFile = open("./data/numStage.dat", "w")
	myFile.write(str(HORIZON))
	myFile.close()
	
	# the location of breakstage
	myFile = open("./data/breakstage.dat", "w")
	myFile.write(str(BREAKSTAGE))
	myFile.close()
	
	# the number of forward samples
	myFile = open("./data/numFWsample.dat", "w")
	myFile.write(str(numFWsample))
	myFile.close()

	# use cuts or not, 1: cut, 0: no cuts
	myFile = open("./data/boolCut.dat", "w")
	myFile.write(str(addCut))
	myFile.close()
	
    # number of scenarios at each stage
	myFile = open("./data/numScen.dat", "w")
	myFile.write("[")
	for i in range(numScen.shape[0]):
		myFile.write(str(numScen[i]))
		if i != numScen.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	# number of integer state variables
	myFile = open("./data/intX.dat", "w")
	myFile.write(str(intX))
	myFile.close()
	
	# construct xBound, x: u (SU), v (SD), y (On-off), x (Gen-amount)
	xBound1 = np.zeros( nType * SUBPERIOD, dtype=int )		# lower bound
	xBound2 = np.ones( (nType - 1) * SUBPERIOD, dtype=int )	# upper bound
	xBound3 = np.ones( SUBPERIOD ) * CUPPER * SUBPERIOD		# upper bound
	xBound = np.vstack((  xBound1 ,  np.hstack((xBound2, xBound3 ))  ))
	print("xBound = ", xBound)
	print("xBound.shape:", xBound.shape)
	
	myFile = open("./data/xBound.dat", "w")
	myFile.write("[")
	for row in range(xBound.shape[0]):
		myFile.write("[")
		for column in range(xBound.shape[1]):
			myFile.write(str(xBound[row, column]))
			if column != xBound.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != xBound.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	# construct thetaBound
	nonAchieveProfit = 2 * NominalPrice.max() * CUPPER * HORIZON * SUBPERIOD
	thetaBound1 = np.ones( HORIZON, dtype=int ) * nonAchieveProfit * (-1)	# lower bound
	thetaBound2 = np.ones( HORIZON, dtype=int ) * 1e+20						# upper bound
	thetaBound = np.vstack((  thetaBound1 ,  thetaBound2  ))
	print("thetaBound = ", thetaBound)
	print("thetaBound.shape:", thetaBound.shape)
	
	myFile = open("./data/thetaBound.dat", "w")
	myFile.write("[")
	for row in range(thetaBound.shape[0]):
		myFile.write("[")
		for column in range(thetaBound.shape[1]):
			myFile.write(str(thetaBound[row, column]))
			if column != thetaBound.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != thetaBound.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	# construct binary expansion/approximation matrix: xBinCoeff^T ~= xCoeff^T * matBin
	# [u,v,y,x]^T = matBin * [u,v,y,\lambda_i]^T
	matBin1 = np.eye( (nType - 1) * SUBPERIOD , dtype=int )			# for u, v, y; upper left matrix
	matBin2 = np.zeros( ( SUBPERIOD, numBinExp * ( SUBPERIOD ) ) )	# for x; lower right matrix
	for i in range(SUBPERIOD):
		for j in range(numBinExp):
			# print "i = ", i, ", j = ", j
			matBin2[i][numBinExp * i + j] = np.power(2,j) * EPLISON
	
	matBin3 = np.zeros(( (nType - 1) * SUBPERIOD, numBinExp * ( SUBPERIOD )  ))		# upper right matrix
	matBin4 = np.zeros(( SUBPERIOD, (nType - 1) * SUBPERIOD  ))						# lower left matrix
	matBin = np.vstack((  np.hstack((matBin1, matBin3 )) ,  np.hstack((matBin4, matBin2 ))  ))
	
	#print matBin
	print("matBin.shape = ", matBin.shape)
	print("matBin2[0][0 : numBinExp] = ", matBin2[0][0 : numBinExp])
	
	matBinTranspose = matBin.transpose()
	
	print("matBinTranspose.shape = ", matBinTranspose.shape)

	myFile = open("./data/T.dat", "w")
	myFile.write("[")
	for row in range(matBin.shape[0]):
		myFile.write("[")
		for column in range(matBin.shape[1]):
			myFile.write(str(matBin[row, column]))
			if column != matBin.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != matBin.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	myFile = open("./data/TT.dat", "w")
	myFile.write("[")
	for row in range(matBinTranspose.shape[0]):
		myFile.write("[")
		for column in range(matBinTranspose.shape[1]):
			myFile.write(str(matBinTranspose[row, column]))
			if column != matBinTranspose.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != matBinTranspose.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	''' Construct data for objective function c x + b y '''
    # construct c for x (i.e., state variables u, v, y, q, x) in the original space
	StartUpCoeff = np.ones(SUBPERIOD) * SUCOST
	ShutDownCoeff = np.ones(SUBPERIOD) * SDCOST
	OnoffCoeff = np.zeros(SUBPERIOD)
	xCoef1 = np.hstack(( StartUpCoeff, ShutDownCoeff, OnoffCoeff ))
	print("xCoef1.shape", xCoef1.shape)
	
	xCoef = np.zeros(( HORIZON, nType * SUBPERIOD ))
	for t in np.arange(HORIZON):
		xCoef[t] = np.hstack(( xCoef1, (-1) * NominalPrice[t] ))	# NominalPrice is a matrix
	print("xCoef.shape", xCoef.shape)
	
	myFile = open("./data/x.dat", "w")
	myFile.write("[")
	for row in range(xCoef.shape[0]):
		myFile.write("[")
		for column in range(xCoef.shape[1]):
			myFile.write(str(xCoef[row, column]))
			if column != xCoef.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != xCoef.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	# construct c for x in the binary expansion space
	xCoefBin = np.zeros((HORIZON, (nType - 1 + numBinExp) * SUBPERIOD ))
	for t in np.arange(HORIZON):
		xCoefBin[t] = np.hstack(( xCoef1, ( (-1) * NominalPrice[t][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten() ))
	print("xCoefBin.shape", xCoefBin.shape)
	
	myFile = open("./data/xBin.dat", "w")
	myFile.write("[")
	for row in range(xCoefBin.shape[0]):
		myFile.write("[")
		for column in range(xCoefBin.shape[1]):
			myFile.write(str(xCoefBin[row, column]))
			if column != xCoefBin.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != xCoefBin.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
    # construct b for the continuous local variable (i.e., piecewise linear function variable w)
	yCoef = np.ones((HORIZON, SUBPERIOD))
	print("yCoef.shape", yCoef.shape)

	myFile = open("./data/y2.dat", "w")
	myFile.write("[")
	for row in range(yCoef.shape[0]):
		myFile.write("[")
		for column in range(yCoef.shape[1]):
			myFile.write(str(yCoef[row, column]))
			if column != yCoef.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != yCoef.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	
	''' Construct data for constraints A x + W y + B z >= rhs '''
	''' x: state variable, y: local variable, z_t = x_t-1 '''
	# minimum up time constraints
	MUSU = np.eye( SUBPERIOD ) * (-1)	# for the u part in the current time period
	for i in np.arange(1, MINUP):
		MUSU += np.eye( SUBPERIOD, k = (-1) * i ) * (-1)
	
	MinUpConsX = np.hstack(( MUSU, np.zeros( (SUBPERIOD, SUBPERIOD) ), np.eye( SUBPERIOD ), np.zeros( (SUBPERIOD, SUBPERIOD) ) ))
	MinUpConsXBin = np.hstack(( MUSU, np.zeros( (SUBPERIOD, SUBPERIOD) ), np.eye( SUBPERIOD ), np.zeros( (SUBPERIOD, (SUBPERIOD) * numBinExp) ) ))
	
	MUSUZ = np.zeros( (SUBPERIOD, SUBPERIOD) ) # z part
	for i in np.arange(1, MINUP):
		MUSUZ += np.eye( SUBPERIOD, k = SUBPERIOD - i ) * (-1)
	
	MinUpConsZ = np.hstack(( MUSUZ, np.zeros( (SUBPERIOD, SUBPERIOD * (nType - 1) ) )  ))
	MinUpConsZBin = np.hstack(( MUSUZ, np.zeros( ( SUBPERIOD, SUBPERIOD * (nType - 2 + numBinExp) ) )  ))

	print("MinUpConsX: ", MinUpConsX.shape, MinUpConsX)
	print("MinUpConsXBin: ", MinUpConsXBin.shape)
	print("MinUpConsZ: ", MinUpConsZ.shape, MinUpConsZ)
	print("MinUpConsZBin: ", MinUpConsZBin.shape)
	
	
	# minimum down time constraints
	MDSD = np.eye( SUBPERIOD ) * (-1)
	for i in np.arange(1, MINDOWN):
		MDSD += np.eye( SUBPERIOD, k = (-1) * i ) * (-1)
	
	MinDnConsX = np.hstack(( np.zeros( (SUBPERIOD, SUBPERIOD) ), MDSD, np.eye( SUBPERIOD ) * (-1), np.zeros( ( SUBPERIOD, SUBPERIOD ) ) ))
	MinDnConsXBin = np.hstack(( np.zeros( (SUBPERIOD, SUBPERIOD) ), MDSD, np.eye( SUBPERIOD ) * (-1), np.zeros(( SUBPERIOD, (SUBPERIOD) * numBinExp) ) ))
	
	MDSDZ = np.zeros( (SUBPERIOD, SUBPERIOD) ) # z part
	for i in np.arange(1, MINDOWN):
		MDSDZ += np.eye( SUBPERIOD, k = SUBPERIOD - i ) * (-1)
	
	MinDnConsZ = np.hstack(( np.zeros( (SUBPERIOD, SUBPERIOD) ), MDSDZ, np.zeros( ( SUBPERIOD, SUBPERIOD * (nType - 2) ) )  ))
	MinDnConsZBin = np.hstack(( np.zeros( (SUBPERIOD, SUBPERIOD) ), MDSDZ, np.zeros(( SUBPERIOD, SUBPERIOD * (nType - 3 + numBinExp) ))  ))

	print("MinDnConsX: ", MinDnConsX.shape, MinDnConsX)
	print("MinDnConsXBin: ", MinDnConsXBin.shape)
	print("MinDnConsZ: ", MinDnConsZ.shape, MinDnConsZ)
	print("MinDnConsZBin: ", MinDnConsZBin.shape)
	
	# start-up constraints, i.e., u_t >= y_t - y_{t-1}; & shut-down constraints: u_t - v_t + y_{t-1} - y_t = 0
	SUConsSU = np.eye( SUBPERIOD )
	SUConsOnOff = np.eye( SUBPERIOD ) * (-1) + np.eye( SUBPERIOD, k = -1 )
	SUConsOnOffZ = np.eye( SUBPERIOD, k = SUBPERIOD - 1 )
	
	SUConsX = np.hstack(( SUConsSU, np.zeros( (SUBPERIOD, SUBPERIOD) ), SUConsOnOff, np.zeros( (SUBPERIOD, SUBPERIOD) ) ))
	SUConsXBin = np.hstack(( SUConsSU, np.zeros( (SUBPERIOD, SUBPERIOD) ), SUConsOnOff, np.zeros(( SUBPERIOD, ( SUBPERIOD ) * numBinExp)) ))
	SUConsZ = np.hstack(( np.zeros( (SUBPERIOD, 2 * SUBPERIOD) ), SUConsOnOffZ, np.zeros(( SUBPERIOD, SUBPERIOD ))  ))
	SUConsZBin = np.hstack(( np.zeros(( SUBPERIOD, 2 * SUBPERIOD )), SUConsOnOffZ, np.zeros(( SUBPERIOD, ( SUBPERIOD ) * numBinExp ))  ))
	
	print("SUConsX: ", SUConsX.shape, SUConsX)
	print("SUConsXBin: ", SUConsXBin.shape)
	print("SUConsZ: ", SUConsZ.shape, SUConsZ)
	print("SUConsZBin: ", SUConsZBin.shape)
	
	SDConsSU = np.vstack((SUConsSU, -SUConsSU ))
	SDConsSD = np.vstack((-SUConsSU, SUConsSU ))
	SDConsOnOff = np.vstack(( SUConsOnOff, -SUConsOnOff ))
	SDConsOnOffZ = np.vstack((SUConsOnOffZ, -SUConsOnOffZ ))
	
	SDConsX = np.hstack(( SDConsSU, SDConsSD, SDConsOnOff, np.zeros( (2 * SUBPERIOD, SUBPERIOD) ) ))
	SDConsXBin = np.hstack(( SDConsSU, SDConsSD, SDConsOnOff, np.zeros( (2 * SUBPERIOD, ( SUBPERIOD ) * numBinExp) ) ))
	SDConsZ = np.hstack(( np.zeros( (2 * SUBPERIOD, 2 * SUBPERIOD) ), SDConsOnOffZ, np.zeros( (2 * SUBPERIOD, SUBPERIOD) )  ))
	SDConsZBin = np.hstack(( np.zeros( (2 * SUBPERIOD, 2 * SUBPERIOD) ), SDConsOnOffZ, np.zeros( (2 * SUBPERIOD, (SUBPERIOD) * numBinExp) )  ))
	
	print("SDConsX: ", SDConsX.shape, SDConsX)
	print("SDConsXBin: ", SDConsXBin.shape)
	print("SDConsZ: ", SDConsZ.shape, SDConsZ)
	print("SDConsZBin: ", SDConsZBin.shape)
	
	# capacity upper bound and lower bound
	UBConsX = np.hstack((  np.zeros( ( SUBPERIOD, 2 * SUBPERIOD) ),  CUPPER * np.eye( SUBPERIOD ), (-1) *np.eye( SUBPERIOD )   ))
	UBConsXBin = np.hstack((  np.zeros( ( SUBPERIOD, 2 * SUBPERIOD) ),  CUPPER * np.eye( SUBPERIOD ), (-1) * matBin2   ))
	LBConsX = np.hstack((  np.zeros( ( SUBPERIOD, 2 * SUBPERIOD) ),  CLOWER * np.eye( SUBPERIOD ) * (-1), np.eye( SUBPERIOD )   ))
	LBConsXBin = np.hstack((  np.zeros( ( SUBPERIOD, 2 * SUBPERIOD) ),  CLOWER * np.eye( SUBPERIOD ) * (-1), matBin2   ))

	print("UBConsX: ", UBConsX.shape, UBConsX)
	print("UBConsXBin: ", UBConsXBin.shape, UBConsXBin)
	print("LBConsX: ", LBConsX.shape, LBConsX)
	print("LBConsXBin: ", LBConsXBin.shape, LBConsXBin)
	
	# ramping up constraints
	RampUpConsGen = (-1) * np.eye( SUBPERIOD ) + np.eye( SUBPERIOD, k = -1 )
	RampUpConsX = np.hstack(( np.zeros( ( SUBPERIOD, 2 * SUBPERIOD) ), (VRAMP - VUPPER) * np.eye( SUBPERIOD, k = -1 ), RampUpConsGen  ))
	
	RampUpConsZGen = np.eye( SUBPERIOD, k = SUBPERIOD-1 )
	RampUpConsZ = np.hstack(( np.zeros( ( SUBPERIOD, 2 * SUBPERIOD) ), (VRAMP - VUPPER) * np.eye( SUBPERIOD, k = SUBPERIOD-1 ), RampUpConsZGen  ))
	
	# binary space
	RampUpConsGenBin = ( RampUpConsGen[0][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()
	for i in np.arange(1, RampUpConsGen.shape[0] ):
		RampUpConsGenBin = np.vstack(( RampUpConsGenBin, ( RampUpConsGen[i][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()  ))
	
	RampUpConsXBin = np.hstack(( np.zeros( ( SUBPERIOD, 2 * SUBPERIOD) ), (VRAMP - VUPPER) * np.eye( SUBPERIOD, k = -1 ), RampUpConsGenBin  ))

	RampUpConsZGenBin = ( RampUpConsZGen[0][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()
	for i in np.arange(1, RampUpConsZGen.shape[0] ):
		RampUpConsZGenBin = np.vstack(( RampUpConsZGenBin, ( RampUpConsZGen[i][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()  ))	
	
	RampUpConsZBin = np.hstack(( np.zeros( ( SUBPERIOD, 2 * SUBPERIOD) ), (VRAMP - VUPPER) * np.eye( SUBPERIOD, k = SUBPERIOD-1 ), RampUpConsZGenBin  ))
	
	# ramping down constraints
	RampDnConsGen = np.eye( SUBPERIOD ) + (-1) * np.eye( SUBPERIOD, k = -1 )
	RampDnConsX = np.hstack(( np.zeros( ( SUBPERIOD, 2 * SUBPERIOD) ), (VRAMP - VUPPER) * np.eye(SUBPERIOD), RampDnConsGen ))
	RampDnConsZGen = (-1) * np.eye( SUBPERIOD, k = SUBPERIOD-1 )
	RampDnConsZ = np.hstack(( np.zeros( ( SUBPERIOD, 3 * SUBPERIOD) ), RampDnConsZGen  ))
	
	# binary space
	RampDnConsGenBin = ( RampDnConsGen[0][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()
	for i in np.arange(1, RampDnConsGen.shape[0] ):
		RampDnConsGenBin = np.vstack(( RampDnConsGenBin, ( RampDnConsGen[i][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()  ))
	
	RampDnConsXBin = np.hstack(( np.zeros( ( SUBPERIOD, 2 * SUBPERIOD) ), (VRAMP - VUPPER) * np.eye(SUBPERIOD), RampDnConsGenBin  ))

	RampDnConsZGenBin = ( RampDnConsZGen[0][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()
	for i in np.arange(1, RampDnConsZGen.shape[0] ):
		RampDnConsZGenBin = np.vstack(( RampDnConsZGenBin, ( RampDnConsZGen[i][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()  ))	
	
	RampDnConsZBin = np.hstack(( np.zeros( ( SUBPERIOD, 3 * SUBPERIOD) ), RampDnConsZGenBin  ))
	
	print("RampUpConsX: ", RampUpConsX.shape, RampUpConsX)
	print("RampUpConsXBin: ", RampUpConsXBin.shape)
	print("RampUpConsZ: ", RampUpConsZ.shape, RampUpConsZ)
	print("RampUpConsZBin: ", RampUpConsZBin.shape)
	print("RampDnConsX: ", RampDnConsX.shape, RampDnConsX)
	print("RampDnConsXBin: ", RampDnConsXBin.shape)
	print("RampDnConsZ: ", RampDnConsZ.shape, RampDnConsZ)
	print("RampDnConsZBin: ", RampDnConsZBin.shape)
	
	# fuel limit constraints
	FuelConsGen = (-1) * np.ones(SUBPERIOD)
	FuelConsX = np.hstack(( np.zeros( 3 * SUBPERIOD ), FuelConsGen ))
	FuelConsZ = np.hstack(( np.zeros( 3 * SUBPERIOD ), np.zeros( SUBPERIOD ) ))
	
	FuelConsGenBin = ( FuelConsGen[:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()
	
	FuelConsXBin = np.hstack(( np.zeros( 3 * SUBPERIOD ), FuelConsGenBin ))
	FuelConsZBin = np.hstack(( np.zeros( 3 * SUBPERIOD), np.zeros( SUBPERIOD * numBinExp) ))
	
	print("FuelConsX: ", FuelConsX.shape, FuelConsX)
	print("FuelConsXBin: ", FuelConsXBin.shape, FuelConsXBin)
	
	# piecewise linearization constraints, i.e., w_t >= \mu_k x_t + \eta_k y_t, \forall k
	muArray = np.zeros( numPieceSegment )
	etaArray = np.zeros( numPieceSegment )
	for i in range(numPieceSegment ):
		print( i, " , ", (CLOWER + i * ((CUPPER - CLOWER) / float(numPieceSegment - 1) ) ) )
		muArray[i] = 2 * quadA * (CLOWER + i * ((CUPPER - CLOWER) / float(numPieceSegment - 1) ) ) + quadB
		etaArray[i] = quadC - quadA * (CLOWER + i * ((CUPPER - CLOWER) / float(numPieceSegment - 1) ) ) ** 2

	PieceConsOnOff = (-1) * etaArray[0] * np.eye( SUBPERIOD )
	for i in np.arange( 1, numPieceSegment ):
		PieceConsOnOff = np.vstack((PieceConsOnOff, (-1) * etaArray[i] * np.eye( SUBPERIOD ) ))

	PieceConsGen = (-1) * muArray[0] * np.eye( SUBPERIOD )
	for i in np.arange( 1, numPieceSegment ):
		PieceConsGen = np.vstack(( PieceConsGen, (-1) * muArray[i] * np.eye( SUBPERIOD ) ))
	
	PieceConsGenBin = ( PieceConsGen[0][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()
	for i in np.arange(1, PieceConsGen.shape[0] ):
		PieceConsGenBin = np.vstack(( PieceConsGenBin, ( PieceConsGen[i][:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()  ))
	
	PieceConsX = np.hstack(( np.zeros(( SUBPERIOD * numPieceSegment, 2 * SUBPERIOD )), PieceConsOnOff, PieceConsGen  ))
	PieceConsXBin = np.hstack(( np.zeros((SUBPERIOD * numPieceSegment, 2 * SUBPERIOD )), PieceConsOnOff, PieceConsGenBin  ))

	PieceConsY = np.eye( SUBPERIOD )
	for i in np.arange(1, numPieceSegment ):
		PieceConsY = np.vstack((PieceConsY, np.eye( SUBPERIOD ) ))		
		
	print("PieceConsX: ", PieceConsX.shape, PieceConsX)
	print("PieceConsXBin: ", PieceConsXBin.shape)
	print("PieceConsY: ", PieceConsY.shape, PieceConsY)

	# begin to construct the whole matrix A, W, and B
	A = np.vstack((MinUpConsX, MinDnConsX, SUConsX, SDConsX, UBConsX, LBConsX,
					RampUpConsX, RampDnConsX, FuelConsX, PieceConsX))
	ABin = np.vstack((MinUpConsXBin, MinDnConsXBin, SUConsXBin, SDConsXBin, UBConsXBin, LBConsXBin,
					RampUpConsXBin, RampDnConsXBin, FuelConsXBin, PieceConsXBin))

	print("A: ", A.shape, A)
	print("ABin: ", ABin.shape)

	W = np.vstack(( np.zeros(( 9 * SUBPERIOD + 1, SUBPERIOD  )), PieceConsY ))
	
	print("W: ", W.shape, W)
	
	B = np.vstack((MinUpConsZ, MinDnConsZ, SUConsZ, SDConsZ, np.zeros( ( 2 * SUBPERIOD, nType * SUBPERIOD ) ),
					RampUpConsZ, RampDnConsZ, FuelConsZ, np.zeros( ( numPieceSegment * SUBPERIOD, nType * SUBPERIOD ) ) ))
	BBin = np.vstack((MinUpConsZBin, MinDnConsZBin, SUConsZBin, SDConsZBin, 
					np.zeros( ( 2 * SUBPERIOD, (nType - 1 + numBinExp) * SUBPERIOD ) ),
					RampUpConsZBin, RampDnConsZBin, FuelConsZBin, np.zeros( ( numPieceSegment * SUBPERIOD, (nType - 1 + numBinExp) * SUBPERIOD ) ) ))
					
	print("B: ", B.shape, B)
	print("BBin: ", BBin.shape)
	
	myFile = open("./data/A.dat", "w")
	myFile.write("[")
	for row in range(A.shape[0]):
		myFile.write("[")
		for column in range(A.shape[1]):
			myFile.write(str(A[row, column]))
			if column != A.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != A.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	myFile = open("./data/ABin.dat", "w")
	myFile.write("[")
	for row in range(ABin.shape[0]):
		myFile.write("[")
		for column in range(ABin.shape[1]):
			myFile.write(str(ABin[row, column]))
			if column != ABin.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != ABin.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	myFile = open("./data/B.dat", "w")
	myFile.write("[")
	for row in range(B.shape[0]):
		myFile.write("[")
		for column in range(B.shape[1]):
			myFile.write(str(B[row, column]))
			if column != B.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != B.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	myFile = open("./data/BBin.dat", "w")
	myFile.write("[")
	for row in range(BBin.shape[0]):
		myFile.write("[")
		for column in range(BBin.shape[1]):
			myFile.write(str(BBin[row, column]))
			if column != BBin.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != BBin.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	myFile = open("./data/W2.dat", "w")
	myFile.write("[")
	for row in range(W.shape[0]):
		myFile.write("[")
		for column in range(W.shape[1]):
			myFile.write(str(W[row, column]))
			if column != W.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != W.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	# construct rhs
	Percentage = 0.2
	QSupply = np.zeros(HORIZON)
	for t in range(HORIZON):
		QSupply[t] = (1 - Percentage) * Q0 + 2 * Percentage * Q0 * np.random.sample()
		if(QSupply[t] <= (MINUP + 1) * CLOWER):
			QSupply[t] = (MINUP + 2) * CLOWER
		
	rhs = np.zeros(( HORIZON, A.shape[0] ))
	
	print("QSupply:", QSupply)
	print("rhs.shape: ", rhs.shape)
	
	myFile = open("./data/QSupply.dat", "w")
	myFile.write(str(list(QSupply)))
	myFile.close()	
	
	for t in range(HORIZON):
		rhs[t] = list ( np.hstack(( np.zeros((SUBPERIOD)), -np.ones((SUBPERIOD)), np.zeros((5 * SUBPERIOD)),
						(-1) * np.ones((2 * SUBPERIOD)) * VUPPER, -QSupply[t], np.zeros((  numPieceSegment * SUBPERIOD )) )) )

	print(rhs)

	myFile = open("./data/rhs.dat", "w")
	myFile.write("[")
	for row in range(rhs.shape[0]):
		myFile.write("[")
		for column in range(rhs.shape[1]):
			myFile.write(str(rhs[row, column]))
			if column != rhs.shape[1] - 1:
				myFile.write(",")
		myFile.write("]")
		if row != rhs.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()

	# uncertaintySource
	uncertaintySource = np.zeros(8, dtype=int)	# the uncertainty source can be from c, d1, d2, A, B, W1, W2, b
	uncertaintySource[0] = 1	# uncertainty source is the coefficients of objective function.

	myFile = open("./data/uncertaintySource.dat", "w")
	myFile.write("[")
	for i in range(uncertaintySource.shape[0]):
		myFile.write(str(uncertaintySource[i]))
		if i != uncertaintySource.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()


	# construct the scenarios at each stage, we have uncertain electricity price
	rndPercent = 0.6
	scenarios = [[0] * numScen[i] for i in range(HORIZON)]
	scenarios[0][0] = [0] * ( nType * SUBPERIOD )
	
	# for the root node scenario
	temp2 = np.zeros(SUBPERIOD)
	for d in range(SUBPERIOD):
		temp2[d] = (1 - rndPercent) * NominalPrice[0][d] + 2 * rndPercent * NominalPrice[0][d] * np.random.sample((1,))
	
	temp2 = np.hstack(( SUCOST * np.ones(SUBPERIOD),  SDCOST * np.ones(SUBPERIOD), np.zeros(SUBPERIOD), -temp2 ))
	scenarios[0][0] = list(temp2)
	
	for t in range(1, HORIZON):
		for j in range(numScen[t]):
			temp = np.zeros(SUBPERIOD)
			for d in range(SUBPERIOD):
				temp[d] = (1 - rndPercent) * NominalPrice[t][d] + 2 * rndPercent * NominalPrice[t][d] * np.random.sample((1,))
			temp = np.hstack(( SUCOST * np.ones(SUBPERIOD),  SDCOST * np.ones(SUBPERIOD),  np.zeros(SUBPERIOD), -temp ))
			
			scenarios[t][j] = list(temp)

	print( "scenarios.row_length = ", len(scenarios), "; column_length = ", len(scenarios[1][0]) )

	scenariosBin = [[0] * numScen[i] for i in range(HORIZON)]
	scenariosBin[0][0] = [0] * ( (nType - 1 + numBinExp) * SUBPERIOD )
	
	tempBin2_temp = np.zeros( SUBPERIOD )
	for d in range(SUBPERIOD):
		tempBin2_temp[d] = (1 - rndPercent) * NominalPrice[0][d] + 2 * rndPercent * NominalPrice[0][d] * np.random.sample((1,))
	
	tempBin2 = ( tempBin2_temp[:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()	
	tempBin2 = np.hstack(( SUCOST * np.ones(SUBPERIOD), SDCOST * np.ones(SUBPERIOD), np.zeros(SUBPERIOD), -tempBin2 ))
	scenariosBin[0][0] = list(tempBin2)

	for t in range(1, HORIZON):
		for j in range(numScen[t]):
			tempBin_temp = np.zeros( SUBPERIOD )
			for d in range(SUBPERIOD):
				tempBin_temp[d] = (1 - rndPercent) * NominalPrice[t][d] + 2 * rndPercent * NominalPrice[t][d] * np.random.sample((1,))

			tempBin = ( tempBin_temp[:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()			
			tempBin = np.hstack(( SUCOST * np.ones(SUBPERIOD),  SDCOST * np.ones(SUBPERIOD),  np.zeros(SUBPERIOD), -tempBin ))
			scenariosBin[t][j] = list(tempBin)

	print("scenariosBin.row_length = ", len(scenariosBin), "; column_length = ", len(scenariosBin[1][0]) )
	
    # scenarios
	myFile = open("./data/scenX.dat", "w")
	myFile.write(str(scenarios))
	myFile.close()

	myFile = open("./data/scenXBin.dat", "w")
	myFile.write(str(scenariosBin))
	myFile.close()
	
	
	# construct EVALUATION scenarios at each stage, we have uncertain electricity price
	scenarios_eval = [[0] * numScen[i] for i in range(HORIZON)]
	scenarios_eval[0][0] = [0] * (nType * SUBPERIOD )
	
	temp2 = np.zeros(SUBPERIOD)
	for d in range(SUBPERIOD):
		temp2[d] = (1 - rndPercent) * NominalPrice[0][d] + 2 * rndPercent * NominalPrice[0][d] * np.random.sample((1,))

	temp2 = np.hstack(( SUCOST * np.ones(SUBPERIOD), SDCOST * np.ones(SUBPERIOD), np.zeros(SUBPERIOD), -temp2 ))
	scenarios_eval[0][0] = list(temp2)
	
	for t in range(1, HORIZON):
		for j in range(numScen[t]):
			
			temp = np.zeros(SUBPERIOD)
			for d in range(SUBPERIOD):
				temp[d] = (1 - rndPercent) * NominalPrice[t][d] + 2 * rndPercent * NominalPrice[t][d] * np.random.sample((1,))
			
			temp = np.hstack(( SUCOST * np.ones(SUBPERIOD), SDCOST * np.ones(SUBPERIOD),  np.zeros(SUBPERIOD), -temp ))
			scenarios_eval[t][j] = list(temp)

	print("scenarios_eval.row_length = ", len(scenarios_eval), "; column_length = ", len(scenarios_eval[1][0]) )

	scenarios_evalBin = [[0] * numScen[i] for i in range(HORIZON)]
	scenarios_evalBin[0][0] = [0] * ( (nType - 1 + numBinExp) * SUBPERIOD )
	
	tempBin2_temp = np.zeros( SUBPERIOD )
	for d in range(SUBPERIOD):
		tempBin2_temp[d] = (1 - rndPercent) * NominalPrice[0][d] + 2 * rndPercent * NominalPrice[0][d] * np.random.sample((1,))
	
	tempBin2 = ( tempBin2_temp[:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()
	
	tempBin2 = np.hstack(( SUCOST * np.ones(SUBPERIOD),  SDCOST * np.ones(SUBPERIOD), np.zeros(SUBPERIOD), -tempBin2 ))
	scenarios_evalBin[0][0] = list(tempBin2)

	for t in range(1, HORIZON):
		for j in range(numScen[t]):

			tempBin_temp = np.zeros( SUBPERIOD )
			for d in range(SUBPERIOD):
				tempBin_temp[d] = (1 - rndPercent) * NominalPrice[t][d] + 2 * rndPercent * NominalPrice[t][d] * np.random.sample((1,))

			tempBin = ( tempBin_temp[:, np.newaxis] * matBin2[0][0 : numBinExp] ).flatten()  
			
			tempBin = np.hstack(( SUCOST * np.ones(SUBPERIOD),  SDCOST * np.ones(SUBPERIOD),  np.zeros(SUBPERIOD), -tempBin ))
			scenarios_evalBin[t][j] = list(tempBin)

	print("scenarios_evalBin.row_length = ", len(scenarios_evalBin), "; column_length = ", len(scenarios_evalBin[1][0]) )
	
    # scenarios_eval
	myFile = open("./data/scenX_eval.dat", "w")
	myFile.write(str(scenarios_eval))
	myFile.close()

	myFile = open("./data/scenXBin_eval.dat", "w")
	myFile.write(str(scenarios_evalBin))
	myFile.close()
	
	
	# generate offline initial state
	initState = np.zeros(nType * SUBPERIOD, dtype=int)
	print("offline initState = ", initState)
	initStateBin = np.zeros( (nType - 1) * SUBPERIOD + numBinExp * SUBPERIOD, dtype=int )
	print("offline initStateBin = ", initStateBin)
	
	# generate online initial state
	initState[0] = 1 # start up at the first time period
	initStateBin[0] = 1 # start up at the first time period
	initState[2 * SUBPERIOD: 3 * SUBPERIOD] = np.ones(SUBPERIOD, dtype=int)		# stay online
	initStateBin[2 * SUBPERIOD: 3 * SUBPERIOD] = np.ones(SUBPERIOD, dtype=int)		# stay online
	
	initState[3 * SUBPERIOD : nType * SUBPERIOD] = CLOWER * np.ones(SUBPERIOD)		# power output
	
	for i in range(SUBPERIOD):
		for j in range(numBinExp):
			initStateBin[3 * SUBPERIOD + i * numBinExp + j] = int( np.binary_repr(int(CLOWER / EPLISON), width=numBinExp)[numBinExp - 1 - j] )

	print("online initState = ", initState)
	print("online initStateBin = ", initStateBin)
			
	# the initial state of state variables in the original space
	myFile = open("./data/initState.dat", "w")
	myFile.write("[")
	for i in range(initState.shape[0]):
		myFile.write(str(initState[i]))
		if i != initState.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
	
	# the initial state of state variables in the binary expansion space
	myFile = open("./data/initStateBin.dat", "w")
	myFile.write("[")
	for i in range(initStateBin.shape[0]):
		myFile.write(str(initStateBin[i]))
		if i != initStateBin.shape[0] - 1:
			myFile.write(",")
	myFile.write("]")
	myFile.close()
