#Main methods for creating the transfer attacks 
import torch
import torchvision
import numpy
import DataManagerPytorch as DMP
import AttackWrappersWhiteBoxP
from datetime import date
import os 
import ImageNetLoader

#Create the transfer 
def GenerateTransferResultsImageNetSingleColumn(saveTag, attackerModelIndex, modelPlusList, device, attackSampleNum):
    #Setup file to save the results
    today = date.today()
    dateString = today.strftime("%B"+"-"+"%d"+"-"+"%Y, ")
    experimentDateAndName = dateString + saveTag #Name of experiment with data 
    saveDir = os.path.join(os.getcwd(), experimentDateAndName)
    if not os.path.isdir(saveDir): #If not there, make the directory 
        os.makedirs(saveDir)
    #Place to save the results 
    os.chdir(saveDir)
    resultsTextFile = open(experimentDateAndName+", Results.txt","a+")
    #Get the data in its original form without resizing 
    valData, yData = ImageNetLoader.GetRawImageNet()
    numModels = len(modelPlusList)
    #Go through and attack each model 
    for i in range(0, numModels):
        if i == attackerModelIndex:
            print("skip")
        else:
            modelPlusA = modelPlusList[attackerModelIndex]
            modelPlusB = modelPlusList[i]
            GenerateTransferAttacksAToB(device, modelPlusA, modelPlusB, attackSampleNum, valData, yData, resultsTextFile)
            modelPlusList[i].clearModel() #clear the model from memory because we don't need it anymore
    #Do some house keeping
    resultsTextFile.close() #Close the results file at the end 
    os.chdir("..") #move up one directory to return to original directory 

#Generate using model A, test on model B
def GenerateTransferAttacksAToB(device, modelPlusA, modelPlusB, attackSampleNum, valData, yData, resultsTextFile):
    modelAName = modelPlusA.modelName
    modelBName = modelPlusB.modelName
    #Find the overlapping clean samples 
    cleanLoader = GetFirstCorrectlyOverlappingSamplesRolling(modelPlusA, modelPlusB, attackSampleNum, valData, yData)
    #Everything correct so go ahead and save the clean data 
    torch.save(cleanLoader, "CleanLoader"+modelAName+modelBName)
    RunAllAttacks(cleanLoader, device, modelPlusA, modelPlusB, resultsTextFile)
    #clean up
    del cleanLoader
    torch.cuda.empty_cache() 


#Return a dataloader that has the samples that are correctly identified by both networks 
def GetFirstCorrectlyOverlappingSamplesRolling(modelPlusA, modelPlusB, attackSampleNum, valData, yData):
    #Basic variable setup 
    requiredImgSize = modelPlusA.imgSize
    rs = torchvision.transforms.Resize((requiredImgSize, requiredImgSize)) #We are using modelA to do the attack so need images to correspond to the right size
    t = torchvision.transforms.ToTensor()
    #Generate memory for the solution 
    xData = torch.zeros(attackSampleNum, 3, requiredImgSize, requiredImgSize) #Here I assume color channels are 3 
    yDataFinal = torch.zeros(attackSampleNum)
    currentNumSampleSaved = 0
    for i in range(0, 50000):
        print("Running sample=", i)
        x = rs(valData[i][0]) #change x into a the right shape  
        yPredA = modelPlusA.SinglePredictPILImage(x) 
        yPredB = modelPlusB.SinglePredictPILImage(x)
        #both samples recognize the sample as the same class and it is the correct class 
        if yPredA.argmax(axis=1) == yPredB.argmax(axis=1) and yPredA.argmax(axis=1)==yData[i]:
            print("Sample "+str(i)+" recognized correctly.")
            xData[currentNumSampleSaved] = t(rs(valData[i][0]))
            yDataFinal[currentNumSampleSaved] = yData[i]
            currentNumSampleSaved = currentNumSampleSaved +1
            print("Save count=", currentNumSampleSaved)
        #Check to see if we can stop 
        if currentNumSampleSaved == attackSampleNum:
            dataLoaderFinal = DMP.TensorToDataLoader(xData, yDataFinal, transforms = None, batchSize = modelPlusA.batchSize, randomizer = None)
            return dataLoaderFinal
    #Should never reach this point 
    if currentNumSampleSaved != attackSampleNum:
        raise ValueError("Not enough samples found!")

def RunAllAttacks(cleanLoader, device, modelPlusA, modelPlusB, resultsTextFile):
    attackList = ["FGSM", "MIM", "PGD"]
    modelAName = modelPlusA.modelName
    modelBName = modelPlusB.modelName
    resultsTextFile.write("Generated From: "+modelAName+"\n")
    #Go through and run all the attacks 
    for i in range(0, len(attackList)):
        attackName = attackList[i] #Get the attack name 
        advLoader = RunSingleAttackImageNet(attackName, cleanLoader, device, modelPlusA) #Run the attack
        torch.save(advLoader, "G="+modelAName+",T="+modelBName+","+attackName)
        accA = modelPlusA.validateD(advLoader)
        accB = modelPlusB.validateD(advLoader)
        print("Robust Acc "+modelAName+" "+attackName+"=", accA)
        print("Robust Acc "+modelBName+" "+attackName+"=", accB)
        #Write the results to text file 
        resultsTextFile.write("Robust Acc "+modelAName+" "+attackName+"="+str(accA)+"\n")
        resultsTextFile.write("Robust Acc "+modelBName+" "+attackName+"="+str(accB)+"\n")
        resultsTextFile.write("==========="+"\n")
        #just to be safe
        del advLoader
        torch.cuda.empty_cache()

def RunSingleAttackImageNet(attackName, cleanLoader, device, modelPlusA):
    #Image range should be 0-1 for these attacks 
    decayFactor = 1.0
    epsilonMax = 0.062
    numSteps = 10
    epsilonStep = epsilonMax/float(numSteps)
    clipMin = 0
    clipMax = 1.0
    targeted = False
    #Put the model we are attacking on the GPU
    currentModel = modelPlusA.model
    currentModel.to(device)
    #Figure out what attack to do 
    if attackName == "FGSM":
        advLoader = AttackWrappersWhiteBoxP.FGSMNativePytorch(device, cleanLoader, currentModel, epsilonMax, clipMin, clipMax, targeted)
    elif attackName == "MIM":
        advLoader = AttackWrappersWhiteBoxP.MIMNativePytorch(device, cleanLoader, currentModel, decayFactor, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted)
    elif attackName == "PGD":
        advLoader = AttackWrappersWhiteBoxP.PGDAttackFoolBox(device, cleanLoader, currentModel, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted)
    elif attackName == "EAD":
        #EAD requires changing some parameters 
        epsMax = 50
        binarySearchSteps = 9
        steps = 1000
        initialStepSize = 0.01
        confidence = 0.0
        initialConst = 0.001
        regularization = 0.01
        decisionRule = "L1"
        abortEarly = True
        advLoader = AttackWrappersWhiteBoxP.EADAttackFoolBox(device, cleanLoader, currentModel, epsMax, binarySearchSteps, steps, initialStepSize, confidence, initialConst, regularization, decisionRule, abortEarly, clipMin, clipMax, targeted)    
        xAdv, yAdv = DMP.DataLoaderToTensor(advLoader)
        #DMP.ShowImages(xAdv.permute(0,2,3,1).numpy(), xAdv.permute(0,2,3,1).numpy())
    else:
        print("Attack name not recognized.")
    #Delete the model because we no longer need it 
    del currentModel
    torch.cuda.empty_cache()
    return advLoader