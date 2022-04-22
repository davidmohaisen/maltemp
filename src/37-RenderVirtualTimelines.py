import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
import scipy as sp
from sklearn.neighbors import LocalOutlierFactor





def DistributeData(data,Fam,FamStart):
    FamSamplesCounter = [0]*10
    steps , labels = [], []
    consideredFamilies = []
    for step in range(100):
        steps.append([])
        labels.append([])

        if step in FamStart:
            consideredFamilies.append(list(FamStart).index(step))
            # print("considering",consideredFamilies,"at",step)
            for j in range(8):
                steps[-1].append(data[list(FamStart).index(step)][FamSamplesCounter[list(FamStart).index(step)]])
                labels[-1].append(list(FamStart).index(step))
                FamSamplesCounter[list(FamStart).index(step)] += 1
        for i in consideredFamilies:
            # if step == 0:
            #     for j in range(8):
            #         steps[-1].append(data[i][FamSamplesCounter[i]])
            #         labels[-1].append(i)
            #         FamSamplesCounter[i] += 1
            steps[-1].append(data[i][FamSamplesCounter[i]])
            labels[-1].append(i)
            FamSamplesCounter[i] += 1
        steps[-1] = np.asarray(steps[-1])
        # print("Step",step,"finished")
        # print(labels)
        # print(FamSamplesCounter)

    return steps , labels

TotalFalseAcceptanceRate = 0
TotalAverageFamilyDetection = 0
TotalAverageStepsError = 0

f = open("../Pickles/VirtualTimeline/EmberData","rb")
data,Fam = pickle.load(f)


Metrics_List = []
Actual_Distribution = []
Predicted_Distribution = []

TrialsNumber  = 1000
for render in range(TrialsNumber):
    print("Trial",render+1)
    FamStart = None
    while True:
        FamStart = np.random.randint(1,100, size=10)
        FamStart[FamStart == min(FamStart)] = 0
        if len(set(list(FamStart))) == len(FamStart):
            break

    # FamStart.sort()
    # print("Family rendering start steps:",FamStart)
    steps , labels = DistributeData(data,Fam,FamStart)
    # for i in range(len(steps)):
    #     print(np.asarray(steps[i]).shape,labels[i])
    # exit()
    correct = 0
    all = 0
    P = 0
    TP = 0
    N = 0
    TN = 0
    predictedSteps = [0]

    dicPredicted = {list(FamStart).index(0):0}
    # print(dicPredicted)
    for i in range(len(steps)-1):
        data_to_step = []
        for j in range(i+1):
            if len(data_to_step) == 0:
                data_to_step = list(steps[j])
            else:
                data_to_step += list(steps[j])
        data_to_step = np.asarray(data_to_step)
        # print(i,data_to_step.shape)
        lof = LocalOutlierFactor(metric="cosine",novelty=True,n_neighbors=8,leaf_size=10000,n_jobs=-1,contamination=0.03)
        lof.fit(data_to_step)
        preds = lof.predict(steps[i+1])
        if list(preds).count(-1) != 0:
            predictedSteps.append(i+1)
            for j in range(len(preds)):
                if preds[j] == -1 and labels[i+1][j] not in dicPredicted.keys():
                    dicPredicted[labels[i+1][j]] = i+1
        # print(preds,labels[i])

        for j in range(len(preds)):
            all+=1
            if j < len(labels[i]):
                N += 1
                if preds[j] == 1:
                    correct += 1
                    TN += 1
            elif j >= len(labels[i]):
                P += 1
                if preds[j] == -1:
                    TP += 1
                    correct += 1
    # print("Accuracy",correct/all)
    # print("TPR",TP/P)

    actualSteps = FamStart.copy()
    actualSteps.sort()
    dicActual = {}
    for i in range(len(FamStart)):
        dicActual[i] = FamStart[i]

    FAR = 1-(TN/N)
    AverageFamilyDetection = len(dicPredicted)/len(dicActual)
    AverageStepsError = 0
    for key in dicPredicted.keys():
        AverageStepsError += dicPredicted[key] - dicActual[key]
    AverageStepsError = AverageStepsError/len(dicActual)
    print("False Acceptance Rate",FAR)
    print("Average Family Detection",AverageFamilyDetection)
    print("Average Steps Error",AverageStepsError)
    print("Actual Steps:",dicActual)
    print("Predicted Steps:",dicPredicted)
    TotalFalseAcceptanceRate += FAR
    TotalAverageFamilyDetection += AverageFamilyDetection
    TotalAverageStepsError += AverageStepsError
    Metrics_List.append([FAR,AverageFamilyDetection,AverageStepsError])
    Actual_Distribution.append(dicActual)
    Predicted_Distribution.append(dicPredicted)

TotalFalseAcceptanceRate /= TrialsNumber
TotalAverageFamilyDetection /= TrialsNumber
TotalAverageStepsError /= TrialsNumber
print("Total False Acceptance Rate",TotalFalseAcceptanceRate)
print("Total Average Family Detection",TotalAverageFamilyDetection)
print("Total Average Steps Error",TotalAverageStepsError)

f = open("../Pickles/VirtualTimeline/VirtualSimulationResults2","wb")
pickle.dump([Metrics_List,Actual_Distribution,Predicted_Distribution],f)
