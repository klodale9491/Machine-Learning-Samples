'''
Coder : Alessio Giorgianni
Email : alessio.giorgianni.1991@gmail.com

This class emulates the beahaviour of a bayesian classifier.
'''

import csv
import math

class BayesianClassifier:

    '''
        nc : number of classes
        tsr : Training set ration between 0 and 1
    '''
    def __init__(self,nc,tsr,filename):
        self._loadCsv(filename)
        self.initClasses(nc)
        self._splitDS(tsr)
        self._train()
        self.run()

    # Init classes
    def initClasses(self,n):
        self.classes = []
        for i in range(0,n):
            self.classes.append([])

    # Load dataset from csv file
    def _loadCsv(self,filename):
        lines = csv.reader(open(filename, "r"))
        self.dataset = list(lines)
        for i in range(len(self.dataset)): # The last is the classId, not convert
            self.dataset[i] = [float(x) for x in self.dataset[i]]

    # Get trainingset like a ratio of original dataset
    def _splitDS(self,ratio):
        copy = list(self.dataset)
        self.trainset = []
        if ratio >= 0.0 and ratio <= 1.0:
            trainlen = int(len(self.dataset) * ratio)
            for i in range(0, trainlen):
                self.trainset.append(self.dataset[i])
            self.dataset = copy

    # Return an array of attribute values by class/dataset
    def _getAttrVal(self,index,dataset):
        return [row[index] for row in dataset]

    # Calculate mean of a set o values
    def _mean(self,values):
        return float(sum(values))/float(len(values))

    # Calculate stdev of a set o values
    def _stdev(self,values):
        mean = self._mean(values)
        variance = sum([pow(x-mean,2) for x in values])/float(len(values)-1)
        return math.sqrt(variance)

    # Use a Gaussian Distribution to estimate feature value probability
    def _getValueProbability(self,x,mean,stdev):
        exponent = float(math.exp(-(math.pow(x - mean, 2)) / float((2 * math.pow(stdev, 2)))))
        return (1 / (math.sqrt(2 * math.pi) * float(stdev))) * exponent

    # Train classifier
    def _train(self):
        for i in range(0,len(self.trainset)):
            classIndex = int(self.trainset[i][-1])
            if classIndex < len(self.classes):
                self.classes[classIndex].append(self.trainset[i])

    # Get distribution of global attribute, [mean,stdev], to calculate P(Xj)
    def _getDistributionByAttribute(self,attributeIndex):
        values = self._getAttrVal(attributeIndex,self.dataset)
        return self._mean(values),self._stdev(values)

    # Get distribution of global attribute, [mean,stdev], to calculate P(Xj | Ci)
    def _getDistributionByAttributeClass(self,attributeIndex,classIndex):
        values = self._getAttrVal(attributeIndex,self.classes[classIndex])
        return self._mean(values), self._stdev(values)

    # Get class probability, based on only the number of elements inside, to calculate P(Ci)
    def _getClassProbability(self,classIndex):
        return float(len(self.classes[classIndex])/float(len(self.trainset)))

    # Predict which class the element will belong to
    def _predict(self,x):
        map = 0 # Maximum A Posteriori Probability
        classId = 0 # Current class to belong to
        # Calculate Maximum A Posteriori Probability
        for j in range(0,len(self.classes)): # foreach classes
            px = 1
            px_ci = 1
            for i in range(0,len(x) - 1): # forach attributes
                # Calculate Likelihood P(Xj | Ci)
                mean,stdev = self._getDistributionByAttributeClass(i,j)
                pxj_ci = self._getValueProbability(x[i],mean,stdev)
                px_ci *= pxj_ci
                # Calculate P(X)
                mean, stdev = self._getDistributionByAttribute(i)
                pxj = self._getValueProbability(x[i],mean,stdev)
                px *= pxj
            # Calculate P(Ci)
            pci = self._getClassProbability(j)
            # Calculate current A Posteriori Probability
            pci_x = float(px_ci * pci)/float(px)
            if pci_x > map:
                map = pci_x
                classId = j
        return classId

    def run(self):
        i = 0 # class 0
        j = 0 # class 1
        index = 0
        score = 0.0
        for x in self.dataset:
            exactClass = self.dataset[index][-1]
            predictedClass = self._predict(x)
            if(predictedClass == 0):
                i += 1
            if(predictedClass == 1):
                j += 1
            # Calculate precision
            if exactClass == predictedClass:
                score += 1
            index += 1

        print([float(score)/float(len(self.dataset)),i,j])
        return [i,j]


c = BayesianClassifier(2,0.7,"diabetes-dataset.csv")