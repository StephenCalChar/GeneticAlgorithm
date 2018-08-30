import re
from random import randint
import unicodecsv as csv
import numpy as np
import math


#Declare file to be imported here.
inputFile = 'training.txt'


class Tweet:
    """Tweet Class which contains a list of the words in the tweet and its sentimental score as properties."""
    sentimentScore = None
    words = []
    def __init__(self, sentimentScore, words):
        self.sentimentScore = sentimentScore
        self.words = words


class Chromosome:
    """Chromosome Class with its fitness score percentage and a dictionary of words and their sentimental scores, as
    properties."""
    fitnessScore = 0.00
    wordsAndValues = {}
    def __init__(self, fitnessScore, wordsAndValues):
        self.fitnessScore = fitnessScore
        self.wordsAndValues = wordsAndValues


def assessFitness(chromosome, tweetObjectArray):
    """Method which takes a chromosome and the list of all tweet objects and returns the percentage accuracy of tweets
    correctly identified by using  the dictionary of words and word sentimental score values, to predict each tweets
    sentiment. """
    correctAnswers = 0
    wrongAnswers = 0
    for tweet in tweetObjectArray:
        targetAnswer = int(tweet.sentimentScore)
        #Starting the cumulative sentiment, starting at 0 favors negative but cant have 0.5
        cumulativeSentiment = 0
        for word in tweet.words:
            if word in chromosome.wordsAndValues:
                cumulativeSentiment += chromosome.wordsAndValues[word]
        # The algorithm i going to classify a total score of 0 as negative as there is no neutral in this experiment.
        if cumulativeSentiment < 0:
            cumulativeSentiment =0
        if cumulativeSentiment > 1:
            cumulativeSentiment = 1
        if cumulativeSentiment == targetAnswer:
            correctAnswers += 1
        else:
            wrongAnswers +=1

    # returns number of correct answers as a percentage, kept as a float for accuracy.
    answer = (float(correctAnswers)/(float(len(tweetObjectArray)))*100)
    return answer


def calculateFScore(chromosome, tweetObjectArray):
    """Method which takes a chromosome and the list of all tweet objects and returns the Positive F-Score of tweets
    correctly identified by using the dictionary of words and word sentimental score values, to predict each tweets
     sentiment. """
    correctNegatives =0
    correctPositives =0
    falseNegatives =0
    falsePositives =0
    for tweet in tweetObjectArray:
        targetAnswer = int(tweet.sentimentScore)
        cumulativeSentiment = 0
        for word in tweet.words:
            if word in chromosome.wordsAndValues:
                cumulativeSentiment += chromosome.wordsAndValues[word]
        # The algorithm i going to classify a total score of 0 as negative as there is no neutral in this experiment.
        if cumulativeSentiment < 0:
            cumulativeSentiment = 0
        if cumulativeSentiment > 1:
            cumulativeSentiment = 1
        #check positives tweets, checks whether it is a correct positive or a false negative.
        if targetAnswer == 1:
            if cumulativeSentiment ==1:
                correctPositives +=1
            if cumulativeSentiment ==0:
                falseNegatives +=1
        # looks at negative tweets, checks if the answer is a correct negative or a false positive.
        if targetAnswer ==0:
            if cumulativeSentiment ==0:
                correctNegatives +=1
            if cumulativeSentiment ==1:
                falsePositives +=1
    # start of building fscore calculation
    precisionScore = float(float(correctPositives) /(correctPositives + falsePositives))
    recallScore = float(float(correctPositives)/(correctPositives + falseNegatives))
    fScore = 2.0 * ((precisionScore * recallScore)/(precisionScore+recallScore))
    return fScore


def chooseParent(numberOfChromosomes):
    """Selects the index position of a chromosome that will be used as a parent in the crossover. The chromosomes are
    ordered based on their fitness scores, with the least fit appearing in index 0. The higher the chromosome's fitness
    score, then the higher the probablity they will be selected. It is possible that the least fit chromosomes may
    contain some genes that are worth passing on so there is still a small chance that they are selected."""
    ranNumber = randint(0,100)
    #calculates the position number of the last chromosome inside list.
    totalChromIndex = numberOfChromosomes -1
    #These floats have been chosen in order to split the list positions up. with Ceiling and floor being used to round
    #to the whole index position either size of the calculation result. Examples of what positions these would be for an
    #example sized list of 30, is shown below.
    highestFitnessTier = totalChromIndex * .83
    midFitnessTier = totalChromIndex * 0.49
    if ranNumber <= 50:
        result = randint(math.ceil(highestFitnessTier), totalChromIndex)
        #For 30 chromsoomes this will random between(25, 29)
        return result
    if ranNumber <= 80:
        result = randint(math.ceil(midFitnessTier), math.floor(highestFitnessTier))
        # For 30 chromosomes this will random between(15, 24)
        return result
    else:
        result = randint(0, math.floor(midFitnessTier))
        # For 30 chromosomes this will random between(0, 14)
        return result


def chooseReplacementIndex(numberOfChromosomes):
    """Selects the index position of a chromosome that will be potentially replaced by a new child. The chromosomes are
       ordered based on their fitness scores, with the least fit appearing in index 0. The lower the chromosome's
       fitness score, then the higher the probablity it will be selected. None of the top 50% fittest chromosomes
       will be selected for replacement as they will contain the fittest DNA which should not be lost."""
    ranNumber = randint(0, 100)
    # calculates the position number of the last chromosome inside list.
    totalChromIndex = numberOfChromosomes - 1
    lowestFitnessTier = totalChromIndex * .17
    midFitnessTier = totalChromIndex * .31
    highestFitnessTier = totalChromIndex * 0.49
    if ranNumber <= 50:
        result = randint(0, math.floor(lowestFitnessTier))
        #For 30 chromosomes this will random between(0, 4)
        return result
    if ranNumber <= 90:
        result = randint(math.ceil(lowestFitnessTier), math.ceil(midFitnessTier))
        #For 30 chromosomes this will random between(5, 9)
        return result
    else:
        result = randint(math.floor(midFitnessTier), math.floor(highestFitnessTier))
        #For 30 chromosomes this will random between(10, 14)
        return result


def createNewChildDic(parentOne, parentTwo, mutationRate):
    """Creates a the child chromosome's dictionary values from the two parent chromosomes passed to the function.
     Use the mutation rate parameter, in the crossover so the rate at which a gene is changed can be set easily.
     Returns one dictionary, which can be inserted into the child chromosome."""
    maxValue = 100
    tempDic ={}
    mutationRate *= 100
    for k, v in parentOne.wordsAndValues.iteritems():
        #random number between 1 and 100.
        randomNum = randint(1, maxValue)
        #mutates value to a random number if randomNum is less than the mutation rate.
        if randomNum <= mutationRate:
            tempDic[k] = randomNumber()
        # ((maxValue + mutationRate)/2) should split the remaining results into 50/50 chance of choosing a gene from
        # either of the 2 parents. This way reuses randomNum's valuable without having to re-randomise.
        if randomNum < ((maxValue + mutationRate)/2):
            tempDic[k] = parentOne.wordsAndValues[k]
        else:
            tempDic[k] = parentTwo.wordsAndValues[k]
    return tempDic


#Create randomise between -5 and 5 function. This is where min and max values can be changed.
def randomNumber():
    """Creates a random number between -5 and 5. This is where min and max values could be changed for each words
    sentiment score."""
    randomNum = randint(-5,5)
    return randomNum


def preProcessing(tweets):
    """This method handles all of the pre-processing. It takes a list of tweets as a parameter, cleans them by
    removing punctuation, and splitting the tweet into a list of individual lower case words. Also implements a check
    for negation in bigrams."""
    # new List of tweets, to be populated with tweets which are lists comprised of their words.
    cleanedTweets = []
    # Boolean flag used in order to pre-concatenate words after a negation term.
    nextWordChange = False
    # Cleans up any negation by checking for "n't" and "Not". It removed them and prepends next word with "NOT".
    for line in tweets:
        words = line.split()
        tempTweetArray = []
        for word in words:
            word = word.lower()
            if nextWordChange == True:
                word = "NOT" + word
                # for each word in the words array remove any special characters and replace them with nothing.
                word = re.sub('[^A-Za-z0-9]+', '', word)
                # only appends to tweet list if string is not empty
                if word != "":
                    tempTweetArray.append(word)
                nextWordChange = False
            elif word.find("n't") != -1 or word.find("not") != -1:
                nextWordChange = True
            else:
                # for each word in the words list remove any special characters and replace them with nothing.
                word = re.sub('[^A-Za-z0-9]+', '', word)
                # only appends if string is not empty
                if word != "":
                    tempTweetArray.append(word)
                nextWordChange == False
        cleanedTweets.append(tempTweetArray)
    return cleanedTweets


# START OF PROGRAM

# Loads File
with open(inputFile, 'r') as loadedFile:
    data = loadedFile.readlines()
loadedFile.close()


#shuffles/randomises lines/tweets so that each experiment is different.
np.random.shuffle(data)


#Declares variales used for the KFold Test
kFoldValue =10
kFoldArrays =[]
totalKFoldFScore =0
totalKFoldAccuracy =0
totalLines = len(data)
kFoldSegment = totalLines/kFoldValue




# Loop which will split data into 10 equal sized arrays (the last one may be slightly larger or smaller depending on line count)
counter = 0
while counter < kFoldValue:
    #index for slicing start.
    fromValue = counter * kFoldSegment
    #index for slicing end.
    toValue = (counter + 1) * kFoldSegment
    #on last iteration, we ensure that the final slice index is the last line  we have imported in, so no tweets are missed.
    if counter ==(kFoldValue-1):
        toValue= totalLines
    #The slicing and appending to a new array.
    kFoldSingleArray = data[fromValue:toValue]
    kFoldArrays.append(kFoldSingleArray)
    counter +=1

##Main KFold Loop where the testing occurrs
kFoldCounter =0
while kFoldCounter < kFoldValue:
    #arrays to be populated with the Kfold data
    trainingData =[]
    testData =[]
    for array in kFoldArrays:
        #if index of array matches the counter then it will be the test data. If not it will be added to the training data array
        if kFoldArrays.index(array) ==kFoldCounter:
            testData = array
        else:
            trainingData.extend(array)

    ##Only used for 75%/25% Split of training and test data. Currently commented out as not in use for KFold.
    """
    tweetsToBeTested = int(round(0.25 * len(data)))
    #Creates Test Data
    testData= data[:tweetsToBeTested]
    #Creates Training Data
    trainingData = data[tweetsToBeTested:]
    """

    ## main variables declared here
    ratings = []
    tweets =[]
    # mutation rate is set here as a decimal percentage.
    mutationRate = 0.15
    numberOfIterations = 3000
    numberOfChromosomes = 40
    minWordOccurrenceForInclusion = 2



    #split out the ratings using slice and add to the ratings array and split out the reviews and add to the tweets array.
    for line in trainingData:
        #going to convert to int here for good measure
        ratings.append(int(line[:1]))
        tweets.append(line[2:])

    # new Array of tweet word arrays to be populated with negation words omitted.
    cleanedTweets = preProcessing(tweets)

    #Array that will store Tweet objects
    tweetObjectArray =[]

    #creates new Tweet objects which will hold the sentiment score and list of words in each tweet as properties.
    x =0
    while x < len(cleanedTweets):
        tweet = Tweet(ratings[x],cleanedTweets[x])
        tweetObjectArray.append(tweet)
        x+= 1

    #Dictionary to hold each distinct word and its appearence count in the training data. To be used when words that
    # appear less than a certain number of times have to be omitted.
    wordTotals = {}
    for tweet in tweetObjectArray:
        for word in tweet.words:
            if word not in wordTotals:
                wordTotals[word] = 1
            else:
                wordTotals[word] += 1

    finalWordArray =[]
    # Here, only words that appear a certain number of times in the training data are added to the array.
    # This can be adjusted via the minWordOccurrenceForInclusion variable
    for k,v in wordTotals.iteritems():
        if v >= minWordOccurrenceForInclusion:
            finalWordArray.append(k)


    chromosomeArray =[]
    #Create the desired number of chromosome objects and adds them to the chromosomeArray list. Each chromosomes word
    #dictionary is populated by the distinct words which are given an initial random value between +5 and -5.
    counter =0
    while counter < numberOfChromosomes:
        tempDic ={}
        for word in finalWordArray:
            tempDic[word] = randomNumber()
        chromosome = Chromosome(0, tempDic)
        chromosomeArray.append(chromosome)
        counter +=1

    ##The loop for number of iterations of the genetic algorithm occurr commences here.
    iterationCounter =0
    ##we want to activate the top half of this code 1 more time after completing the cycle, so that fresh fitness
    #dictionary's and order arrays can be populated. This is because they will be required once the loop has finished
    # in order to work out which chromosome has the highest fitness (just in case they do not all have the same fitness
    # score). The new dictionary and lists are needed in case the last crossover changed any values.
    while iterationCounter <= numberOfIterations:
        # Asses fitness score of each chromosome and calculate percentage accuracy.
        # add index of chromosome within chromosomeArray and its fitness score to a dictionary so they can be ordered
        # by fitness level while still keeping track of their index in chromosomeArray to reference later.
        # Added to fitnessDic with array index of the chromsome in chromsomearray as key and fitness score as value
        fitnessDic = {}
        indexCounter = 0
        for chromosome in chromosomeArray:
            fitPercentage = assessFitness(chromosome, tweetObjectArray)
            #store fitnessscore in chromosome object
            chromosome.fitnessScore = fitPercentage
            fitnessDic[indexCounter] = fitPercentage
            indexCounter +=1

        # Create a list with the fitness scores sorted in order from left to right(right being the best level of fitness)
        fitnessOrder =sorted(fitnessDic.values())

        ##calculates average fitness of chromosomes so we can see if this is constantly improving.
        averageFitness = (sum(fitnessOrder)/ len(fitnessOrder))
        print ("Average Fitness is : " + str(averageFitness))
        #print("Highest fitness is: " + str((fitnessOrder[len(fitnessOrder)-1])))

        ##This part of the cycle only needs to iterate the number of times specified in the cycle variable.
        #Therefore the following if statement will not activate past total cycles reached
        if iterationCounter < numberOfIterations:
            indexParentOne = -1
            indexParentTwo = -1

            #while both parents = each other select another 2. This avoids a parent crossingover with itself.
            while indexParentOne == indexParentTwo:
                indexParentOne = chooseParent(numberOfChromosomes)
                indexParentTwo = chooseParent(numberOfChromosomes)

            # Retrieve  the fitness scores of the 2 selected parents from the fitness dictionary. This will be used to
            # retrieve them from the fitnessDic.
            valueParentOne = fitnessOrder[indexParentOne]
            valueParentTwo = fitnessOrder[indexParentTwo]

            #retrieved position of the parent in the main chromsome object array
            chromosomeParentOneIndex = -1
            chromosomeParentTwoIndex = -1

            #searches for the two fitness score values taken from the fitness order list and finds the chromosomes they
            #beling to and creates a reference to each chromosome with the 2 variables chromosomeParentOneIndex &
            # chromosomeParentTwoIndex.
            for k, v in fitnessDic.iteritems():
                if v == valueParentOne:
                    chromosomeParentOneIndex = k
                elif v == valueParentTwo:
                    chromosomeParentTwoIndex = k

            #creates two new references to the parent chrosomes to be used in crossover
            parentOne = chromosomeArray[chromosomeParentOneIndex]
            parentTwo = chromosomeArray[chromosomeParentTwoIndex]


            ## creates 2 new dictionarys to be inserted into children
            tempWordValueOne = createNewChildDic(parentOne, parentTwo, mutationRate)
            tempWordValueTwo = createNewChildDic(parentOne, parentTwo, mutationRate)

            #create 2 new child chromosomes
            childChromosomeOne = Chromosome(0,tempWordValueOne)
            childChromosomeTwo = Chromosome(0,tempWordValueTwo)

            ## get values of fitness of each child and stores it in variables.
            childOneFitness = assessFitness(childChromosomeOne, tweetObjectArray)
            childTwoFitness = assessFitness(childChromosomeTwo, tweetObjectArray)

            #Adds the calculated fitness score to the chromosome object's property to store.
            childChromosomeOne.fitnessScore = childOneFitness
            childChromosomeTwo.fitnessScore = childTwoFitness

            #find 2 chromsomes to replace and returns their indexs within fitnessOrder List.
            potentialReplacedIndexOne = -1
            potentialReplacedIndexTwo = -1
            while potentialReplacedIndexOne == potentialReplacedIndexTwo:
                potentialReplacedIndexOne = chooseReplacementIndex(numberOfChromosomes)
                potentialReplacedIndexTwo = chooseReplacementIndex(numberOfChromosomes)

            # Retrieves  the fitness scores for the 2 chromosomes in the indexes returned above.
            valueReplacementOne = fitnessOrder[potentialReplacedIndexOne]
            valueReplacementTwo = fitnessOrder[potentialReplacedIndexTwo]

            #retrieved position of the parent in the main chromsome object array and stores them in variable
            chromosomeReplacementOneIndex = -1
            chromosomeReplacementTwoIndex = -1
            #Returns index within chromosome list of the 2 chromosomes selected for replacement.
            for k, v in fitnessDic.iteritems():
                if v == valueParentOne:
                    chromosomeReplacementOneIndex = k
                elif v == valueParentTwo:
                    chromosomeReplacementTwoIndex = k

            #creates a reference to the two chromosomes selected for replacement.
            possibleReplaceOne = chromosomeArray[chromosomeReplacementOneIndex]
            possibleReplaceTwo = chromosomeArray[chromosomeReplacementTwoIndex]

            #compares each of the child chromosomes against one of the chromosomes selected for replacement. If the
            # fitness of the child is higher then it replaces the selected chromsosome. If it is not fitter than nothing
            # changes in the population
            if childChromosomeOne.fitnessScore > possibleReplaceOne.fitnessScore:
                chromosomeArray[chromosomeReplacementOneIndex] = childChromosomeOne

            if childChromosomeTwo.fitnessScore > possibleReplaceTwo.fitnessScore:
                chromosomeArray[chromosomeReplacementTwoIndex] = childChromosomeTwo

        iterationCounter +=1

    #finds value of highest fitness function. As the array is ordered so the last index will contain the highest fitness
    #chromosome.
    highestFitness = fitnessOrder[len(fitnessOrder)-1]
    highestFitnessIndex = -1

    #locates the index of the highest value fitness function chromosome by looking at its key in the dictionary.
    #This returns its index in the chromosomeArray.
    for k, v in fitnessDic.iteritems():
        if v == highestFitness:
            highestFitnessIndex = k

    #create a csv with dic of highest fitness value chromosome. This was used more for when a 25%/75% was recorded to
    #analyse the final words and their sentimental scores.
    #opens csv in write mode, "wb" so that there is no empty line between each line
    with open('FinalValuesDic.csv', 'wb') as csvFile:
        writer = csv.writer(csvFile)
        count =0
        #iterates through the fittest chromosome's dictionary and prints out word appearance count and the key value
        # pairs with one per line separated by commas.
        for k,v in chromosomeArray[highestFitnessIndex].wordsAndValues.iteritems():
            appearances = wordTotals[k]
            writer.writerow([appearances,k,v])
            count +=1

    csvFile.close()
    print(str(count) + ' Lines Saved To Document.')

    # This section calculates the accuracy and F score of the test data using the key value pairs in the dictionary
    # stored in the chromosome with the highest fitness score
    bestChromosome = chromosomeArray[highestFitnessIndex]

    testRatings =[]
    testTweets =[]

    #splits the lines in the test data into sentimental score and the tweet string.
    for line in testData:
        testRatings.append(int(line[:1]))
        testTweets.append(line[2:])

    #cleans the tweets using the same preprocessing as the training data.
    cleanedTestTweets = preProcessing(testTweets)
    tweetObjectTestArray =[]

    #creates new Tweet objects which will hold the tweets polarity as an int and the word and sentimental score
    # dictonary. It stores each tweet in a list.
    x =0
    while x < len(cleanedTestTweets):
        tweet = Tweet(testRatings[x],cleanedTestTweets[x])
        tweetObjectTestArray.append(tweet)
        x+= 1

    #calculates accuracy and F score.
    testScore = assessFitness(bestChromosome, tweetObjectTestArray)
    fScore = calculateFScore(bestChromosome, tweetObjectTestArray)

    fScore *= 100
    print "KFold Iteration Number " + str(kFoldCounter+1)
    print "Test Score Accuracy is " + str(testScore)
    print "Test Score F-Score is " + str(fScore)
    #Adds the current Fold score to the total cumulative score which is divided by total folds below to get average.
    totalKFoldFScore += fScore
    totalKFoldAccuracy += testScore
    kFoldCounter +=1

#calculates average score
finalKFoldScore = totalKFoldFScore / kFoldValue
finalAccuracyKFoldScore = totalKFoldAccuracy / kFoldValue

print "Overall K Fold F- Score is " + str(finalKFoldScore)
print "Overall K-Fold Accuracy is " + str(finalAccuracyKFoldScore)