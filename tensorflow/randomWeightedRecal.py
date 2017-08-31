#!/usr/bin/env python
import random
import unittest
class RandomWeightedRecal():
    def __init__(self,historyLength):
        self.historyLength = historyLength
        self.weightedItemList= []
        self.weightSum = 0

    def addItem(self,weight,item):

        self.weightedItemList.append((weight,item))
        self.weightSum += weight

        while len(self.weightedItemList) > self.historyLength:
            oldWeight,oldItem = self.weightedItemList[0]
            self.weightSum -= oldWeight
            del(self.weightedItemList[0])


    def getWeightedRandomItem(self):
        randNum = random.uniform(0,self.weightSum)

        runningWeightSum = 0
        for weight,item in self.weightedItemList:
            runningWeightSum += weight
            if randNum < runningWeightSum:
                return item

        return random.choice(self.weightedItemList)[1]

    def getWeightedRandomList(self,listLength):
        randomList = []
        for i in range(listLength):
            item = self.getWeightedRandomItem()
            randomList.append(item)
        return randomList




class Test(unittest.TestCase):
    def test_historyLength(self):
        rwr = RandomWeightedRecal(10)
        self.assertEqual(len(rwr.weightedItemList),0)
        rwr.addItem(1,"test")
        self.assertEqual(len(rwr.weightedItemList),1)
        for i in range(20):
            rwr.addItem(1,"test")
        self.assertEqual(len(rwr.weightedItemList),10)


    def test_distrobution(self):
        n = 1000000
        s = 2
        rwr = RandomWeightedRecal(10)
        bins = 4
        total = 0.0
        count = []
        for i in range(bins):
            rwr.addItem(i,i)
            count.append(0.0)
            total += i

        for _ in range(n):
            iList = rwr.getWeightedRandomList(s)
            for i in iList:
                count[i] += 1.0/float(n)/s

        for i in range(bins):
            diff = abs(count[i] - i/total)
            print("%f %f" % (count[i],diff))
            self.assertLess(diff,0.001)

if __name__ == "__main__":
    unittest.main(verbosity=2)
