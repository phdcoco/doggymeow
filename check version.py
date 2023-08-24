import random

class shufflenum:
    def __init__(self):
        self.number = [i for i in range(1, 46)] 
        
    def pick(self):
        print('1. before shuffling - >> ', self.number)
        random.shuffle(self.number)
        print('2. after shuffling - >>', self.number)
        return sorted(random.choice(self.number) for _ in range(7))
    
    def __call__(self):
        return self.pick()

randnum = shufflenum()
print(randnum())

list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
print(list1[:-10])