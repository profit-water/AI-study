import numpy as np

class SystolicCell:
    def __init__(self):
        #input
        self.in_weight = 0
        self.in_activation = 0
        
        #output
        self.out_weight = 0
        self.out_activation = 0
        
        #stationary
        self.weight = 0
        self.activation = 0
        self.partial_sum = 0
    
    def compute(self):
        self.partial_sum = self.partial_sum +\
                           self.weight*self.activation
        self.out_weight = self.weight
        self.out_activation = self.activation

    def read(self):
        self.weight = self.in_weight
        self.activation = self.in_activation

class SystolicArray:
    def __init__(self,array_size):
        self.array_size = array_size
        
        #Create array
        self.array = [[SystolicCell() for _ in range(self.array_size)]\
                      for _ in range(self.array_size)]
        
        #weight/act buffer
        self.buf_weight = [list() for _ in range(self.array_size)]
        self.buf_activation = [list() for _ in range(self.array_size)]

    def fill_weights(self,weight):
        for row in range(self.array_size):
            for _ in range(row):
                self.buf_weight[row].append(0)
        
        for row in range(self.array_size):
            col = [weight[x][row] for x in range(self.array_size)]
            self.buf_weight[row].extend(col)

    def fill_activations(self,activation):
        for row in range(self.array_size):
            for _ in range(row):
                self.buf_activation[row].append(0)

        for row in range(self.array_size):
            self.buf_activation[row].extend(activation[row])

    def update(self):
        for pos_x in range(self.array_size):
            for pos_y in range(self.array_size):
                #if pos_x is 0, fetch weight from buffer
                if pos_x == 0:
                    if self.buf_weight[pos_y]:
                        self.array[pos_x][pos_y].in_weight =\
                            self.buf_weight[pos_y].pop(0)
                    else:
                        self.array[pos_x][pos_y].in_weight = 0
                else:
                    self.array[pos_x][pos_y].in_weight = \
                    self.array[pos_x-1][pos_y].out_weight
                
                #if pos_y is 0, fetch activations from buffer
                if pos_y == 0:
                    if self.buf_activation[pos_x]:
                        self.array[pos_x][pos_y].in_activation =\
                            self.buf_activation[pos_x].pop(0)
                    else:
                        self.array[pos_x][pos_y].in_activation = 0
                else:
                    self.array[pos_x][pos_y].in_activation = \
                    self.array[pos_x][pos_y-1].out_activation
 
    def display_status(self):
        #activation
        print("activation")
        print(self.buf_activation)
        for row in range(self.array_size):
            for col in range(self.array_size):
                print(self.array[row][col].in_activation,end=" ")
            print(" ")
        print("--------")
        #activation
        print("weight")
        for row in range(self.array_size):
            for col in range(self.array_size):
                print(self.array[row][col].in_weight,end=" ")
            print(" ")
        print("--------")
        #partial_sum
        print("partial_sum")
        for row in range(self.array_size):
            for col in range(self.array_size):
                print(self.array[row][col].partial_sum,end=" ")
            print(" ")
        print("==============")

    def cycle(self):
        #read(update internal RF) and compute
        for row in self.array:
            for cell in row:
                cell.read()
                cell.compute()
    
    def run(self,input_size):
        for _ in range(input_size * self.array_size - (input_size - 1)):
            self.update()
            self.cycle()
            self.display_status()
        
        result = [[self.array[y][x].partial_sum for x in range(self.array_size)]\
                   for y in range(self.array_size)]
        return result

#Example code
myArray = SystolicArray(4)

activations = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
]
myArray.fill_activations(activations)

weights = [
    [10,20,30,40],
    [40,50,60,70],
    [70,80,90,100],
    [110,120,130,140]
]
myArray.fill_weights(weights)

result = myArray.run(4)
print(result)
assert (result == np.matmul(activations, weights)).all()
print('Systolic array matches numpy matmul')