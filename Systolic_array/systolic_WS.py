import numpy as np

class SystolicArrayCell:
    def __init__(self):
        # for debug_position
        self.pos_x = 0
        self.pos_y = 0

        # input
        self.weight = 0
        self.in_partial_sum = 0
        self.in_activation = 0

        # output
        self.partial_sum_out = 0
        self.activation_out = 0

        # stored_value
        self.partial_sum = 0
        self.activation = 0

    def set_position(self,row,col):
        self.pos_x = row
        self.pos_y = col

    def set_weight(self, weight):
        self.weight = weight

    def read(self):
        self.partial_sum = self.in_partial_sum
        self.activation = self.in_activation

    def compute(self):
        result = self.weight * self.activation
        self.partial_sum_out = self.partial_sum + result
        self.activation_out = self.activation

class SystolicArray:
    def __init__(self, array_size):
        self.array_size = array_size

        self.input = [list() for _ in range(self.array_size)]
        self.output = [list() for _ in range(self.array_size)]

        self.cells = []

        self.cell_array = [[SystolicArrayCell() for _ in range(self.array_size)] for _ in range(self.array_size)]

        #for debug - positioning
        for row in range(self.array_size):
            for col in range(self.array_size):
                self.cell_array[row][col].set_position(row,col)

    def update(self):
        for pos_x in range(self.array_size):
            for pos_y in range(self.array_size):
                if pos_y == 0:
                    if self.input[pos_x]:
                        self.cell_array[pos_x][pos_y].in_activation = self.input[pos_x].pop(0)
                else:
                    self.cell_array[pos_x][pos_y].in_activation = self.cell_array[pos_x][pos_y-1].activation_out
                if pos_x == 0:
                    self.cell_array[pos_x][pos_y].in_partial_sum = 0
                else:
                    self.cell_array[pos_x][pos_y].in_partial_sum = self.cell_array[pos_x-1][pos_y].partial_sum_out

    def fill_weight(self, weights):
        for pos_x in range(self.array_size):
            for pos_y in range(self.array_size):
                self.cell_array[pos_x][pos_y].set_weight(weights[pos_x][pos_y])

    def fill_activations(self, activations):
        # padded with a triangle of zeroes
        for row_num in range(self.array_size):
            for _ in range(row_num):
                self.input[row_num].append(0)

        # Activations must be transposed
        for row_num in range(self.array_size):
            col = [activations[x][row_num] for x in range(self.array_size)]
            for activation in col:
                self.input[row_num].append(activation)

    def read(self):
        for row in self.cell_array:
            for cell in row:
                cell.read()

    def compute(self):
        for row in self.cell_array:
            for cell in row:
                cell.compute()
        # output
        for col in range(self.array_size):
            self.output[col].append(self.cell_array[-1][col].partial_sum_out)

    def cycle(self):
        self.read()
        self.compute()

    def display_status(self):
        for row in range(self.array_size):
            for col in range(self.array_size):
                print(self.cell_array[row][col].in_activation,end=" ")
            print(" ")
        print("--------")
        for row in range(self.array_size):
            for col in range(self.array_size):
                print(self.cell_array[row][col].partial_sum_out,end=" ")
            print(" ")
        print("==============")

    def run(self, input_size):
        for _ in range(input_size * self.array_size - (input_size - 1)):
            self.update()
            self.cycle()
            self.display_status()

        return self.get_result()

    def get_result(self):
        result = []
        ## skip zero 
        for row in range(self.array_size): 
            a = []
            for col in range(row+self.array_size-1, row+(self.array_size)*2-1):
                a.append(self.output[row][col])
            result.append(a)

        return np.transpose(result)

myArray = SystolicArray(4)

activations = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
]
myArray.fill_activations(activations)
print(myArray.input)

weights = [
    [10,20,30,40],
    [40,50,60,70],
    [70,80,90,100],
    [110,120,130,140]
]
myArray.fill_weight(weights)

result = myArray.run(4)
print(result)
assert (result == np.matmul(activations, weights)).all()
print('Systolic array matches numpy matmul')