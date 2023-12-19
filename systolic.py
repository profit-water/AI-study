from queue import Queue
import numpy as np

class SystolicArrayCell:
    def __init__(self):
        #for debug_position
        self.pos_x = 0
        self.pos_y = 0

        #input
        self.weight = 0
        self.in_partial_sum = 0
        self.in_activation = 0

        #output
        self.partial_sum_out = 0
        self.activation_out = 0

        #stored_value
        self.partial_sum = None
        self.activation = None

    def set_weight(self,weight):
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

        self.cell_array = [[SystolicArrayCell() for _ in range(array_size)] for _ in range(array_size)]

    def connect(self):
        for pos_x in range(self.array_size):
            for pos_y in range(self.array_size):
                if pos_x is 0:
                    self.cell_array[pos_x][pos_y].in_activation = self.input[pos_y]
                else:
                    self.cell_array[pos_x][pos_y].in_activation = self.cell_array[pos_x - 1][pos_y].activation_out
                if pos_y is 0:
                    self.cell_array[pos_x][pos_y].in_partial_sum = 0
                else:
                    self.cell_array[pos_x][pos_y].in_partial_sum = self.cell_array[pos_x][pos_y - 1].partial_sum_out

    def fill_weight(self, weights):
        for pos_x in range(self.array_size):
            for pos_y in range(self.array_size):
                self.cell_array[pos_x][pos_y].set_weight(weights[pos_x][pos_y])

    def fill_activations(self,activations):
        #padded with a triangle of zeroes
        for row_num in range(self.array_size):
            for _ in range(row_num):
                self.input[row_num].append(0)

        # And the activations must be transposed
        for row_num in range(self.array_size):
            col = [activations[x][row_num] for x in range(self.array_size)]
            for activation in col:
                self.input[row_num].append(activation)










