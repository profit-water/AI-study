import numpy as np

#Step1. floating point's exponent, mantissa parser with zero-padding
def float_bin_parser(len_exp:int, val:np.uint16)->list:
    len_man = 15-len_exp
    
    ref_mask = np.uint16(0b1111_1111_1111_1111)
    exp_mask = (ref_mask << len_man) & np.uint16(0b0111_1111_1111_1111)
    man_mask = ref_mask >> (len_exp+1)
    
    sign_bit = (val >> 15) & 0x1
    exponent = (val & exp_mask) >> len_man
    mantissa = (val & man_mask)
    
    return sign_bit, np.uint16(exponent), np.uint16(mantissa)

def fp_adder(len_exp:int, padding_width:int, num_operand:int, operands:list )->list:
    ##Step1. parsing
    sign,exp,man = [],[],[]
    for _ in range(num_operand):
        sign[_],exp[_],man[_] = float_bin_parser(len_exp, operands[_])
        
    ##Step2. Zero-padding(mantissa)
    man_padded = []
    for _ in range(num_operand):
        temp_arr = np.zeros((1,),dtype=np.uint32)
        temp_arr[0] = man[_]
        man_padded[_] = temp_arr[0] << padding_width
    
    ##Step3. aligned with the biggest exponent
    
        
        
        
    
    
        
    