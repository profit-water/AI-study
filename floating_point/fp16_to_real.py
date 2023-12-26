def fp16_to_float(len_exp:int, bf_val:np.uint16) -> float:
    
    # Figure out mantisa length
    len_man = 15-len_exp
    
    # make exp_mask, man_mask
    ref_mask = np.uint16(0b1111_1111_1111_1111)
    
    exp_mask = (ref_mask << len_man) & np.uint16(0b0111_1111_1111_1111)
    man_mask = ref_mask >> (len_exp+1)
    
    ###print(f'exp_mask:{bin(exp_mask)}, man_mask:{bin(man_mask)}')
    
    sign_bit = (bf_val >> 15) & 0x1
    exponent = (bf_val & exp_mask) >> len_man
    mantisa  = (bf_val & man_mask)
    
    ###print(f'exp:{bin(exponent)}, man:{bin(mantisa)}')
    
    # Special case -> exp is max
    # infinity, NaN
    if exponent == (exp_mask >> len_man):
        if mantisa == 0:
            return np.where(sign_bit, -np.inf, np.inf)
        else:
            return np.nan
        
    exp_bias = 2**(len_exp-1)-1
    real_exponent = exponent - exp_bias
    real_mantisa = mantisa / 2**len_man

    ###print(f'sign:{sign_bit}, real_exp:{real_exponent}, man:{real_mantisa}')
          
    # Special case -> exp is zero
    # 0, de-normalized
    if real_exponent == 0x0:
        if real_mantisa == 0x0:
            return np.where(sign_bit, -np.float16(0), np.float16(0))
        else:
            real_value = (-1) ** sign_bit * real_mantisa * ((2 ** real_exponent) if real_exponent>=0 else (1/(2 ** abs(real_exponent))) ) 
            return np.float16(real_value)
    
    real_value = (-1) ** sign_bit * (1+real_mantisa) * ((2 ** real_exponent) if real_exponent>=0 else 1/(2 ** abs(real_exponent)) ) 
    return np.float16(real_value)