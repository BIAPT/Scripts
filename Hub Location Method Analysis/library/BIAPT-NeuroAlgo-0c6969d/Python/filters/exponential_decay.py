def exponential_decay(input_data, alpha):
    '''
    Exponential Decay Smoothing Filter
    Increase alpha for more filtering: uses more past data to compute an average
    - input_data is a vector (list of values)
    '''

    output_data = input_data.copy() 
    for i in range(0,input_data.num-1):
        output_data[i+1] = (output_data[i,0]*alpha) + (output_data[i+1,0] * (1-alpha))
        
    return output_data