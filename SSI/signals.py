# Pre processing function that will apply the pre-processing technique
# based on what analysis technique we are working with.
def pre_process(timestamps, data_pts, data_type, sample_rate):
    print("Preprocessing: " + data_type)

    # Pre-processing (if you want to tweak them change them here!)
    if data_type == "BVP":
        data_pts = data_pts
    elif data_type == "EDA":
        data_pts = data_pts
    elif data_type == "TEMP":
        data_pts = data_pts
    elif data_type == "HR":
        data_pts = data_pts
        
    return data_pts