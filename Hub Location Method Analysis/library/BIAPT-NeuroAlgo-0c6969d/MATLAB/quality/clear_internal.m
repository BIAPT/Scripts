function clear_internal()
    %%Function used to clear inscrutable internal matlab runtime environment
    %%variables that are causing abnormal results.

    %% This clear all does not clear the base workspace variable space but only affects the function's variable space. The reason why this work is absolutely unknown to me (Pascal) at this moment.
    global data_in
    global s_seg
    global Ftr;
    global Ftr_norm;
    global class;

    keep global trainedClassifier data_in s_seg Ftr Ftr_norm class;
    % clear all

    data_in{1,1}.t = zeros(30000,1);
    data_in{1,1}.bvp = zeros(30000,1);
    data_in{1,1}.sc = zeros(30000,1);
    data_in{1,1}.skt = zeros(30000,1);

    for n=1:80
        s_seg{n}.t = zeros(30000,1);
        s_seg{n}.bvp = zeros(30000,1);
        s_seg{n}.sc = zeros(30000,1);
        s_seg{n}.skt = zeros(30000,1);
        s_seg{n}.pks = zeros(30000,1);
    end

    Ftr = zeros(80,64);
    Ftr_norm = zeros(80,64);

    class = zeros(80,1);    

end