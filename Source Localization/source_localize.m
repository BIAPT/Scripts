%Danielle Nadin 29-10-2019
%Source localization pipeline - automate source localization in
%Brainstorm from uploading .set file to exporting scout time series matrix
%for APCI analysis 

function [Value,Time,Atlas] = source_localize(pData,pCov,InverseMethod,template,bNorm)
    
    %% Variables
    % pData: INPUT, path to .set file
    % pCov: INPUT, path to the .set file used to compute the noise covariance matrix. If empty ([]), use the identity matrix. 
    % InverseMethod: INPUT, specify inverse model you will use (implemented options: MNE ('mne')
    % template: INPUT, specify brain atlas you will use to cluster dipoles into ROIs (implemented options: Desikan-Killiany ('dk'), AAL('aal')
    % bNorm: INPUT, Boolean defining  whether you want to normalize the solution (for visualization, '1') or not (for connectivity analysis,'0')
    % Value: OUTPUT, structure containing the source time series
    % Time: OUTPUT, structure containing the time points associated with source time series
    % Atlas: OUTPUT, labels associated with each ROI in the chosen template
    
    %Input variable definition for testing
%     pData = 'C:\Users\dn-xo\OneDrive - McGill University\Research\BIAPT Lab\Thesis\TDCS\Source localization\TDCSpilot_baseline.set';
%     pCov = [];
%     InverseMethod = 'mne';
%     template = 'dk';
%     bNorm = 0;
    
    %% Create protocol in Brainstorm
    ProtocolName = 'AutomateSourceLocalization';
    if ~brainstorm('status')
        brainstorm nogui 
    end
    
    gui_brainstorm('DeleteProtocol',ProtocolName);
    gui_brainstorm('CreateProtocol',ProtocolName,1,1); %Use default anatomy and one channel file per subject
    disp('Protocol created with default anatomy and channel locations.')
    
    sFiles = [];
    bst_report('Start',sFiles);
    sTemplate = struct('Name','Colin27_2016','FilePath','.\Colin27_2016.zip');
    db_set_template(0,sTemplate,0); %change default template to Colin27
    disp('Default anatomy changed to Colin27 template.')
    
    %% Process: Import data (pData, pCov)
    
    % ImportData
    db_add_subject('Subject',1,1,1); %subject id = 1
    sFiles = import_data(pData,[],'EEG-EEGLAB',[],1);
    disp('Imported EEG data.')
    
    %Import noise covariance data
    if ~isempty(pCov)
        db_add_subject('Noise Covariance',2,1,1); %subject id = 2
        sNoise = import_data(pCov,[],'EEG-EEGLAB',[],2);
    end
    
    %TODO: deal with pop-ups + don't import channel file when you import data
    
    %Project electrodes to the scalp surface
    %sFiles = bst_process('CallProcess', 'process_channel_project', sFiles, []);
    sFiles = bst_process('CallProcess', 'process_channel_addloc', sFiles, [], ...
    'usedefault',  30, ...  % Colin27: GSN HydroCel 128 E1
    'fixunits',    1, ...
    'vox2ras',     1);
    
    %% Process: Compute head model
  
    
    if strcmp('dk',template) %DK atlas: cortex head model
        sFiles = bst_process('CallProcess', 'process_headmodel', sFiles, [], ...
            'sourcespace', 1, ...  % Cortex surface
            'eeg',         3, ...  % OpenMEEG BEM
            'openmeeg',    struct(... %default
                'BemSelect',    [1, 1, 1], ...
                'BemCond',      [1, 0.0125, 1], ...
                'BemNames',     {{'Scalp', 'Skull', 'Brain'}}, ...
                'BemFiles',     {{}}, ...
                'isAdjoint',    0, ...
                'isAdaptative', 1, ...
                'isSplit',      0, ...
                'SplitLength',  4000));
        
    elseif strcmp('aal',template) % AAL atlas: volume head model
        sFiles = bst_process('CallProcess', 'process_headmodel', sFiles, [], ...
            'sourcespace', 2, ...  % MRI volume
            'volumegrid',  struct(... %default
                'Method',        'adaptive', ...
                'nLayers',       17, ...
                'Reduction',     3, ...
                'nVerticesInit', 4000, ...
                'Resolution',    0.005, ...
                'FileName',      []), ...
            'eeg',         3, ...  % OpenMEEG BEM
            'openmeeg',    struct(... %default
                'BemFiles',     {{}}, ...
                'BemNames',     {{'Scalp', 'Skull', 'Brain'}}, ...
                'BemCond',      [1, 0.0125, 1], ...
                'BemSelect',    [1, 1, 1], ...
                'isAdjoint',    0, ...
                'isAdaptative', 1, ...
                'isSplit',      0, ...
                'SplitLength',  4000));
    end
         
    %% Process: Compute noise covariance
    if ~isempty(pCov)
        %Compute noise covariance matrix
        sNoise = bst_process('CallProcess', 'process_noisecov', sNoise, [], ...
            'target',         1, ...  % Noise covariance     (covariance over baseline time window)
            'dcoffset',       1, ...  % Block by block, to avoid effects of slow shifts in data
            'identity',       0, ...
            'copycond',       0, ...
            'copysubj',       0, ...
            'copymatch',      0, ...
            'replacefile',    1);  % Replace
        
        %TODO: Copy matrix to subject file
    else
        %Use identity matrix, no noise modelling
        sFiles = bst_process('CallProcess', 'process_noisecov', sFiles, [], ...
            'target',         1, ...  % Noise covariance     (covariance over baseline time window)
            'dcoffset',       1, ...  % Block by block, to avoid effects of slow shifts in data
            'identity',       1, ...
            'copycond',       0, ...
            'copysubj',       0, ...
            'copymatch',      0, ...
            'replacefile',    1);  % Replace
    end
    %% Process: Compute inverse solution
    
    %TODO: unconstrained dipoles for AAL atlas
    
    if strcmp('mne',InverseMethod)
        if strcmp(1,bNorm) %normalized for visualization (dSPM)
            sFiles = bst_process('CallProcess', 'process_inverse_2018', sFiles, [], ...
                'output',  2, ...  % Kernel only: one per file
                'inverse', struct(...
                'Comment',        'dSPM-unscaled: EEG', ...
                'InverseMethod',  'minnorm', ...
                'InverseMeasure', 'dspm2018', ...
                'SourceOrient',   {{'fixed'}}, ...
                'NoiseMethod',    'diag', ...
                'DataTypes',      {{'EEG'}}));
        else %do not normalize (current density map)
            sFiles = bst_process('CallProcess', 'process_inverse_2018', sFiles, [], ...
                'output',  2, ...  % Kernel only: one per file
                'inverse', struct(...
                'Comment',        'MN: EEG', ...
                'InverseMethod',  'minnorm', ...
                'InverseMeasure', 'amplitude', ...
                'SourceOrient',   {{'fixed'}}, ...
                'NoiseMethod',    'diag', ...
                'DataTypes',      {{'EEG'}}));
        end
    elseif strcmp('beamformer',InverseMethod)
        sFiles = bst_process('CallProcess', 'process_inverse_2018', sFiles, [], ...
            'output',  2, ...  % Kernel only: one per file
            'inverse', struct(...
            'Comment',        'PNAI: EEG', ...
            'InverseMethod',  'lcmv', ...
            'InverseMeasure', 'nai', ...
            'SourceOrient',   {{'fixed'}}, ...
            'NoiseMethod',    'diag', ...
            'DataTypes',      {{'EEG'}}));
    elseif strcmp('wmem',InverseMethod)
        disp('TODO: implement wMEM')
    end
    %%  Process: Compute time series in ROIs
  
    if strcmp('dk',template)
        sFiles = bst_process('CallProcess', 'process_extract_scout', sFiles, [], ...
            'scouts',         {'Desikan-Killiany', {'bankssts L', 'bankssts R', 'caudalanteriorcingulate L', 'caudalanteriorcingulate R', 'caudalmiddlefrontal L', 'caudalmiddlefrontal R', 'cuneus L', 'cuneus R', 'entorhinal L', 'entorhinal R', 'frontalpole L', 'frontalpole R', 'fusiform L', 'fusiform R', 'inferiorparietal L', 'inferiorparietal R', 'inferiortemporal L', 'inferiortemporal R', 'insula L', 'insula R', 'isthmuscingulate L', 'isthmuscingulate R', 'lateraloccipital L', 'lateraloccipital R', 'lateralorbitofrontal L', 'lateralorbitofrontal R', 'lingual L', 'lingual R', 'medialorbitofrontal L', 'medialorbitofrontal R', 'middletemporal L', 'middletemporal R', 'paracentral L', 'paracentral R', 'parahippocampal L', 'parahippocampal R', 'parsopercularis L', 'parsopercularis R', 'parsorbitalis L', 'parsorbitalis R', 'parstriangularis L', 'parstriangularis R', 'pericalcarine L', 'pericalcarine R', 'postcentral L', 'postcentral R', 'posteriorcingulate L', 'posteriorcingulate R', 'precentral L', 'precentral R', 'precuneus L', 'precuneus R', 'rostralanteriorcingulate L', 'rostralanteriorcingulate R', 'rostralmiddlefrontal L', 'rostralmiddlefrontal R', 'superiorfrontal L', 'superiorfrontal R', 'superiorparietal L', 'superiorparietal R', 'superiortemporal L', 'superiortemporal R', 'supramarginal L', 'supramarginal R', 'temporalpole L', 'temporalpole R', 'transversetemporal L', 'transversetemporal R'}}, ...
            'scoutfunc',      1, ...  % Mean
            'isflip',         1, ...
            'isnorm',         1, ...
            'concatenate',    1, ...
            'save',           1, ...
            'addrowcomment',  1, ...
            'addfilecomment', 1);
    elseif strcmp('aal',template)
        disp('TODO: implement AAL atlas')
    end
  
    %% Process: Load output file to populate output variables
    p =  bst_get('BrainstormDbDir');
    load([p '\' ProtocolName '\data\' sFiles.FileName])
end