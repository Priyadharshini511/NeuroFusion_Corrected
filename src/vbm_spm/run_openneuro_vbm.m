% =========================================================================
% NeuroFusion AI - OpenNeuro External VBM Processing
% =========================================================================
%
% INPUT:
%   MNI-registered skull-stripped T1 images
%
% PIPELINE:
%
%   MNI registered T1
%          |
%          v
%   SPM New Segment
%          |
%          +--> c1 = native GM
%          |
%          +--> wc1 = warped GM
%          |
%          +--> mwc1 = modulated warped GM
%          |
%          v
%   8 mm smoothing
%
% PURPOSE:
%   Generate VBM GM maps for the 43-subject OpenNeuro external cohort.
%
% IMPORTANT:
%   - Does NOT modify the canonical 119-subject RMS dataset.
%   - Does NOT modify Exp20.
%   - Does NOT retrain the Linear SVM.
%   - External branch only.
%
% =========================================================================

clear;
clc;

fprintf('\n');
fprintf('====================================================================\n');
fprintf('NeuroFusion AI - OpenNeuro External VBM\n');
fprintf('====================================================================\n\n');


%% ------------------------------------------------------------------------
% PROJECT PATHS
% -------------------------------------------------------------------------

PROJECT_ROOT = 'C:\Users\User\Documents\neurofusion';

REGISTERED_ROOT = fullfile( ...
    PROJECT_ROOT, ...
    'processed_data', ...
    'smri', ...
    'openneuro_ds000115', ...
    'registered');

VBM_ROOT = fullfile( ...
    PROJECT_ROOT, ...
    'processed_data', ...
    'smri', ...
    'openneuro_ds000115', ...
    'vbm');


%% ------------------------------------------------------------------------
% SPM INITIALIZATION
% -------------------------------------------------------------------------

fprintf('Initializing SPM...\n');

spm('Defaults','fmri');
spm_jobman('initcfg');

SPM_ROOT = spm('Dir');

fprintf('SPM directory:\n%s\n\n', SPM_ROOT);


%% ------------------------------------------------------------------------
% CHECK TPM
% -------------------------------------------------------------------------

TPM_FILE = fullfile( ...
    SPM_ROOT, ...
    'tpm', ...
    'TPM.nii');

if ~isfile(TPM_FILE)
    error('SPM TPM file not found:\n%s', TPM_FILE);
end

fprintf('TPM:\n%s\n\n', TPM_FILE);


%% ------------------------------------------------------------------------
% CHECK INPUT DIRECTORY
% -------------------------------------------------------------------------

if ~isfolder(REGISTERED_ROOT)
    error( ...
        'Registered image directory does not exist:\n%s', ...
        REGISTERED_ROOT);
end


if ~isfolder(VBM_ROOT)
    mkdir(VBM_ROOT);
end


%% ------------------------------------------------------------------------
% FIND REGISTERED SUBJECT IMAGES
% -------------------------------------------------------------------------

files = dir(fullfile( ...
    REGISTERED_ROOT, ...
    'sub-*_T1w_mni.nii.gz'));


if isempty(files)

    error([ ...
        'No registered images found.' newline ...
        'Expected files like:' newline ...
        'sub-01_T1w_mni.nii.gz' newline ...
        'Directory:' newline ...
        REGISTERED_ROOT]);

end


fprintf('Registered images found: %d\n\n', numel(files));


%% ------------------------------------------------------------------------
% SUBJECT LOOP
% -------------------------------------------------------------------------

success_count = 0;
failed_count  = 0;

results = cell(numel(files),4);


for i = 1:numel(files)

    filename = files(i).name;

    % -------------------------------------------------------------
    % Extract subject ID
    % -------------------------------------------------------------

    tokens = regexp( ...
        filename, ...
        '^(sub-[0-9]+)_T1w_mni\.nii\.gz$', ...
        'tokens');

    if isempty(tokens)

        fprintf('\nSkipping unexpected filename:\n%s\n', filename);

        results{i,1} = filename;
        results{i,2} = 'UNKNOWN';
        results{i,3} = 'SKIPPED';
        results{i,4} = 'Invalid filename';

        failed_count = failed_count + 1;

        continue;

    end


    subject = tokens{1}{1};


    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('[%d/%d] %s\n', i, numel(files), subject);
    fprintf('====================================================================\n');


    %% ----------------------------------------------------------------
    % SUBJECT OUTPUT DIRECTORY
    % -----------------------------------------------------------------

    subject_dir = fullfile( ...
        VBM_ROOT, ...
        subject);


    if ~isfolder(subject_dir)
        mkdir(subject_dir);
    end


    %% ----------------------------------------------------------------
    % SOURCE FILE
    % -----------------------------------------------------------------

    source_gz = fullfile( ...
        REGISTERED_ROOT, ...
        filename);


    if ~isfile(source_gz)

        fprintf('ERROR: Input file missing.\n');

        results{i,1} = subject;
        results{i,2} = source_gz;
        results{i,3} = 'FAILED';
        results{i,4} = 'Input file missing';

        failed_count = failed_count + 1;

        continue;

    end


    fprintf('Input:\n%s\n', source_gz);


    %% ----------------------------------------------------------------
    % UNCOMPRESS INTO SUBJECT DIRECTORY
    % -----------------------------------------------------------------

    input_nii = fullfile( ...
        subject_dir, ...
        [subject '_T1w_mni.nii']);


    try

        if ~isfile(input_nii)

            fprintf('Uncompressing NIfTI...\n');

            gunzip(source_gz, subject_dir);

        else

            fprintf('NIfTI already exists. Reusing it.\n');

        end


    catch ME

        fprintf('ERROR during gunzip:\n%s\n', ME.message);

        results{i,1} = subject;
        results{i,2} = source_gz;
        results{i,3} = 'FAILED';
        results{i,4} = ME.message;

        failed_count = failed_count + 1;

        continue;

    end


    %% ----------------------------------------------------------------
    % VERIFY NIFTI
    % -----------------------------------------------------------------

    if ~isfile(input_nii)

        fprintf('ERROR: Uncompressed NIfTI not found:\n%s\n', input_nii);

        results{i,1} = subject;
        results{i,2} = input_nii;
        results{i,3} = 'FAILED';
        results{i,4} = 'Uncompressed NIfTI missing';

        failed_count = failed_count + 1;

        continue;

    end


    %% ----------------------------------------------------------------
    % BASIC IMAGE CHECK
    % -----------------------------------------------------------------

    try

        V = spm_vol(input_nii);

        if isempty(V)

            error('SPM could not read the input image.');

        end


        fprintf('Input dimensions: [%s]\n', ...
            num2str(V.dim));


        fprintf('Voxel size: [%s]\n', ...
            num2str(sqrt(sum(V.mat(1:3,1:3).^2))));


    catch ME

        fprintf('ERROR reading image:\n%s\n', ME.message);

        results{i,1} = subject;
        results{i,2} = input_nii;
        results{i,3} = 'FAILED';
        results{i,4} = ME.message;

        failed_count = failed_count + 1;

        continue;

    end


    %% ----------------------------------------------------------------
    % SPM NEW SEGMENT
    % -----------------------------------------------------------------

    fprintf('\n');
    fprintf('Starting SPM New Segment...\n');


    clear matlabbatch;


    % -------------------------------------------------------------
    % CHANNEL
    % -------------------------------------------------------------

    matlabbatch{1}.spm.spatial.preproc.channel.vols = ...
        {[input_nii ',1']};


    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = ...
        0.001;


    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = ...
        60;


    matlabbatch{1}.spm.spatial.preproc.channel.write = ...
        [0 1];


    % -------------------------------------------------------------
    % TISSUE CLASSES
    % -------------------------------------------------------------

    ngauss = [1 1 2 3 4 2];


    for k = 1:6

        matlabbatch{1}.spm.spatial.preproc.tissue(k).tpm = ...
            {[TPM_FILE ',' num2str(k)]};


        matlabbatch{1}.spm.spatial.preproc.tissue(k).ngaus = ...
            ngauss(k);


        % Native GM / WM / CSF
        if k <= 3

            matlabbatch{1}.spm.spatial.preproc.tissue(k).native = ...
                [1 0];

        else

            matlabbatch{1}.spm.spatial.preproc.tissue(k).native = ...
                [0 0];

        end


        % Warped GM / WM / CSF
        if k <= 3

            matlabbatch{1}.spm.spatial.preproc.tissue(k).warped = ...
                [1 1];

        else

            matlabbatch{1}.spm.spatial.preproc.tissue(k).warped = ...
                [0 0];

        end

    end


    %% ----------------------------------------------------------------
    % WARP SETTINGS
    % -----------------------------------------------------------------

    matlabbatch{1}.spm.spatial.preproc.warp.mrf = ...
        1;


    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = ...
        1;


    matlabbatch{1}.spm.spatial.preproc.warp.reg = ...
        [0 0.001 0.5 0.05 0.2];


    matlabbatch{1}.spm.spatial.preproc.warp.affreg = ...
        'mni';


    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = ...
        0;


    matlabbatch{1}.spm.spatial.preproc.warp.samp = ...
        3;


    % -------------------------------------------------------------
    % WRITE DEFORMATION FIELDS
    % -------------------------------------------------------------

    matlabbatch{1}.spm.spatial.preproc.warp.write = ...
        [1 1];


    %% ----------------------------------------------------------------
    % RUN SEGMENTATION
    % -----------------------------------------------------------------

    try

        spm_jobman('run', matlabbatch);

        fprintf('\nSPM segmentation completed.\n');

    catch ME

        fprintf('\nSPM segmentation FAILED.\n');
        fprintf('%s\n', ME.message);

        results{i,1} = subject;
        results{i,2} = input_nii;
        results{i,3} = 'FAILED';
        results{i,4} = ME.message;

        failed_count = failed_count + 1;

        continue;

    end


    %% ----------------------------------------------------------------
    % EXPECTED OUTPUT FILES
    % -----------------------------------------------------------------

    native_gm = fullfile( ...
        subject_dir, ...
        ['c1' subject '_T1w_mni.nii']);


    warped_gm = fullfile( ...
        subject_dir, ...
        ['wc1' subject '_T1w_mni.nii']);


    modulated_gm = fullfile( ...
        subject_dir, ...
        ['mwc1' subject '_T1w_mni.nii']);


    deformation = fullfile( ...
        subject_dir, ...
        ['y_' subject '_T1w_mni.nii']);


    %% ----------------------------------------------------------------
    % VERIFY GM OUTPUTS
    % -----------------------------------------------------------------

    missing = false;


    if ~isfile(native_gm)

        fprintf('MISSING native GM:\n%s\n', native_gm);

        missing = true;

    end


    if ~isfile(warped_gm)

        fprintf('MISSING warped GM:\n%s\n', warped_gm);

        missing = true;

    end


    if ~isfile(modulated_gm)

        fprintf('MISSING modulated GM:\n%s\n', modulated_gm);

        missing = true;

    end


    if ~isfile(deformation)

        fprintf('MISSING deformation field:\n%s\n', deformation);

        missing = true;

    end


    if missing

        results{i,1} = subject;
        results{i,2} = input_nii;
        results{i,3} = 'FAILED';
        results{i,4} = 'One or more SPM outputs missing';

        failed_count = failed_count + 1;

        continue;

    end


    fprintf('\n');
    fprintf('SPM VBM outputs verified.\n');


    %% ----------------------------------------------------------------
    % SMOOTHING
    % -----------------------------------------------------------------

    smooth_dir = fullfile( ...
        subject_dir, ...
        'smooth');


    if ~isfolder(smooth_dir)

        mkdir(smooth_dir);

    end


    smooth_input = fullfile( ...
        smooth_dir, ...
        ['mwc1' subject '_T1w_mni.nii']);


    smooth_output = fullfile( ...
        smooth_dir, ...
        ['s8mwc1' subject '_T1w_mni.nii']);


    if ~isfile(smooth_input)

        copyfile( ...
            modulated_gm, ...
            smooth_input);

    end


    fprintf('\n');
    fprintf('Starting 8-mm smoothing...\n');


    clear matlabbatch;


    matlabbatch{1}.spm.spatial.smooth.data = ...
        {smooth_input};


    matlabbatch{1}.spm.spatial.smooth.fwhm = ...
        [8 8 8];


    matlabbatch{1}.spm.spatial.smooth.dtype = ...
        0;


    matlabbatch{1}.spm.spatial.smooth.im = ...
        0;


    matlabbatch{1}.spm.spatial.smooth.prefix = ...
        's8';


    try

        spm_jobman('run', matlabbatch);

        fprintf('8-mm smoothing completed.\n');

    catch ME

        fprintf('Smoothing FAILED:\n%s\n', ME.message);

        results{i,1} = subject;
        results{i,2} = modulated_gm;
        results{i,3} = 'FAILED';
        results{i,4} = ME.message;

        failed_count = failed_count + 1;

        continue;

    end


    %% ----------------------------------------------------------------
    % VERIFY SMOOTHED OUTPUT
    % -----------------------------------------------------------------

    if ~isfile(smooth_output)

        fprintf('ERROR: Smoothed GM file was not created.\n');
        fprintf('%s\n', smooth_output);

        results{i,1} = subject;
        results{i,2} = modulated_gm;
        results{i,3} = 'FAILED';
        results{i,4} = 'Smoothed GM missing';

        failed_count = failed_count + 1;

        continue;

    end


    %% ----------------------------------------------------------------
    % NUMERICAL QC
    % -----------------------------------------------------------------

    fprintf('\n');
    fprintf('Running numerical QC...\n');


    try

        Vgm = spm_vol(modulated_gm);
        Ygm = spm_read_vols(Vgm);


        Vs = spm_vol(smooth_output);
        Ys = spm_read_vols(Vs);


        finite_gm = isfinite(Ygm(:));
        finite_sm = isfinite(Ys(:));


        gm_min = min(Ygm(finite_gm));
        gm_max = max(Ygm(finite_gm));
        gm_mean = mean(Ygm(finite_gm));


        sm_min = min(Ys(finite_sm));
        sm_max = max(Ys(finite_sm));
        sm_mean = mean(Ys(finite_sm));


        fprintf('Modulated GM dimensions: [%s]\n', ...
            num2str(Vgm.dim));


        fprintf('Modulated GM finite: %d / %d\n', ...
            sum(finite_gm), numel(Ygm));


        fprintf('Modulated GM range: %.6g to %.6g\n', ...
            gm_min, gm_max);


        fprintf('Modulated GM mean: %.6g\n', ...
            gm_mean);


        fprintf('Smoothed GM dimensions: [%s]\n', ...
            num2str(Vs.dim));


        fprintf('Smoothed GM finite: %d / %d\n', ...
            sum(finite_sm), numel(Ys));


        fprintf('Smoothed GM range: %.6g to %.6g\n', ...
            sm_min, sm_max);


        fprintf('Smoothed GM mean: %.6g\n', ...
            sm_mean);


        % ---------------------------------------------------------
        % Check NaN / Inf
        % ---------------------------------------------------------

        if any(~finite_gm)

            error('Modulated GM contains NaN or Inf values.');

        end


        if any(~finite_sm)

            error('Smoothed GM contains NaN or Inf values.');

        end


    catch ME

        fprintf('\nQC FAILED:\n%s\n', ME.message);

        results{i,1} = subject;
        results{i,2} = modulated_gm;
        results{i,3} = 'FAILED';
        results{i,4} = ME.message;

        failed_count = failed_count + 1;

        continue;

    end


    %% ----------------------------------------------------------------
    % SUBJECT QC REPORT
    % -----------------------------------------------------------------

    qc_file = fullfile( ...
        subject_dir, ...
        [subject '_VBM_QC.txt']);


    fid = fopen(qc_file,'w');


    if fid ~= -1

        fprintf(fid, ...
            'NeuroFusion AI - OpenNeuro VBM QC\n\n');


        fprintf(fid, ...
            'Subject: %s\n\n', subject);


        fprintf(fid, ...
            'Input:\n%s\n\n', input_nii);


        fprintf(fid, ...
            'Native GM:\n%s\n\n', native_gm);


        fprintf(fid, ...
            'Warped GM:\n%s\n\n', warped_gm);


        fprintf(fid, ...
            'Modulated GM:\n%s\n\n', modulated_gm);


        fprintf(fid, ...
            'Deformation:\n%s\n\n', deformation);


        fprintf(fid, ...
            'Smoothed modulated GM:\n%s\n\n', smooth_output);


        fprintf(fid, ...
            'Modulated dimensions: [%s]\n', ...
            num2str(Vgm.dim));


        fprintf(fid, ...
            'Modulated finite voxels: %d / %d\n', ...
            sum(finite_gm), numel(Ygm));


        fprintf(fid, ...
            'Modulated minimum: %.8g\n', gm_min);


        fprintf(fid, ...
            'Modulated maximum: %.8g\n', gm_max);


        fprintf(fid, ...
            'Modulated mean: %.8g\n', gm_mean);


        fprintf(fid, ...
            '\nSmoothed dimensions: [%s]\n', ...
            num2str(Vs.dim));


        fprintf(fid, ...
            'Smoothed finite voxels: %d / %d\n', ...
            sum(finite_sm), numel(Ys));


        fprintf(fid, ...
            'Smoothed minimum: %.8g\n', sm_min);


        fprintf(fid, ...
            'Smoothed maximum: %.8g\n', sm_max);


        fprintf(fid, ...
            'Smoothed mean: %.8g\n', sm_mean);


        fclose(fid);

    end


    %% ----------------------------------------------------------------
    % SUCCESS
    % -----------------------------------------------------------------

    results{i,1} = subject;
    results{i,2} = input_nii;
    results{i,3} = 'SUCCESS';
    results{i,4} = smooth_output;


    success_count = success_count + 1;


    fprintf('\n');
    fprintf('STATUS: SUCCESS\n');


end


%% =========================================================================
% FINAL REPORT
% =========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('OPENNEURO VBM PROCESSING COMPLETE\n');
fprintf('====================================================================\n\n');


fprintf('Total subjects: %d\n', numel(files));
fprintf('Successful:     %d\n', success_count);
fprintf('Failed:         %d\n\n', failed_count);


%% ------------------------------------------------------------------------
% WRITE CSV REPORT
% -------------------------------------------------------------------------

report_file = fullfile( ...
    VBM_ROOT, ...
    'openneuro_vbm_report.csv');


fid = fopen(report_file,'w');


if fid == -1

    warning('Could not create report file.');

else

    fprintf(fid, ...
        'subject,input,status,output_or_error\n');


    for i = 1:size(results,1)

        fprintf(fid, ...
            '"%s","%s","%s","%s"\n', ...
            results{i,1}, ...
            results{i,2}, ...
            results{i,3}, ...
            strrep(results{i,4},'"','""'));

    end


    fclose(fid);

end


fprintf('Report:\n%s\n\n', report_file);


%% ------------------------------------------------------------------------
% FINAL STATUS
% -------------------------------------------------------------------------

if failed_count == 0

    fprintf('====================================================================\n');
    fprintf('OVERALL STATUS: PASS\n');
    fprintf('All OpenNeuro subjects completed VBM successfully.\n');
    fprintf('====================================================================\n');

else

    fprintf('====================================================================\n');
    fprintf('OVERALL STATUS: CHECK FAILED SUBJECTS\n');
    fprintf('%d subject(s) failed.\n', failed_count);
    fprintf('====================================================================\n');

end


fprintf('\nVBM root:\n%s\n', VBM_ROOT);

fprintf('\nNext stage after VBM:\n');
fprintf('AAL atlas ROI extraction -> 116 regional GM-volume features.\n');

fprintf('\n');