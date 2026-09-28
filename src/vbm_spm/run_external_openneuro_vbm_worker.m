% =========================================================================
% NeuroFusion AI - External OpenNeuro sMRI VBM
% FINAL 43-SUBJECT WORKER
%
% TARGET:
%   43 independent OpenNeuro ds000115 subjects
%
% INPUT:
%   processed_data\smri\openneuro_ds000115\preprocessed\normalized\
%
% IMPORTANT:
%   The normalized files are FLAT files.
%
%   Example:
%   sub-01_T1w_ras_n4_bet_normalized.nii.gz
%
%   There are NO:
%       sub-01\anat\
%
% WORKER SPLIT:
%   Worker 1 -> subjects 1,3,5,...
%   Worker 2 -> subjects 2,4,6,...
%
% PIPELINE:
%   normalized T1
%       ->
%   SPM25 New Segment
%       ->
%   c1 GM
%   wc1 GM
%   mwc1 GM
%   y_ deformation
%       ->
%   8 mm smoothing
%       ->
%   s8mwc1 GM
%       ->
%   numerical QC
%
% RESUMABLE:
%   Existing completed outputs are reused.
%
% THIS FILE IS A SCRIPT.
% DO NOT ADD "function ... end".
% =========================================================================

clear;
clc;

% =========================================================================
% INITIALIZE SPM
% =========================================================================

spm('Defaults','fmri');
spm_get_defaults('cmdline',true);
spm_jobman('initcfg');

fprintf('\n');
fprintf('====================================================================\n');
fprintf('NeuroFusion AI - EXTERNAL OPENNEURO VBM WORKER\n');
fprintf('====================================================================\n');

% =========================================================================
% PATHS
% =========================================================================

PROJECT_ROOT = ...
    'C:\Users\User\Documents\neurofusion';

INPUT_ROOT = fullfile( ...
    PROJECT_ROOT, ...
    'processed_data', ...
    'smri', ...
    'openneuro_ds000115', ...
    'preprocessed', ...
    'normalized');

OUTPUT_ROOT = fullfile( ...
    PROJECT_ROOT, ...
    'processed_data', ...
    'smri', ...
    'openneuro_ds000115', ...
    'vbm');

TPM_FILE = fullfile( ...
    spm('Dir'), ...
    'tpm', ...
    'TPM.nii');

% =========================================================================
% WORKER ID
% =========================================================================

worker_string = getenv('VBM_WORKER_ID');

if isempty(worker_string)

    error([ ...
        'VBM_WORKER_ID was not supplied.' newline ...
        'Start this script using the supplied 2-worker BAT launcher.']);

end

worker_id = str2double(worker_string);

if isnan(worker_id) || ~ismember(worker_id,[1 2])

    error('VBM_WORKER_ID must be 1 or 2.');

end

n_workers = 2;

% =========================================================================
% DISPLAY CONFIGURATION
% =========================================================================

fprintf('\nWorker: %d/%d\n',worker_id,n_workers);

fprintf('\nSPM directory:\n%s\n',spm('Dir'));

fprintf('\nTPM:\n%s\n',TPM_FILE);

fprintf('\nInput root:\n%s\n',INPUT_ROOT);

fprintf('\nOutput root:\n%s\n',OUTPUT_ROOT);

% =========================================================================
% VERIFY PATHS
% =========================================================================

if ~isfolder(INPUT_ROOT)

    error([ ...
        'Input directory not found:' newline ...
        '%s'],INPUT_ROOT);

end

if ~isfile(TPM_FILE)

    error([ ...
        'SPM TPM file not found:' newline ...
        '%s'],TPM_FILE);

end

if ~isfolder(OUTPUT_ROOT)

    mkdir(OUTPUT_ROOT);

end

% =========================================================================
% DISCOVER FLAT NORMALIZED FILES
% =========================================================================

fprintf('\n');
fprintf('Discovering normalized external subjects...\n');

files = dir(fullfile( ...
    INPUT_ROOT, ...
    '*_normalized.nii.gz'));

if isempty(files)

    error([ ...
        'No *_normalized.nii.gz files found in:' newline ...
        '%s'],INPUT_ROOT);

end

% =========================================================================
% EXTRACT SUBJECT IDs
% =========================================================================

subjects = {};
subject_numbers = [];

for i = 1:numel(files)

    filename = files(i).name;

    token = regexp( ...
        filename, ...
        '^sub-(\d+)_.*_normalized\.nii\.gz$', ...
        'tokens', ...
        'once');

    if isempty(token)

        fprintf('\nWARNING: Ignoring unexpected file:\n%s\n',filename);

        continue;

    end

    subject_number = str2double(token{1});

    subjects{end+1,1} = sprintf('sub-%d',subject_number);

    subject_numbers(end+1,1) = subject_number;

end

% =========================================================================
% SORT BY NUMERIC SUBJECT NUMBER
% =========================================================================

[subject_numbers,sort_idx] = sort(subject_numbers);

subjects = subjects(sort_idx);

% Reorder file list using the same subject ordering.

valid_files = files;

valid_names = {valid_files.name};

sorted_files = repmat(valid_files(1),numel(sort_idx),1);

for i = 1:numel(sort_idx)

    sorted_files(i) = valid_files(sort_idx(i));

end

files = sorted_files;

% =========================================================================
% HARD SAFETY CHECK
% =========================================================================

n_subjects = numel(subjects);

fprintf('\n');
fprintf('Normalized external subjects discovered: %d\n',n_subjects);

if n_subjects ~= 43

    error([ ...
        newline ...
        '============================================================' newline ...
        'EXPECTED EXACTLY 43 EXTERNAL SUBJECTS' newline ...
        'FOUND %d SUBJECTS' newline ...
        'NO VBM PROCESSING WAS STARTED' newline ...
        '============================================================'], ...
        n_subjects);

end

% =========================================================================
% DISPLAY SUBJECT LIST
% =========================================================================

fprintf('\nComplete external subject list:\n');

for i = 1:n_subjects

    fprintf( ...
        '  %02d -> %s -> %s\n', ...
        i, ...
        subjects{i}, ...
        files(i).name);

end

% =========================================================================
% SPLIT BETWEEN WORKERS
% =========================================================================

worker_indices = worker_id:n_workers:n_subjects;

worker_subjects = subjects(worker_indices);

worker_files = files(worker_indices);

fprintf('\n');
fprintf('====================================================================\n');
fprintf('WORKER %d ASSIGNMENT\n',worker_id);
fprintf('====================================================================\n');

fprintf('Subjects assigned: %d\n',numel(worker_subjects));

fprintf('\nAssigned subjects:\n');

for i = 1:numel(worker_subjects)

    fprintf( ...
        '  global index %02d -> %s\n', ...
        worker_indices(i), ...
        worker_subjects{i});

end

% =========================================================================
% COUNTERS
% =========================================================================

n_processed = 0;
n_skipped = 0;
n_failed = 0;

% =========================================================================
% MAIN PROCESSING LOOP
% =========================================================================

for ii = 1:numel(worker_subjects)

    subject = worker_subjects{ii};

    global_index = worker_indices(ii);

    input_filename = worker_files(ii).name;

    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('[%d/%d] WORKER %d -> %s\n', ...
        ii, ...
        numel(worker_subjects), ...
        worker_id, ...
        subject);
    fprintf('Global index: %d\n',global_index);
    fprintf('Input file: %s\n',input_filename);
    fprintf('====================================================================\n');

    % =====================================================================
    % SOURCE INPUT
    % =====================================================================

    source_file = fullfile( ...
        INPUT_ROOT, ...
        input_filename);

    if ~isfile(source_file)

        fprintf('\nERROR: Input file does not exist.\n');

        n_failed = n_failed + 1;

        continue;

    end

    fprintf('\nSource input:\n%s\n',source_file);

    % =====================================================================
    % SUBJECT OUTPUT DIRECTORY
    % =====================================================================

    subject_dir = fullfile( ...
        OUTPUT_ROOT, ...
        subject);

    smooth_dir = fullfile( ...
        subject_dir, ...
        'smooth');

    if ~isfolder(subject_dir)

        mkdir(subject_dir);

    end

    if ~isfolder(smooth_dir)

        mkdir(smooth_dir);

    end

    % =====================================================================
    % CORRECTLY HANDLE .nii.gz
    % =====================================================================

    % Remove .gz first.
    uncompressed_name = regexprep( ...
        input_filename, ...
        '\.gz$', ...
        '');

    working_gz = fullfile( ...
        subject_dir, ...
        input_filename);

    working_nii = fullfile( ...
        subject_dir, ...
        uncompressed_name);

    % =====================================================================
    % EXPECTED SPM OUTPUTS
    % =====================================================================

    [~,base_name,ext_name] = fileparts(working_nii);

    % base_name should be:
    %
    % sub-01_T1w_ras_n4_bet_normalized
    %
    % ext_name should be:
    %
    % .nii

    input_base = [base_name ext_name];

    native_gm = fullfile( ...
        subject_dir, ...
        ['c1' input_base]);

    warped_gm = fullfile( ...
        subject_dir, ...
        ['wc1' input_base]);

    modulated_gm = fullfile( ...
        subject_dir, ...
        ['mwc1' input_base]);

    deformation = fullfile( ...
        subject_dir, ...
        ['y_' input_base]);

    % =====================================================================
    % SMOOTHING PATHS
    % =====================================================================

    smooth_input = fullfile( ...
        smooth_dir, ...
        ['mwc1' input_base]);

    smooth_gm = fullfile( ...
        smooth_dir, ...
        ['s8mwc1' input_base]);

    qc_file = fullfile( ...
        subject_dir, ...
        [subject '_VBM_QC.txt']);

    % =====================================================================
    % RESUME CHECK
    % =====================================================================

    if isfile(smooth_gm)

        fprintf('\nFINAL SMOOTHED VBM ALREADY EXISTS.\n');

        fprintf('Skipping:\n%s\n',smooth_gm);

        n_skipped = n_skipped + 1;

        continue;

    end

    % =====================================================================
    % SUBJECT PROCESSING
    % =====================================================================

    try

        % =================================================================
        % STEP 1 - COPY INPUT
        % =================================================================

        if ~isfile(working_gz)

            fprintf('\nCopying normalized input...\n');

            [ok,msg] = copyfile( ...
                source_file, ...
                working_gz);

            if ~ok

                error( ...
                    'Could not copy normalized input: %s', ...
                    msg);

            end

        else

            fprintf('\nExisting working input found.\n');

        end

        % =================================================================
        % STEP 2 - UNCOMPRESS
        % =================================================================

        if ~isfile(working_nii)

            fprintf('\nUncompressing normalized input...\n');

            gunzip( ...
                working_gz, ...
                subject_dir);

        else

            fprintf('\nExisting uncompressed NIfTI found.\n');

        end

        if ~isfile(working_nii)

            error( ...
                'Uncompressed NIfTI was not created:\n%s', ...
                working_nii);

        end

        fprintf('\nWorking NIfTI:\n%s\n',working_nii);

        % =================================================================
        % STEP 3 - INPUT QC
        % =================================================================

        fprintf('\nReading input with SPM...\n');

        Vinput = spm_vol([working_nii ',1']);

        Yinput = spm_read_vols(Vinput);

        fprintf('\nInput dimensions:\n');

        fprintf( ...
            '  %d x %d x %d\n', ...
            Vinput.dim(1), ...
            Vinput.dim(2), ...
            Vinput.dim(3));

        if any(~isfinite(Yinput(:)))

            error( ...
                'Input contains non-finite voxel values.');

        end

        if all(Yinput(:) == 0)

            error( ...
                'Input is completely zero.');

        end

        % =================================================================
        % STEP 4 - SPM NEW SEGMENT
        % =================================================================

        segmentation_complete = ...
            isfile(native_gm) && ...
            isfile(warped_gm) && ...
            isfile(modulated_gm) && ...
            isfile(deformation);

        if segmentation_complete

            fprintf('\nSegmentation outputs already complete.\n');

            fprintf('Reusing existing segmentation.\n');

        else

            fprintf('\n');
            fprintf('------------------------------------------------------------\n');
            fprintf('Running SPM25 New Segment: %s\n',subject);
            fprintf('------------------------------------------------------------\n');

            clear matlabbatch;

            % =============================================================
            % CHANNEL
            % =============================================================

            matlabbatch{1}.spm.spatial.preproc.channel.vols = {
                [working_nii ',1']
            };

            matlabbatch{1}.spm.spatial.preproc.channel.biasreg = ...
                0.001;

            matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = ...
                60;

            matlabbatch{1}.spm.spatial.preproc.channel.write = ...
                [0 1];

            % =============================================================
            % TISSUE 1 - GREY MATTER
            % =============================================================

            matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {
                [TPM_FILE ',1']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;

            matlabbatch{1}.spm.spatial.preproc.tissue(1).native = ...
                [1 0];

            matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = ...
                [1 1];

            % =============================================================
            % TISSUE 2 - WHITE MATTER
            % =============================================================

            matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {
                [TPM_FILE ',2']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;

            matlabbatch{1}.spm.spatial.preproc.tissue(2).native = ...
                [1 0];

            matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = ...
                [1 1];

            % =============================================================
            % TISSUE 3 - CSF
            % =============================================================

            matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {
                [TPM_FILE ',3']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;

            matlabbatch{1}.spm.spatial.preproc.tissue(3).native = ...
                [1 0];

            matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = ...
                [1 1];

            % =============================================================
            % TISSUE 4
            % =============================================================

            matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {
                [TPM_FILE ',4']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;

            matlabbatch{1}.spm.spatial.preproc.tissue(4).native = ...
                [0 0];

            matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = ...
                [0 0];

            % =============================================================
            % TISSUE 5
            % =============================================================

            matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {
                [TPM_FILE ',5']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;

            matlabbatch{1}.spm.spatial.preproc.tissue(5).native = ...
                [0 0];

            matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = ...
                [0 0];

            % =============================================================
            % TISSUE 6
            % =============================================================

            matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {
                [TPM_FILE ',6']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;

            matlabbatch{1}.spm.spatial.preproc.tissue(6).native = ...
                [0 0];

            matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = ...
                [0 0];

            % =============================================================
            % WARP
            % =============================================================

            matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;

            matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;

            matlabbatch{1}.spm.spatial.preproc.warp.reg = ...
                [0 0.001 0.5 0.05 0.2];

            matlabbatch{1}.spm.spatial.preproc.warp.affreg = ...
                'mni';

            matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;

            matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;

            matlabbatch{1}.spm.spatial.preproc.warp.write = ...
                [1 1];

            % =============================================================
            % RUN SEGMENTATION
            % =============================================================

            spm_jobman('initcfg');

            spm_jobman('run',matlabbatch);

            % =============================================================
            % VERIFY SEGMENTATION
            % =============================================================

            if ~isfile(native_gm)

                error( ...
                    'Native GM c1 was not created:\n%s', ...
                    native_gm);

            end

            if ~isfile(warped_gm)

                error( ...
                    'Warped GM wc1 was not created:\n%s', ...
                    warped_gm);

            end

            if ~isfile(modulated_gm)

                error( ...
                    'Modulated GM mwc1 was not created:\n%s', ...
                    modulated_gm);

            end

            if ~isfile(deformation)

                error( ...
                    'Deformation field was not created:\n%s', ...
                    deformation);

            end

            fprintf('\nSPM segmentation completed successfully.\n');

        end

        % =================================================================
        % STEP 5 - MWC1 QC
        % =================================================================

        fprintf('\nChecking modulated GM...\n');

        M = spm_vol([modulated_gm ',1']);

        Y = spm_read_vols(M);

        finite_count = sum(isfinite(Y(:)));

        total_count = numel(Y);

        nonzero_count = sum(Y(:) ~= 0);

        fprintf('\nMWC1 dimensions:\n');

        fprintf( ...
            '  %d x %d x %d\n', ...
            M.dim(1), ...
            M.dim(2), ...
            M.dim(3));

        fprintf('\nFinite voxels:\n');

        fprintf( ...
            '  %d / %d\n', ...
            finite_count, ...
            total_count);

        fprintf('\nNon-zero voxels:\n');

        fprintf( ...
            '  %d\n', ...
            nonzero_count);

        fprintf('\nMinimum:\n');

        fprintf( ...
            '  %.8g\n', ...
            min(Y(:)));

        fprintf('\nMaximum:\n');

        fprintf( ...
            '  %.8g\n', ...
            max(Y(:)));

        fprintf('\nMean:\n');

        fprintf( ...
            '  %.8g\n', ...
            mean(Y(:)));

        if finite_count ~= total_count

            error( ...
                'MWC1 contains non-finite values.');

        end

        if nonzero_count == 0

            error( ...
                'MWC1 is completely zero.');

        end

        % =================================================================
        % STEP 6 - PREPARE SMOOTHING
        % =================================================================

        if ~isfile(smooth_input)

            fprintf('\nCopying MWC1 to smoothing directory...\n');

            [ok,msg] = copyfile( ...
                modulated_gm, ...
                smooth_input);

            if ~ok

                error( ...
                    'Could not copy MWC1 for smoothing: %s', ...
                    msg);

            end

        else

            fprintf('\nExisting smoothing input found.\n');

        end

        % =================================================================
        % STEP 7 - 8 MM SMOOTHING
        % =================================================================

        if isfile(smooth_gm)

            fprintf('\nExisting smoothed MWC1 found.\n');

        else

            fprintf('\n');
            fprintf('Running 8-mm smoothing...\n');

            clear matlabbatch;

            matlabbatch{1}.spm.spatial.smooth.data = {
                [smooth_input ',1']
            };

            matlabbatch{1}.spm.spatial.smooth.fwhm = ...
                [8 8 8];

            matlabbatch{1}.spm.spatial.smooth.dtype = 0;

            matlabbatch{1}.spm.spatial.smooth.im = 0;

            matlabbatch{1}.spm.spatial.smooth.prefix = ...
                's8';

            spm_jobman('run',matlabbatch);

        end

        % =================================================================
        % STEP 8 - VERIFY SMOOTHED VBM
        % =================================================================

        if ~isfile(smooth_gm)

            error( ...
                'Final smoothed VBM image was not created:\n%s', ...
                smooth_gm);

        end

        S = spm_vol([smooth_gm ',1']);

        Ys = spm_read_vols(S);

        smooth_finite = sum(isfinite(Ys(:)));

        if smooth_finite ~= numel(Ys)

            error( ...
                'Smoothed VBM contains non-finite values.');

        end

        if all(Ys(:) == 0)

            error( ...
                'Smoothed VBM is completely zero.');

        end

        % =================================================================
        % STEP 9 - QC REPORT
        % =================================================================

        fid = fopen(qc_file,'w');

        if fid ~= -1

            fprintf( ...
                fid, ...
                'NeuroFusion AI - External OpenNeuro VBM QC\n\n');

            fprintf( ...
                fid, ...
                'Subject: %s\n', ...
                subject);

            fprintf( ...
                fid, ...
                'Worker: %d/%d\n', ...
                worker_id, ...
                n_workers);

            fprintf( ...
                fid, ...
                'Global index: %d\n\n', ...
                global_index);

            fprintf( ...
                fid, ...
                'Source input:\n%s\n\n', ...
                source_file);

            fprintf( ...
                fid, ...
                'Working NIfTI:\n%s\n\n', ...
                working_nii);

            fprintf( ...
                fid, ...
                'Native GM:\n%s\n\n', ...
                native_gm);

            fprintf( ...
                fid, ...
                'Warped GM:\n%s\n\n', ...
                warped_gm);

            fprintf( ...
                fid, ...
                'Modulated GM:\n%s\n\n', ...
                modulated_gm);

            fprintf( ...
                fid, ...
                'Deformation:\n%s\n\n', ...
                deformation);

            fprintf( ...
                fid, ...
                'Smoothed GM:\n%s\n\n', ...
                smooth_gm);

            fprintf( ...
                fid, ...
                'Input dimensions: [%s]\n', ...
                num2str(Vinput.dim));

            fprintf( ...
                fid, ...
                'MWC1 dimensions: [%s]\n', ...
                num2str(M.dim));

            fprintf( ...
                fid, ...
                'Smoothed dimensions: [%s]\n', ...
                num2str(S.dim));

            fprintf( ...
                fid, ...
                'MWC1 finite voxels: %d / %d\n', ...
                finite_count, ...
                total_count);

            fprintf( ...
                fid, ...
                'MWC1 non-zero voxels: %d\n', ...
                nonzero_count);

            fprintf( ...
                fid, ...
                'MWC1 minimum: %.8g\n', ...
                min(Y(:)));

            fprintf( ...
                fid, ...
                'MWC1 maximum: %.8g\n', ...
                max(Y(:)));

            fprintf( ...
                fid, ...
                'MWC1 mean: %.8g\n', ...
                mean(Y(:)));

            fprintf( ...
                fid, ...
                'Smoothed finite voxels: %d / %d\n', ...
                smooth_finite, ...
                numel(Ys));

            fprintf( ...
                fid, ...
                'Smoothed minimum: %.8g\n', ...
                min(Ys(:)));

            fprintf( ...
                fid, ...
                'Smoothed maximum: %.8g\n', ...
                max(Ys(:)));

            fprintf( ...
                fid, ...
                'Smoothed mean: %.8g\n', ...
                mean(Ys(:)));

            fprintf( ...
                fid, ...
                '\nTimestamp: %s\n', ...
                datestr(now));

            fclose(fid);

        end

        % =================================================================
        % SUCCESS
        % =================================================================

        fprintf('\n');
        fprintf('############################################################\n');
        fprintf('SUCCESS: %s\n',subject);
        fprintf('############################################################\n');

        fprintf('\nMWC1:\n%s\n',modulated_gm);

        fprintf('\nSmoothed VBM:\n%s\n',smooth_gm);

        fprintf('\nQC:\n%s\n',qc_file);

        n_processed = n_processed + 1;

    catch ME

        % =================================================================
        % FAILURE
        % =================================================================

        n_failed = n_failed + 1;

        fprintf('\n');
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
        fprintf('ERROR processing %s\n',subject);
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');

        fprintf('\n%s\n',ME.message);

        fail_file = fullfile( ...
            subject_dir, ...
            [subject '_VBM_FAILED.txt']);

        fid = fopen(fail_file,'w');

        if fid ~= -1

            fprintf( ...
                fid, ...
                'NeuroFusion AI - External OpenNeuro VBM FAILURE\n\n');

            fprintf( ...
                fid, ...
                'Subject: %s\n', ...
                subject);

            fprintf( ...
                fid, ...
                'Worker: %d/%d\n', ...
                worker_id, ...
                n_workers);

            fprintf( ...
                fid, ...
                'Global index: %d\n\n', ...
                global_index);

            fprintf( ...
                fid, ...
                'Source input:\n%s\n\n', ...
                source_file);

            fprintf( ...
                fid, ...
                'Error:\n%s\n\n', ...
                ME.message);

            fprintf( ...
                fid, ...
                'Timestamp:\n%s\n', ...
                datestr(now));

            fclose(fid);

        end

        fprintf('\nContinuing to next subject...\n');

    end

end

% =========================================================================
% FINAL SUMMARY
% =========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('WORKER %d/%d FINISHED\n',worker_id,n_workers);
fprintf('====================================================================\n');

fprintf('\nAssigned subjects: %d\n',numel(worker_subjects));

fprintf('Processed:                 %d\n',n_processed);

fprintf('Skipped (already done):    %d\n',n_skipped);

fprintf('Failed:                    %d\n',n_failed);

fprintf('\nOutput root:\n%s\n',OUTPUT_ROOT);

if n_failed > 0

    fprintf('\nWARNING: %d subject(s) failed.\n',n_failed);

else

    fprintf('\nWorker completed without processing errors.\n');

end

fprintf('\n');
fprintf('====================================================================\n');
fprintf('WORKER %d DONE\n',worker_id);
fprintf('====================================================================\n');