% =========================================================================
% NeuroFusion AI - External sMRI VBM Worker
% =========================================================================
%
% Purpose:
%   Process the 43-subject independent external cohort using SPM25 VBM.
%
% Two-worker design:
%   Worker 1 -> subjects 1,3,5,...,43  (~22 subjects)
%   Worker 2 -> subjects 2,4,6,...,42  (~21 subjects)
%
% Important:
%   - This is ONLY for the independent external cohort.
%   - The previously completed 119 RMS subjects are NOT touched.
%   - Subjects are automatically discovered from the normalized directory.
%   - Exactly 43 normalized external subjects are required.
%   - Processing is resumable.
%
% Pipeline:
%   normalized T1
%       -> SPM New Segment
%       -> c1 GM
%       -> wc1 GM
%       -> mwc1 GM
%       -> 8-mm smoothing
%       -> numerical QC
%
% =========================================================================

clear;
clc;

spm('Defaults','fmri');
spm_jobman('initcfg');

% =========================================================================
% CONFIGURATION
% =========================================================================

PROJECT_ROOT = ...
    'C:\Users\User\Documents\neurofusion';

EXTERNAL_ROOT = fullfile( ...
    PROJECT_ROOT, ...
    'processed_data', ...
    'smri', ...
    'external_test', ...
    'ds004302');

INPUT_ROOT = fullfile( ...
    EXTERNAL_ROOT, ...
    'pilot_normalized');

OUTPUT_ROOT = fullfile( ...
    EXTERNAL_ROOT, ...
    'external_vbm');

EXPECTED_SUBJECTS = 43;

% =========================================================================
% WORKER ID
% =========================================================================

worker_string = getenv('VBM_EXTERNAL_WORKER_ID');

if isempty(worker_string)

    error([ ...
        'VBM_EXTERNAL_WORKER_ID was not supplied.' newline ...
        'Start this script using run_external_vbm_2workers.bat' ...
    ]);

end

worker_id = str2double(worker_string);

if isnan(worker_id) || ~ismember(worker_id,[1 2])

    error( ...
        'VBM_EXTERNAL_WORKER_ID must be 1 or 2.');

end

n_workers = 2;

% =========================================================================
% SPM TPM
% =========================================================================

TPM_FILE = fullfile( ...
    spm('Dir'), ...
    'tpm', ...
    'TPM.nii');

% =========================================================================
% HEADER
% =========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('NeuroFusion AI - EXTERNAL sMRI VBM WORKER %d/%d\n', ...
    worker_id,n_workers);
fprintf('====================================================================\n');

fprintf('\nSPM directory:\n%s\n',spm('Dir'));

fprintf('\nTPM:\n%s\n',TPM_FILE);

fprintf('\nExternal cohort:\n%s\n',EXTERNAL_ROOT);

fprintf('\nNormalized input root:\n%s\n',INPUT_ROOT);

fprintf('\nVBM output root:\n%s\n',OUTPUT_ROOT);

% =========================================================================
% SAFETY CHECKS
% =========================================================================

if ~isfolder(INPUT_ROOT)

    error( ...
        'External normalized input directory not found:\n%s', ...
        INPUT_ROOT);

end

if ~isfile(TPM_FILE)

    error( ...
        'SPM TPM file not found:\n%s', ...
        TPM_FILE);

end

if ~isfolder(OUTPUT_ROOT)

    mkdir(OUTPUT_ROOT);

end

% =========================================================================
% DISCOVER SUBJECTS
% =========================================================================

fprintf('\n');
fprintf('Discovering external normalized subjects...\n');

subject_dirs = dir(fullfile(INPUT_ROOT,'sub-*'));

subject_dirs = subject_dirs([subject_dirs.isdir]);

subjects = {};

for i = 1:numel(subject_dirs)

    sid = subject_dirs(i).name;

    anat_file = fullfile( ...
        INPUT_ROOT, ...
        sid, ...
        'anat', ...
        [sid '_T1w_n4_normalized.nii.gz']);

    if isfile(anat_file)

        subjects{end+1,1} = sid;

    end

end

% =========================================================================
% SORT SUBJECTS NUMERICALLY
% =========================================================================

if isempty(subjects)

    error( ...
        'No normalized external subjects were found.');

end

subject_numbers = zeros(numel(subjects),1);

for i = 1:numel(subjects)

    token = regexp( ...
        subjects{i}, ...
        '^sub-(\d+)$', ...
        'tokens', ...
        'once');

    if isempty(token)

        error( ...
            'Unexpected subject ID format: %s', ...
            subjects{i});

    end

    subject_numbers(i) = str2double(token{1});

end

[~,sort_idx] = sort(subject_numbers);

subjects = subjects(sort_idx);

% =========================================================================
% EXACT 43-SUBJECT SAFETY CHECK
% =========================================================================

fprintf('\n');
fprintf('Discovered normalized external subjects: %d\n', ...
    numel(subjects));

if numel(subjects) ~= EXPECTED_SUBJECTS

    fprintf('\nDiscovered subjects:\n');

    for i = 1:numel(subjects)

        fprintf('  %02d  %s\n',i,subjects{i});

    end

    error([ ...
        newline ...
        'SAFETY STOP:' newline ...
        'Expected exactly 43 external normalized subjects.' newline ...
        'Found %d subjects.' newline ...
        'NO VBM PROCESSING WAS STARTED.' newline ...
    ],numel(subjects));

end

% =========================================================================
% DISPLAY COMPLETE EXTERNAL COHORT
% =========================================================================

fprintf('\n');
fprintf('Validated external cohort: %d subjects\n', ...
    numel(subjects));

fprintf('\nComplete external subject list:\n');

for i = 1:numel(subjects)

    fprintf('  %02d  %s\n',i,subjects{i});

end

% =========================================================================
% SPLIT BETWEEN WORKERS
% =========================================================================

worker_indices = worker_id:n_workers:numel(subjects);

worker_subjects = subjects(worker_indices);

fprintf('\n');
fprintf('====================================================================\n');
fprintf('WORKER ASSIGNMENT\n');
fprintf('====================================================================\n');

fprintf('\nTotal external subjects: %d\n',numel(subjects));

fprintf('Worker %d assigned: %d subjects\n', ...
    worker_id,numel(worker_subjects));

fprintf('Worker %d first position: %d\n', ...
    worker_id,worker_indices(1));

fprintf('Worker %d last position: %d\n', ...
    worker_id,worker_indices(end));

fprintf('\nWorker %d subjects:\n',worker_id);

for i = 1:numel(worker_subjects)

    fprintf('  %02d  %s\n', ...
        worker_indices(i), ...
        worker_subjects{i});

end

% =========================================================================
% VERIFY EXACT INPUTS FOR THIS WORKER
% =========================================================================

fprintf('\n');
fprintf('Checking exact normalized inputs...\n');

missing_inputs = {};

for i = 1:numel(worker_subjects)

    sid = worker_subjects{i};

    input_file = fullfile( ...
        INPUT_ROOT, ...
        sid, ...
        'anat', ...
        [sid '_T1w_n4_normalized.nii.gz']);

    if ~isfile(input_file)

        missing_inputs{end+1,1} = input_file;

    end

end

if ~isempty(missing_inputs)

    fprintf('\nMissing inputs:\n');

    for i = 1:numel(missing_inputs)

        fprintf('  %s\n',missing_inputs{i});

    end

    error('Missing external normalized input(s) detected.');

end

fprintf( ...
    'All %d worker inputs found.\n', ...
    numel(worker_subjects));

% =========================================================================
% COUNTERS
% =========================================================================

n_processed = 0;
n_skipped = 0;
n_failed = 0;

% =========================================================================
% MAIN LOOP
% =========================================================================

for ii = 1:numel(worker_subjects)

    subject = worker_subjects{ii};

    fprintf('\n');
    fprintf('--------------------------------------------------------------------\n');
    fprintf('[%d/%d] Worker %d processing %s\n', ...
        ii, ...
        numel(worker_subjects), ...
        worker_id, ...
        subject);
    fprintf('--------------------------------------------------------------------\n');

    % ---------------------------------------------------------------------
    % INPUT
    % ---------------------------------------------------------------------

    source_file = fullfile( ...
        INPUT_ROOT, ...
        subject, ...
        'anat', ...
        [subject '_T1w_n4_normalized.nii.gz']);

    % ---------------------------------------------------------------------
    % SUBJECT OUTPUT DIRECTORY
    % ---------------------------------------------------------------------

    subject_dir = fullfile( ...
        OUTPUT_ROOT, ...
        subject);

    if ~isfolder(subject_dir)

        mkdir(subject_dir);

    end

    smooth_dir = fullfile( ...
        subject_dir, ...
        'smooth');

    if ~isfolder(smooth_dir)

        mkdir(smooth_dir);

    end

    % ---------------------------------------------------------------------
    % FILE PATHS
    % ---------------------------------------------------------------------

    working_gz = fullfile( ...
        subject_dir, ...
        [subject '_T1w_n4_normalized.nii.gz']);

    working_nii = fullfile( ...
        subject_dir, ...
        [subject '_T1w_n4_normalized.nii']);

    native_gm = fullfile( ...
        subject_dir, ...
        ['c1' subject '_T1w_n4_normalized.nii']);

    warped_gm = fullfile( ...
        subject_dir, ...
        ['wc1' subject '_T1w_n4_normalized.nii']);

    modulated_gm = fullfile( ...
        subject_dir, ...
        ['mwc1' subject '_T1w_n4_normalized.nii']);

    deformation = fullfile( ...
        subject_dir, ...
        ['y_' subject '_T1w_n4_normalized.nii']);

    smooth_input = fullfile( ...
        smooth_dir, ...
        ['mwc1' subject '_T1w_n4_normalized.nii']);

    smooth_gm = fullfile( ...
        smooth_dir, ...
        ['s8mwc1' subject '_T1w_n4_normalized.nii']);

    qc_file = fullfile( ...
        subject_dir, ...
        [subject '_VBM_QC.txt']);

    fail_file = fullfile( ...
        subject_dir, ...
        [subject '_VBM_FAILED.txt']);

    % ---------------------------------------------------------------------
    % RESUME CHECK
    % ---------------------------------------------------------------------

    if isfile(smooth_gm)

        fprintf('\nFINAL OUTPUT ALREADY EXISTS.\n');
        fprintf('%s\n',smooth_gm);
        fprintf('\n%s - COMPLETE, skipping.\n',subject);

        n_skipped = n_skipped + 1;

        continue;

    end

    try

        % ================================================================
        % STEP 1 - COPY INPUT
        % ================================================================

        if ~isfile(working_gz)

            fprintf('\nCopying normalized input...\n');

            [ok,msg] = copyfile( ...
                source_file, ...
                working_gz);

            if ~ok

                error( ...
                    'Could not copy input: %s', ...
                    msg);

            end

        else

            fprintf('\nWorking input already exists.\n');

        end

        % ================================================================
        % STEP 2 - UNCOMPRESS
        % ================================================================

        if ~isfile(working_nii)

            fprintf('\nUncompressing normalized input...\n');

            gunzip(working_gz,subject_dir);

        else

            fprintf('\nUncompressed input already exists.\n');

        end

        if ~isfile(working_nii)

            error( ...
                'Uncompressed NIfTI was not created.');

        end

        % ================================================================
        % STEP 3 - INPUT QC
        % ================================================================

        V = spm_vol([working_nii ',1']);

        fprintf('\nInput dimensions:\n');
        fprintf('  %d x %d x %d\n', ...
            V.dim(1), ...
            V.dim(2), ...
            V.dim(3));

        fprintf('\nInput affine:\n');
        disp(V.mat);

        % ================================================================
        % STEP 4 - SEGMENTATION
        % ================================================================

        segmentation_complete = ...
            isfile(native_gm) && ...
            isfile(warped_gm) && ...
            isfile(modulated_gm) && ...
            isfile(deformation);

        if segmentation_complete

            fprintf('\n');
            fprintf('SPM segmentation outputs already complete.\n');
            fprintf('Reusing existing segmentation.\n');

        else

            fprintf('\n');
            fprintf('Starting SPM25 New Segment...\n');

            clear matlabbatch;

            % ------------------------------------------------------------
            % CHANNEL
            % ------------------------------------------------------------

            matlabbatch{1}.spm.spatial.preproc.channel.vols = {
                [working_nii ',1']
            };

            matlabbatch{1}.spm.spatial.preproc.channel.biasreg = ...
                0.001;

            matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = ...
                60;

            matlabbatch{1}.spm.spatial.preproc.channel.write = ...
                [0 1];

            % ------------------------------------------------------------
            % TISSUES
            % ------------------------------------------------------------

            ngauss = [1 1 2 3 4 2];

            for k = 1:6

                matlabbatch{1}.spm.spatial.preproc.tissue(k).tpm = {
                    [TPM_FILE ',' num2str(k)]
                };

                matlabbatch{1}.spm.spatial.preproc.tissue(k).ngaus = ...
                    ngauss(k);

                if k <= 3

                    matlabbatch{1}.spm.spatial.preproc.tissue(k).native = ...
                        [1 0];

                    matlabbatch{1}.spm.spatial.preproc.tissue(k).warped = ...
                        [1 1];

                else

                    matlabbatch{1}.spm.spatial.preproc.tissue(k).native = ...
                        [0 0];

                    matlabbatch{1}.spm.spatial.preproc.tissue(k).warped = ...
                        [0 0];

                end

            end

            % ------------------------------------------------------------
            % WARP
            % ------------------------------------------------------------

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

            % ------------------------------------------------------------
            % RUN
            % ------------------------------------------------------------

            spm_jobman('initcfg');

            spm_jobman('run',matlabbatch);

            % ------------------------------------------------------------
            % VERIFY
            % ------------------------------------------------------------

            if ~isfile(native_gm)

                error('Native GM c1 was not created.');

            end

            if ~isfile(warped_gm)

                error('Warped GM wc1 was not created.');

            end

            if ~isfile(modulated_gm)

                error('Modulated GM mwc1 was not created.');

            end

            if ~isfile(deformation)

                error('Deformation field was not created.');

            end

        end

        % ================================================================
        % STEP 5 - MWC1 QC
        % ================================================================

        fprintf('\nChecking mwc1...\n');

        M = spm_vol([modulated_gm ',1']);

        Y = spm_read_vols(M);

        finite_count = sum(isfinite(Y(:)));

        total_count = numel(Y);

        nonzero_count = sum(Y(:) ~= 0);

        fprintf('Dimensions:\n');
        fprintf('  %d x %d x %d\n', ...
            M.dim(1), ...
            M.dim(2), ...
            M.dim(3));

        fprintf('Finite voxels:\n');
        fprintf('  %d / %d\n', ...
            finite_count, ...
            total_count);

        fprintf('Non-zero voxels:\n');
        fprintf('  %d\n',nonzero_count);

        fprintf('Minimum:\n');
        fprintf('  %.8g\n',min(Y(:)));

        fprintf('Maximum:\n');
        fprintf('  %.8g\n',max(Y(:)));

        fprintf('Mean:\n');
        fprintf('  %.8g\n',mean(Y(:)));

        if finite_count ~= total_count

            error( ...
                'mwc1 contains non-finite values.');

        end

        if nonzero_count == 0

            error( ...
                'mwc1 is completely zero.');

        end

        % ================================================================
        % STEP 6 - PREPARE SMOOTHING
        % ================================================================

        if ~isfile(smooth_input)

            fprintf('\nCopying mwc1 to smoothing directory...\n');

            [ok,msg] = copyfile( ...
                modulated_gm, ...
                smooth_input);

            if ~ok

                error( ...
                    'Could not create smoothing input: %s', ...
                    msg);

            end

        else

            fprintf('\nSmoothing input already exists.\n');

        end

        % ================================================================
        % STEP 7 - 8-MM SMOOTHING
        % ================================================================

        if isfile(smooth_gm)

            fprintf('\nSmoothed mwc1 already exists.\n');

        else

            fprintf('\nStarting 8-mm smoothing...\n');

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

        % ================================================================
        % STEP 8 - FINAL SMOOTHED QC
        % ================================================================

        if ~isfile(smooth_gm)

            error( ...
                'Final smoothed VBM image was not created.');

        end

        S = spm_vol([smooth_gm ',1']);

        Ys = spm_read_vols(S);

        smooth_finite_count = sum(isfinite(Ys(:)));

        smooth_total_count = numel(Ys);

        smooth_nonzero_count = sum(Ys(:) ~= 0);

        if smooth_finite_count ~= smooth_total_count

            error( ...
                'Smoothed mwc1 contains non-finite values.');

        end

        if smooth_nonzero_count == 0

            error( ...
                'Smoothed mwc1 is completely zero.');

        end

        % ================================================================
        % STEP 9 - QC REPORT
        % ================================================================

        fid = fopen(qc_file,'w');

        if fid == -1

            error( ...
                'Could not create QC file.');

        end

        fprintf(fid, ...
            'NeuroFusion AI - External ds004302 VBM QC\n\n');

        fprintf(fid, ...
            'Subject: %s\n',subject);

        fprintf(fid, ...
            'Worker: %d/%d\n\n',worker_id,n_workers);

        fprintf(fid, ...
            'Source input:\n%s\n\n',source_file);

        fprintf(fid, ...
            'Working input:\n%s\n\n',working_nii);

        fprintf(fid, ...
            'Native GM:\n%s\n\n',native_gm);

        fprintf(fid, ...
            'Warped GM:\n%s\n\n',warped_gm);

        fprintf(fid, ...
            'Modulated GM:\n%s\n\n',modulated_gm);

        fprintf(fid, ...
            'Forward deformation:\n%s\n\n',deformation);

        fprintf(fid, ...
            'Smoothed modulated GM:\n%s\n\n',smooth_gm);

        fprintf(fid, ...
            'Input dimensions: [%s]\n',num2str(V.dim));

        fprintf(fid, ...
            'Modulated dimensions: [%s]\n',num2str(M.dim));

        fprintf(fid, ...
            'Modulated finite voxels: %d / %d\n', ...
            finite_count,total_count);

        fprintf(fid, ...
            'Modulated non-zero voxels: %d\n', ...
            nonzero_count);

        fprintf(fid, ...
            'Modulated minimum: %.8g\n', ...
            min(Y(:)));

        fprintf(fid, ...
            'Modulated maximum: %.8g\n', ...
            max(Y(:)));

        fprintf(fid, ...
            'Modulated mean: %.8g\n', ...
            mean(Y(:)));

        fprintf(fid,'\n');

        fprintf(fid, ...
            'Smoothed dimensions: [%s]\n',num2str(S.dim));

        fprintf(fid, ...
            'Smoothed finite voxels: %d / %d\n', ...
            smooth_finite_count, ...
            smooth_total_count);

        fprintf(fid, ...
            'Smoothed non-zero voxels: %d\n', ...
            smooth_nonzero_count);

        fprintf(fid, ...
            'Smoothed minimum: %.8g\n', ...
            min(Ys(:)));

        fprintf(fid, ...
            'Smoothed maximum: %.8g\n', ...
            max(Ys(:)));

        fprintf(fid, ...
            'Smoothed mean: %.8g\n', ...
            mean(Ys(:)));

        fprintf(fid, ...
            '\nStatus: PASS\n');

        fclose(fid);

        % ================================================================
        % SUCCESS
        % ================================================================

        fprintf('\n');
        fprintf('STATUS: PASS\n');

        fprintf('\nSubject: %s\n',subject);

        fprintf('\nmwc1:\n%s\n',modulated_gm);

        fprintf('\nSmoothed VBM:\n%s\n',smooth_gm);

        fprintf('\nQC:\n%s\n',qc_file);

        n_processed = n_processed + 1;

    catch ME

        n_failed = n_failed + 1;

        fprintf('\n');
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
        fprintf('ERROR processing %s\n',subject);
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');

        fprintf('\n%s\n',ME.message);

        % ---------------------------------------------------------------
        % FAILURE RECORD
        % ---------------------------------------------------------------

        fid = fopen(fail_file,'w');

        if fid ~= -1

            fprintf(fid, ...
                'NeuroFusion AI - External VBM FAILURE\n\n');

            fprintf(fid, ...
                'Subject: %s\n',subject);

            fprintf(fid, ...
                'Worker: %d/%d\n\n',worker_id,n_workers);

            fprintf(fid, ...
                'Source input:\n%s\n\n',source_file);

            fprintf(fid, ...
                'Error:\n%s\n\n',ME.message);

            fprintf(fid, ...
                'Timestamp:\n%s\n',datestr(now));

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
fprintf('EXTERNAL VBM WORKER %d/%d FINISHED\n',worker_id,n_workers);
fprintf('====================================================================\n');

fprintf('\nExternal cohort:          %d subjects\n',numel(subjects));

fprintf('Assigned to this worker:  %d\n',numel(worker_subjects));

fprintf('Processed:                %d\n',n_processed);

fprintf('Skipped:                  %d\n',n_skipped);

fprintf('Failed:                   %d\n',n_failed);

fprintf('\nOutput root:\n%s\n',OUTPUT_ROOT);

fprintf('\n');

if n_failed == 0

    fprintf('WORKER %d STATUS: PASS\n',worker_id);

else

    fprintf('WORKER %d STATUS: FAILURES PRESENT\n',worker_id);

end

fprintf('\n');
fprintf('====================================================================\n');