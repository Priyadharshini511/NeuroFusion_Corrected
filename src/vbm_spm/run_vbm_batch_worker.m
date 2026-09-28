% ================================================================
% NeuroFusion AI - SPM25 VBM Batch Worker
% FINAL CORRECTED VERSION
%
% Uses exactly 119 validated RMS subjects from canonical CV file.
% Two workers:
%   Worker 1 = subjects 1,3,5,...
%   Worker 2 = subjects 2,4,6,...
%
% Resumable:
%   - completed subjects are skipped
%   - completed segmentation is reused
%   - completed smoothing is reused
%
% ================================================================

clear;
clc;

spm('Defaults','fmri');
spm_jobman('initcfg');


% ================================================================
% CONFIGURATION
% ================================================================

project_root = ...
    'C:\Users\User\Documents\neurofusion';

n4_dir = fullfile( ...
    project_root, ...
    'processed_data', ...
    'smri', ...
    'n4');

vbm_root = fullfile( ...
    project_root, ...
    'processed_data', ...
    'smri', ...
    'vbm_spm');

assignments_path = fullfile( ...
    project_root, ...
    'processed_data', ...
    'smri', ...
    'cnn_dataset', ...
    'cross_validation', ...
    'nested', ...
    'nested_cv_assignments.csv');


% ================================================================
% WORKER ID
% ================================================================

worker_string = getenv('VBM_WORKER_ID');

if isempty(worker_string)
    error(['VBM_WORKER_ID was not supplied. ' ...
           'Start this script using run_vbm_2workers.bat.']);
end

worker_id = str2double(worker_string);

if isnan(worker_id) || ~ismember(worker_id,[1 2])
    error('VBM_WORKER_ID must be 1 or 2.');
end

n_workers = 2;


% ================================================================
% CHECK PATHS
% ================================================================

if ~exist(assignments_path,'file')
    error('Canonical CV assignment file not found:\n%s', ...
        assignments_path);
end

if ~exist(n4_dir,'dir')
    error('N4 directory not found:\n%s',n4_dir);
end

if ~exist(vbm_root,'dir')
    mkdir(vbm_root);
end


% ================================================================
% LOAD CANONICAL ASSIGNMENTS
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('NeuroFusion AI - SPM25 VBM WORKER %d/%d\n', ...
    worker_id,n_workers);
fprintf('============================================================\n');

fprintf('\nCanonical assignment file:\n%s\n', ...
    assignments_path);

T = readtable(assignments_path);


% ================================================================
% REQUIRED COLUMNS
% ================================================================

required_columns = { ...
    'subject_id', ...
    'acquisition_source', ...
    'label', ...
    'outer_fold', ...
    'inner_fold', ...
    'role'};

for i = 1:numel(required_columns)

    if ~ismember(required_columns{i},T.Properties.VariableNames)

        error( ...
            'Required column missing: %s', ...
            required_columns{i});

    end

end


% ================================================================
% SELECT ONLY RMS SUBJECTS
% ================================================================

source_values = upper(strtrim(string(T.acquisition_source)));

rms_rows = source_values == "RMS";

rms_table = T(rms_rows,:);

rms_subjects = unique( ...
    string(rms_table.subject_id), ...
    'stable');

rms_subjects = sort(rms_subjects);

n_rms = numel(rms_subjects);


fprintf('\n');
fprintf('Canonical assignment rows: %d\n',height(T));
fprintf('Canonical RMS rows:        %d\n',height(rms_table));
fprintf('Unique RMS subjects:       %d\n',n_rms);


% ================================================================
% HARD SAFETY CHECK
% ================================================================

if n_rms ~= 119

    error([ ...
        '\nSAFETY STOP:\n' ...
        'Expected exactly 119 RMS subjects.\n' ...
        'Found %d RMS subjects.\n' ...
        'NO VBM PROCESSING WAS STARTED.\n'], ...
        n_rms);

end


% ================================================================
% SPLIT BETWEEN WORKERS
% ================================================================

worker_indices = worker_id:n_workers:n_rms;

worker_subjects = rms_subjects(worker_indices);


fprintf('\n');
fprintf('Validated RMS cohort: 119 subjects\n');
fprintf('Worker %d subject count: %d\n', ...
    worker_id,numel(worker_subjects));

fprintf('Worker %d first subject index: %d\n', ...
    worker_id,worker_indices(1));

fprintf('Worker %d last subject index: %d\n', ...
    worker_id,worker_indices(end));


% ================================================================
% VERIFY EXACT INPUTS
% ================================================================

fprintf('\nChecking exact N4 inputs...\n');

missing_inputs = strings(0,1);

for i = 1:numel(worker_subjects)

    sid = char(worker_subjects(i));

    input_gz = fullfile( ...
        n4_dir, ...
        [sid '_rms_n4.nii.gz']);

    if ~exist(input_gz,'file')
        missing_inputs(end+1,1) = string(sid);
    end

end


if ~isempty(missing_inputs)

    fprintf('\nMissing inputs:\n');

    for i = 1:numel(missing_inputs)
        fprintf('  %s\n',missing_inputs(i));
    end

    error('Missing N4 input(s) detected.');

end


fprintf('All %d exact N4 inputs found.\n', ...
    numel(worker_subjects));


% ================================================================
% COUNTERS
% ================================================================

n_processed = 0;
n_skipped = 0;
n_failed = 0;


% ================================================================
% MAIN LOOP
% ================================================================

for ii = 1:numel(worker_subjects)

    sid = char(worker_subjects(ii));

    fprintf('\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('[%d/%d] Worker %d processing %s\n', ...
        ii,numel(worker_subjects),worker_id,sid);
    fprintf('------------------------------------------------------------\n');


    % ------------------------------------------------------------
    % Directories
    % ------------------------------------------------------------

    subject_dir = fullfile(vbm_root,sid);

    smooth_dir = fullfile(subject_dir,'smooth');

    if ~exist(subject_dir,'dir')
        mkdir(subject_dir);
    end

    if ~exist(smooth_dir,'dir')
        mkdir(smooth_dir);
    end


    % ------------------------------------------------------------
    % Files
    % ------------------------------------------------------------

    input_gz = fullfile( ...
        n4_dir, ...
        [sid '_rms_n4.nii.gz']);

    input_nii = fullfile( ...
        subject_dir, ...
        [sid '_rms_n4.nii']);

    native_gm = fullfile( ...
        subject_dir, ...
        ['c1' sid '_rms_n4.nii']);

    warped_gm = fullfile( ...
        subject_dir, ...
        ['wc1' sid '_rms_n4.nii']);

    modulated_gm = fullfile( ...
        subject_dir, ...
        ['mwc1' sid '_rms_n4.nii']);

    deformation = fullfile( ...
        subject_dir, ...
        ['y_' sid '_rms_n4.nii']);

    % ------------------------------------------------------------
    % SPM smoothing normally writes s8... beside its input.
    %
    % We therefore put the modulated GM into smooth_dir first.
    % ------------------------------------------------------------

    smooth_input = fullfile( ...
        smooth_dir, ...
        ['mwc1' sid '_rms_n4.nii']);

    smooth_gm = fullfile( ...
        smooth_dir, ...
        ['s8mwc1' sid '_rms_n4.nii']);

    qc_file = fullfile( ...
        subject_dir, ...
        [sid '_VBM_QC.txt']);


    % ============================================================
    % RESUME CHECK
    %
    % Accept both:
    %   1. Correct new location: smooth\s8mwc1...
    %   2. Previously generated location: subject\s8mwc1...
    % ============================================================

    old_smooth_gm = fullfile( ...
        subject_dir, ...
        ['s8mwc1' sid '_rms_n4.nii']);


    if exist(smooth_gm,'file')

        fprintf('%s - COMPLETE, skipping.\n',sid);

        n_skipped = n_skipped + 1;

        continue;

    end


    % If previous version already created smoothing in subject_dir,
    % copy it into the correct smooth directory and continue.

    if exist(old_smooth_gm,'file')

        fprintf('Previously generated smoothed VBM found.\n');
        fprintf('Copying it into the standard smooth directory...\n');

        copyfile(old_smooth_gm,smooth_gm);

        if exist(smooth_gm,'file')

            fprintf('%s - COMPLETE, skipping.\n',sid);

            n_skipped = n_skipped + 1;

            continue;

        end

    end


    try

        % ========================================================
        % STEP 1 - UNCOMPRESS N4
        % ========================================================

        if ~exist(input_nii,'file')

            fprintf('Uncompressing N4 input...\n');

            gunzip(input_gz,subject_dir);

        else

            fprintf('Uncompressed N4 input already exists.\n');

        end


        if ~exist(input_nii,'file')

            error( ...
                'Uncompressed N4 input was not created.');

        end


        % ========================================================
        % STEP 2 - SEGMENTATION
        % ========================================================

        segmentation_complete = ...
            exist(native_gm,'file') && ...
            exist(warped_gm,'file') && ...
            exist(modulated_gm,'file') && ...
            exist(deformation,'file');


        if segmentation_complete

            fprintf('Segmentation already complete.\n');
            fprintf('Reusing existing segmentation outputs.\n');

        else

            fprintf('\nRunning SPM25 New Segment...\n');


            tpm = fullfile( ...
                spm('Dir'), ...
                'tpm', ...
                'TPM.nii');


            if ~exist(tpm,'file')

                error( ...
                    'SPM TPM not found:\n%s',tpm);

            end


            clear matlabbatch;


            matlabbatch{1}.spm.spatial.preproc.channel.vols = ...
                {[input_nii ',1']};

            matlabbatch{1}.spm.spatial.preproc.channel.biasreg = ...
                0.001;

            matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = ...
                60;

            matlabbatch{1}.spm.spatial.preproc.channel.write = ...
                [0 1];


            ngauss = [1 1 2 3 4 2];


            for k = 1:6

                matlabbatch{1}.spm.spatial.preproc.tissue(k).tpm = ...
                    {[tpm ',' num2str(k)]};

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


            spm_jobman('run',matlabbatch);


            if ~exist(native_gm,'file')
                error('Native GM was not created.');
            end

            if ~exist(warped_gm,'file')
                error('Warped GM was not created.');
            end

            if ~exist(modulated_gm,'file')
                error('Modulated GM was not created.');
            end

            if ~exist(deformation,'file')
                error('Deformation field was not created.');
            end

        end


        % ========================================================
        % STEP 3 - PREPARE SMOOTHING INPUT
        % ========================================================

        if ~exist(smooth_input,'file')

            fprintf('\nCopying modulated GM to smooth directory...\n');

            copyfile(modulated_gm,smooth_input);

        else

            fprintf('Smoothing input already exists.\n');

        end


        if ~exist(smooth_input,'file')

            error( ...
                'Could not create smoothing input.');

        end


        % ========================================================
        % STEP 4 - 8 MM SMOOTHING
        % ========================================================

        if exist(smooth_gm,'file')

            fprintf('Smoothed VBM already exists.\n');

        else

            fprintf('\nRunning 8-mm smoothing...\n');

            clear matlabbatch;


            matlabbatch{1}.spm.spatial.smooth.data = ...
                {[smooth_input ',1']};

            matlabbatch{1}.spm.spatial.smooth.fwhm = ...
                [8 8 8];

            matlabbatch{1}.spm.spatial.smooth.dtype = 0;

            matlabbatch{1}.spm.spatial.smooth.im = 0;

            matlabbatch{1}.spm.spatial.smooth.prefix = ...
                's8';


            spm_jobman('run',matlabbatch);

        end


        % ========================================================
        % STEP 5 - FINAL CHECK
        % ========================================================

        if ~exist(smooth_gm,'file')

            error( ...
                'Final smoothed VBM image was not created.');

        end


        % ========================================================
        % STEP 6 - NUMERICAL QC
        % ========================================================

        V = spm_vol(modulated_gm);
        Y = spm_read_vols(V);

        Vs = spm_vol(smooth_gm);
        Ys = spm_read_vols(Vs);


        fid = fopen(qc_file,'w');


        fprintf(fid, ...
            'NeuroFusion AI - SPM25 VBM QC\n\n');

        fprintf(fid, ...
            'Subject: %s\n\n',sid);

        fprintf(fid, ...
            'Input: %s\n\n',input_gz);

        fprintf(fid, ...
            'Native GM: %s\n\n',native_gm);

        fprintf(fid, ...
            'Warped GM: %s\n\n',warped_gm);

        fprintf(fid, ...
            'Modulated GM: %s\n\n',modulated_gm);

        fprintf(fid, ...
            'Deformation: %s\n\n',deformation);

        fprintf(fid, ...
            'Smoothed VBM: %s\n\n',smooth_gm);


        fprintf(fid, ...
            'Modulated dimensions: [%s]\n', ...
            num2str(V.dim));

        fprintf(fid, ...
            'Modulated finite voxels: %d / %d\n', ...
            sum(isfinite(Y(:))),numel(Y));

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
            'Smoothed dimensions: [%s]\n', ...
            num2str(Vs.dim));

        fprintf(fid, ...
            'Smoothed finite voxels: %d / %d\n', ...
            sum(isfinite(Ys(:))),numel(Ys));

        fprintf(fid, ...
            'Smoothed minimum: %.8g\n', ...
            min(Ys(:)));

        fprintf(fid, ...
            'Smoothed maximum: %.8g\n', ...
            max(Ys(:)));

        fprintf(fid, ...
            'Smoothed mean: %.8g\n', ...
            mean(Ys(:)));

        fclose(fid);


        fprintf('\n');
        fprintf('SUCCESS: %s\n',sid);
        fprintf('Final VBM:\n%s\n',smooth_gm);


        n_processed = n_processed + 1;


    catch ME

        n_failed = n_failed + 1;

        fprintf('\n');
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
        fprintf('ERROR processing %s\n',sid);
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');

        fprintf('%s\n',ME.message);


        fail_file = fullfile( ...
            subject_dir, ...
            [sid '_VBM_FAILED.txt']);

        fid = fopen(fail_file,'w');

        fprintf(fid, ...
            'NeuroFusion AI - VBM FAILURE\n\n');

        fprintf(fid, ...
            'Subject: %s\n',sid);

        fprintf(fid, ...
            'Worker: %d/%d\n\n',worker_id,n_workers);

        fprintf(fid, ...
            'Error:\n%s\n\n',ME.message);

        fprintf(fid, ...
            'Timestamp:\n%s\n',datestr(now));

        fclose(fid);


        fprintf('\nContinuing to next subject...\n');

    end

end


% ================================================================
% FINAL SUMMARY
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('WORKER %d/%d FINISHED\n',worker_id,n_workers);
fprintf('============================================================\n');

fprintf('Validated RMS cohort:     119\n');
fprintf('Assigned to this worker:  %d\n',numel(worker_subjects));
fprintf('Processed:                %d\n',n_processed);
fprintf('Skipped:                  %d\n',n_skipped);
fprintf('Failed:                   %d\n',n_failed);

fprintf('============================================================\n');

if n_failed > 0

    fprintf('\nWARNING: %d subject(s) failed.\n',n_failed);

else

    fprintf('\nWorker completed without processing errors.\n');

end

fprintf('\n');