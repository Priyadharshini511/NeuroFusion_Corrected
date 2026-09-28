% =========================================================================
% NeuroFusion AI - ds004302 3-Subject VBM PILOT
% 2-Worker Parallel Launcher TEST
%
% Worker 1 -> sub-01, sub-55
% Worker 2 -> sub-26
%
% This is ONLY a launcher test.
% It is NOT the 43-subject OpenNeuro run.
% =========================================================================

clear;
clc;

spm('Defaults','fmri');
spm_jobman('initcfg');

fprintf('\n');
fprintf('====================================================================\n');
fprintf('NeuroFusion AI - ds004302 VBM PILOT WORKER\n');
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
    'external_test', ...
    'ds004302', ...
    'pilot_normalized');

OUTPUT_ROOT = fullfile( ...
    PROJECT_ROOT, ...
    'processed_data', ...
    'smri', ...
    'external_test', ...
    'ds004302', ...
    'pilot_vbm_2workers_test');

TPM_FILE = fullfile( ...
    spm('Dir'), ...
    'tpm', ...
    'TPM.nii');

% =========================================================================
% WORKER ID
% =========================================================================

worker_string = getenv('VBM_WORKER_ID');

if isempty(worker_string)
    error('VBM_WORKER_ID was not supplied.');
end

worker_id = str2double(worker_string);

if isnan(worker_id) || ~ismember(worker_id,[1 2])
    error('VBM_WORKER_ID must be 1 or 2.');
end

n_workers = 2;

fprintf('\nWorker: %d/%d\n',worker_id,n_workers);

fprintf('\nSPM:\n%s\n',spm('Dir'));
fprintf('\nTPM:\n%s\n',TPM_FILE);
fprintf('\nInput:\n%s\n',INPUT_ROOT);
fprintf('\nOutput:\n%s\n',OUTPUT_ROOT);

% =========================================================================
% VERIFY
% =========================================================================

if ~isfolder(INPUT_ROOT)
    error('Pilot normalized input directory not found:\n%s',INPUT_ROOT);
end

if ~isfile(TPM_FILE)
    error('SPM TPM file not found:\n%s',TPM_FILE);
end

if ~isfolder(OUTPUT_ROOT)
    mkdir(OUTPUT_ROOT);
end

% =========================================================================
% EXACT 3 SUBJECT PILOT
% =========================================================================

subjects = {
    'sub-01'
    'sub-26'
    'sub-55'
};

fprintf('\n');
fprintf('Pilot subjects: %d\n',numel(subjects));

for i = 1:numel(subjects)
    fprintf('  %d -> %s\n',i,subjects{i});
end

% =========================================================================
% SPLIT
% =========================================================================

worker_indices = worker_id:n_workers:numel(subjects);

worker_subjects = subjects(worker_indices);

fprintf('\n');
fprintf('============================================================\n');
fprintf('WORKER %d ASSIGNMENT\n',worker_id);
fprintf('============================================================\n');

fprintf('Assigned subjects: %d\n',numel(worker_subjects));

for i = 1:numel(worker_subjects)

    fprintf('  global index %02d -> %s\n', ...
        worker_indices(i), ...
        worker_subjects{i});

end

% =========================================================================
% COUNTERS
% =========================================================================

successful = 0;
failed = 0;
skipped = 0;

% =========================================================================
% PROCESS
% =========================================================================

for ii = 1:numel(worker_subjects)

    subject = worker_subjects{ii};

    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('WORKER %d -> %s\n',worker_id,subject);
    fprintf('====================================================================\n');

    try

        % -----------------------------------------------------------------
        % INPUT
        % -----------------------------------------------------------------

        anat_dir = fullfile( ...
            INPUT_ROOT, ...
            subject, ...
            'anat');

        nii_files = dir(fullfile( ...
            anat_dir, ...
            '*_normalized.nii.gz'));

        if isempty(nii_files)
            error('Normalized input not found.');
        end

        if numel(nii_files) ~= 1
            error('Expected exactly one normalized NIfTI.');
        end

        source_file = fullfile( ...
            anat_dir, ...
            nii_files(1).name);

        fprintf('\nInput:\n%s\n',source_file);

        % -----------------------------------------------------------------
        % OUTPUT
        % -----------------------------------------------------------------

        subject_dir = fullfile(OUTPUT_ROOT,subject);

        if ~isfolder(subject_dir)
            mkdir(subject_dir);
        end

        smooth_dir = fullfile(subject_dir,'smooth');

        if ~isfolder(smooth_dir)
            mkdir(smooth_dir);
        end

        input_name = nii_files(1).name;

        working_gz = fullfile( ...
            subject_dir,input_name);

        [~,input_base,~] = fileparts(input_name);

        working_nii = fullfile( ...
            subject_dir,[input_base '.nii']);

        native_gm = fullfile( ...
            subject_dir,['c1' input_base '.nii']);

        warped_gm = fullfile( ...
            subject_dir,['wc1' input_base '.nii']);

        modulated_gm = fullfile( ...
            subject_dir,['mwc1' input_base '.nii']);

        deformation = fullfile( ...
            subject_dir,['y_' input_base '.nii']);

        smooth_input = fullfile( ...
            smooth_dir,['mwc1' input_base '.nii']);

        smooth_gm = fullfile( ...
            smooth_dir,['s8mwc1' input_base '.nii']);

        qc_file = fullfile( ...
            subject_dir,[subject '_VBM_QC.txt']);

        % -----------------------------------------------------------------
        % RESUME
        % -----------------------------------------------------------------

        if isfile(smooth_gm)

            fprintf('\nFINAL OUTPUT ALREADY EXISTS.\n');
            fprintf('%s\n',smooth_gm);

            skipped = skipped + 1;
            continue;

        end

        % -----------------------------------------------------------------
        % COPY
        % -----------------------------------------------------------------

        if ~isfile(working_gz)

            fprintf('\nCopying normalized input...\n');

            [ok,msg] = copyfile(source_file,working_gz);

            if ~ok
                error('Copy failed: %s',msg);
            end

        else

            fprintf('\nExisting working input found.\n');

        end

        % -----------------------------------------------------------------
        % GUNZIP
        % -----------------------------------------------------------------

        if ~isfile(working_nii)

            fprintf('\nUncompressing...\n');

            gunzip(working_gz,subject_dir);

        else

            fprintf('\nExisting uncompressed input found.\n');

        end

        if ~isfile(working_nii)
            error('Uncompressed NIfTI was not created.');
        end

        % -----------------------------------------------------------------
        % INPUT QC
        % -----------------------------------------------------------------

        Vinput = spm_vol([working_nii ',1']);
        Yinput = spm_read_vols(Vinput);

        fprintf('\nInput dimensions:\n');
        fprintf('%d x %d x %d\n', ...
            Vinput.dim(1), ...
            Vinput.dim(2), ...
            Vinput.dim(3));

        if any(~isfinite(Yinput(:)))
            error('Input contains non-finite values.');
        end

        if all(Yinput(:) == 0)
            error('Input is completely zero.');
        end

        % -----------------------------------------------------------------
        % NEW SEGMENT
        % -----------------------------------------------------------------

        segmentation_complete = ...
            isfile(native_gm) && ...
            isfile(warped_gm) && ...
            isfile(modulated_gm) && ...
            isfile(deformation);

        if segmentation_complete

            fprintf('\nExisting segmentation outputs found.\n');
            fprintf('Reusing them.\n');

        else

            fprintf('\n');
            fprintf('Running SPM New Segment...\n');

            clear matlabbatch;

            % CHANNEL
            matlabbatch{1}.spm.spatial.preproc.channel.vols = {
                [working_nii ',1']
            };

            matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
            matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
            matlabbatch{1}.spm.spatial.preproc.channel.write = [0 1];

            % GM
            matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {
                [TPM_FILE ',1']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
            matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [1 1];

            % WM
            matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {
                [TPM_FILE ',2']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
            matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [1 1];

            % CSF
            matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {
                [TPM_FILE ',3']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
            matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [1 1];

            % TISSUE 4
            matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {
                [TPM_FILE ',4']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
            matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [0 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];

            % TISSUE 5
            matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {
                [TPM_FILE ',5']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
            matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [0 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];

            % TISSUE 6
            matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {
                [TPM_FILE ',6']
            };

            matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
            matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];

            % WARP
            matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
            matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
            matlabbatch{1}.spm.spatial.preproc.warp.reg = ...
                [0 0.001 0.5 0.05 0.2];

            matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
            matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
            matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
            matlabbatch{1}.spm.spatial.preproc.warp.write = [1 1];

            spm_jobman('initcfg');
            spm_jobman('run',matlabbatch);

        end

        % -----------------------------------------------------------------
        % VERIFY SEGMENTATION
        % -----------------------------------------------------------------

        if ~isfile(modulated_gm)
            error('MWC1 was not created.');
        end

        if ~isfile(deformation)
            error('Deformation field was not created.');
        end

        % -----------------------------------------------------------------
        % MWC1 QC
        % -----------------------------------------------------------------

        M = spm_vol([modulated_gm ',1']);
        Y = spm_read_vols(M);

        if any(~isfinite(Y(:)))
            error('MWC1 contains non-finite values.');
        end

        if all(Y(:) == 0)
            error('MWC1 is completely zero.');
        end

        fprintf('\nMWC1 dimensions:\n');
        fprintf('%d x %d x %d\n', ...
            M.dim(1),M.dim(2),M.dim(3));

        % -----------------------------------------------------------------
        % COPY FOR SMOOTHING
        % -----------------------------------------------------------------

        if ~isfile(smooth_input)

            fprintf('\nCopying MWC1 for smoothing...\n');

            [ok,msg] = copyfile(modulated_gm,smooth_input);

            if ~ok
                error('Could not copy MWC1: %s',msg);
            end

        end

        % -----------------------------------------------------------------
        % SMOOTH
        % -----------------------------------------------------------------

        if ~isfile(smooth_gm)

            fprintf('\nRunning 8-mm smoothing...\n');

            clear matlabbatch;

            matlabbatch{1}.spm.spatial.smooth.data = {
                [smooth_input ',1']
            };

            matlabbatch{1}.spm.spatial.smooth.fwhm = [8 8 8];
            matlabbatch{1}.spm.spatial.smooth.dtype = 0;
            matlabbatch{1}.spm.spatial.smooth.im = 0;
            matlabbatch{1}.spm.spatial.smooth.prefix = 's8';

            spm_jobman('run',matlabbatch);

        end

        % -----------------------------------------------------------------
        % FINAL QC
        % -----------------------------------------------------------------

        if ~isfile(smooth_gm)
            error('Smoothed MWC1 was not created.');
        end

        S = spm_vol([smooth_gm ',1']);
        Ys = spm_read_vols(S);

        if any(~isfinite(Ys(:)))
            error('Smoothed image contains non-finite values.');
        end

        if all(Ys(:) == 0)
            error('Smoothed image is completely zero.');
        end

        % -----------------------------------------------------------------
        % QC FILE
        % -----------------------------------------------------------------

        fid = fopen(qc_file,'w');

        fprintf(fid,'NeuroFusion AI - ds004302 VBM PILOT QC\n\n');
        fprintf(fid,'Subject: %s\n',subject);
        fprintf(fid,'Worker: %d/%d\n\n',worker_id,n_workers);

        fprintf(fid,'Source:\n%s\n\n',source_file);
        fprintf(fid,'MWC1:\n%s\n\n',modulated_gm);
        fprintf(fid,'Smoothed:\n%s\n\n',smooth_gm);

        fprintf(fid,'Input dimensions: [%s]\n',num2str(Vinput.dim));
        fprintf(fid,'MWC1 dimensions: [%s]\n',num2str(M.dim));
        fprintf(fid,'Smoothed dimensions: [%s]\n',num2str(S.dim));

        fprintf(fid,'MWC1 finite voxels: %d / %d\n', ...
            sum(isfinite(Y(:))),numel(Y));

        fprintf(fid,'MWC1 non-zero voxels: %d\n', ...
            sum(Y(:) ~= 0));

        fprintf(fid,'Smoothed finite voxels: %d / %d\n', ...
            sum(isfinite(Ys(:))),numel(Ys));

        fprintf(fid,'Timestamp: %s\n',datestr(now));

        fclose(fid);

        fprintf('\n');
        fprintf('############################################################\n');
        fprintf('SUCCESS: %s\n',subject);
        fprintf('############################################################\n');

        successful = successful + 1;

    catch ME

        failed = failed + 1;

        fprintf('\n');
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
        fprintf('FAILED: %s\n',subject);
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');

        fprintf('\n%s\n',ME.message);

    end

end

% =========================================================================
% SUMMARY
% =========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('PILOT WORKER %d/%d FINISHED\n',worker_id,n_workers);
fprintf('====================================================================\n');

fprintf('\nPilot cohort: 3 subjects\n');
fprintf('Assigned: %d\n',numel(worker_subjects));
fprintf('Successful: %d\n',successful);
fprintf('Skipped: %d\n',skipped);
fprintf('Failed: %d\n',failed);

fprintf('\nOutput:\n%s\n',OUTPUT_ROOT);

fprintf('\n====================================================================\n');