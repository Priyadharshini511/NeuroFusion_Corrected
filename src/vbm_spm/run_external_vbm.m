% =========================================================================
% NeuroFusion AI - External sMRI VBM Pilot
% =========================================================================

clear;
clc;

spm('Defaults','fmri');
spm_jobman('initcfg');

fprintf('\n');
fprintf('====================================================================\n');
fprintf('NeuroFusion AI - EXTERNAL ds004302 VBM PROCESSING\n');
fprintf('====================================================================\n');

% =========================================================================
% PATHS
% =========================================================================

PROJECT_ROOT = 'C:\Users\User\Documents\neurofusion';

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
    'pilot_vbm');

TPM_FILE = fullfile( ...
    spm('Dir'), ...
    'tpm', ...
    'TPM.nii');

% =========================================================================
% SUBJECTS
% =========================================================================

subjects = {
    'sub-01'
    'sub-26'
    'sub-55'
};

% =========================================================================
% DISPLAY
% =========================================================================

fprintf('\nSPM directory:\n%s\n', spm('Dir'));
fprintf('\nTPM:\n%s\n', TPM_FILE);
fprintf('\nInput root:\n%s\n', INPUT_ROOT);
fprintf('\nOutput root:\n%s\n', OUTPUT_ROOT);

% =========================================================================
% VERIFY SPM
% =========================================================================

if ~isfile(TPM_FILE)
    error('SPM TPM file not found: %s', TPM_FILE);
end

% =========================================================================
% CREATE OUTPUT ROOT
% =========================================================================

if ~isfolder(OUTPUT_ROOT)
    mkdir(OUTPUT_ROOT);
end

successful = 0;
failed = 0;

% =========================================================================
% PROCESS SUBJECTS
% =========================================================================

for i = 1:numel(subjects)

    subject = subjects{i};

    fprintf('\n');
    fprintf('--------------------------------------------------------------------\n');
    fprintf('SUBJECT %d/%d: %s\n', i, numel(subjects), subject);
    fprintf('--------------------------------------------------------------------\n');

    % ---------------------------------------------------------------------
    % INPUT
    % ---------------------------------------------------------------------

    source_file = fullfile( ...
        INPUT_ROOT, ...
        subject, ...
        'anat', ...
        [subject '_T1w_n4_normalized.nii.gz']);

    if ~isfile(source_file)

        fprintf('\nERROR: Input file not found:\n%s\n', source_file);

        failed = failed + 1;
        continue;

    end

    fprintf('\nInput:\n%s\n', source_file);

    % ---------------------------------------------------------------------
    % OUTPUT DIRECTORY
    % ---------------------------------------------------------------------

    subject_dir = fullfile(OUTPUT_ROOT, subject);

    if ~isfolder(subject_dir)
        mkdir(subject_dir);
    end

    % ---------------------------------------------------------------------
    % COPY INPUT
    % ---------------------------------------------------------------------

    working_file = fullfile( ...
        subject_dir, ...
        [subject '_T1w_n4_normalized.nii.gz']);

    if ~isfile(working_file)

        fprintf('\nCopying input into VBM working directory...\n');

        [ok,msg] = copyfile(source_file, working_file);

        if ~ok

            fprintf('\nERROR copying input:\n%s\n', msg);

            failed = failed + 1;
            continue;

        end

    else

        fprintf('\nExisting working input found:\n%s\n', working_file);

    end

    % ---------------------------------------------------------------------
    % UNCOMPRESS
    % ---------------------------------------------------------------------

    working_nii = fullfile( ...
        subject_dir, ...
        [subject '_T1w_n4_normalized.nii']);

    if ~isfile(working_nii)

        fprintf('\nUncompressing input for SPM...\n');

        try
            gunzip(working_file, subject_dir);
        catch ME

            fprintf('\nERROR during gunzip:\n%s\n', ME.message);

            failed = failed + 1;
            continue;

        end

    else

        fprintf('\nExisting uncompressed input found:\n%s\n', working_nii);

    end

    if ~isfile(working_nii)

        fprintf('\nERROR: Uncompressed NIfTI was not created.\n');

        failed = failed + 1;
        continue;

    end

    % ---------------------------------------------------------------------
    % INPUT INFORMATION
    % ---------------------------------------------------------------------

    try

        V = spm_vol([working_nii ',1']);

    catch ME

        fprintf('\nERROR reading input with SPM:\n%s\n', ME.message);

        failed = failed + 1;
        continue;

    end

    fprintf('\nInput dimensions:\n');
    fprintf('  %d x %d x %d\n', ...
        V.dim(1), V.dim(2), V.dim(3));

    fprintf('\nInput affine:\n');
    disp(V.mat);

    % ---------------------------------------------------------------------
    % EXPECTED OUTPUTS
    % ---------------------------------------------------------------------

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

    % ---------------------------------------------------------------------
    % SPM NEW SEGMENT
    % ---------------------------------------------------------------------

    if isfile(modulated_gm)

        fprintf('\nExisting mwc1 found:\n%s\n', modulated_gm);
        fprintf('Skipping segmentation.\n');

    else

        fprintf('\nStarting SPM New Segment...\n');

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

        % RUN
        try

            spm_jobman('initcfg');
            spm_jobman('run', matlabbatch);

        catch ME

            fprintf('\nERROR during SPM segmentation for %s\n', subject);
            fprintf('%s\n', ME.message);

            failed = failed + 1;
            continue;

        end

    end

    % ---------------------------------------------------------------------
    % VERIFY MWC1
    % ---------------------------------------------------------------------

    fprintf('\nChecking SPM outputs...\n');

    fprintf('  c1   : ');
    if isfile(native_gm)
        fprintf('FOUND\n');
    else
        fprintf('MISSING\n');
    end

    fprintf('  wc1  : ');
    if isfile(warped_gm)
        fprintf('FOUND\n');
    else
        fprintf('MISSING\n');
    end

    fprintf('  mwc1 : ');
    if isfile(modulated_gm)
        fprintf('FOUND\n');
    else
        fprintf('MISSING\n');
    end

    if ~isfile(modulated_gm)

        fprintf('\nSTATUS: FAIL - mwc1 missing\n');

        failed = failed + 1;
        continue;

    end

    % ---------------------------------------------------------------------
    % MWC1 QC
    % ---------------------------------------------------------------------

    try

        M = spm_vol([modulated_gm ',1']);
        Y = spm_read_vols(M);

    catch ME

        fprintf('\nERROR reading mwc1:\n%s\n', ME.message);

        failed = failed + 1;
        continue;

    end

    finite_count = sum(isfinite(Y(:)));
    total_count = numel(Y);
    nonzero_count = sum(Y(:) ~= 0);

    fprintf('\nmwc1 dimensions:\n');
    fprintf('  %d x %d x %d\n', ...
        M.dim(1), M.dim(2), M.dim(3));

    fprintf('\nFinite voxels:\n');
    fprintf('  %d / %d\n', finite_count, total_count);

    fprintf('\nNon-zero voxels:\n');
    fprintf('  %d\n', nonzero_count);

    fprintf('\nMinimum:\n');
    fprintf('  %.8g\n', min(Y(:)));

    fprintf('\nMaximum:\n');
    fprintf('  %.8g\n', max(Y(:)));

    fprintf('\nMean:\n');
    fprintf('  %.8g\n', mean(Y(:)));

    if finite_count ~= total_count

        fprintf('\nERROR: mwc1 contains non-finite values.\n');

        failed = failed + 1;
        continue;

    end

    if nonzero_count == 0

        fprintf('\nERROR: mwc1 is completely zero.\n');

        failed = failed + 1;
        continue;

    end

    % ---------------------------------------------------------------------
    % 8-MM SMOOTHING
    % ---------------------------------------------------------------------

    smooth_dir = fullfile(subject_dir, 'smooth');

    if ~isfolder(smooth_dir)
        mkdir(smooth_dir);
    end

    smooth_input = fullfile( ...
        smooth_dir, ...
        ['mwc1' subject '_T1w_n4_normalized.nii']);

    smooth_gm = fullfile( ...
        smooth_dir, ...
        ['s8mwc1' subject '_T1w_n4_normalized.nii']);

    if ~isfile(smooth_gm)

        fprintf('\nStarting 8-mm smoothing...\n');

        if ~isfile(smooth_input)

            copyfile(modulated_gm, smooth_input);

        end

        clear matlabbatch;

        matlabbatch{1}.spm.spatial.smooth.data = {
            [smooth_input ',1']
        };

        matlabbatch{1}.spm.spatial.smooth.fwhm = [8 8 8];
        matlabbatch{1}.spm.spatial.smooth.dtype = 0;
        matlabbatch{1}.spm.spatial.smooth.im = 0;
        matlabbatch{1}.spm.spatial.smooth.prefix = 's8';

        try

            spm_jobman('run', matlabbatch);

        catch ME

            fprintf('\nERROR during smoothing:\n%s\n', ME.message);

            failed = failed + 1;
            continue;

        end

    else

        fprintf('\nExisting smoothed mwc1 found:\n%s\n', smooth_gm);

    end

    % ---------------------------------------------------------------------
    % VERIFY SMOOTHED OUTPUT
    % ---------------------------------------------------------------------

    if ~isfile(smooth_gm)

        fprintf('\nERROR: Smoothed mwc1 was not created.\n');

        failed = failed + 1;
        continue;

    end

    try

        S = spm_vol([smooth_gm ',1']);
        Ys = spm_read_vols(S);

    catch ME

        fprintf('\nERROR reading smoothed mwc1:\n%s\n', ME.message);

        failed = failed + 1;
        continue;

    end

    if any(~isfinite(Ys(:)))

        fprintf('\nERROR: Smoothed mwc1 contains non-finite values.\n');

        failed = failed + 1;
        continue;

    end

    if all(Ys(:) == 0)

        fprintf('\nERROR: Smoothed mwc1 is completely zero.\n');

        failed = failed + 1;
        continue;

    end

    % ---------------------------------------------------------------------
    % QC REPORT
    % ---------------------------------------------------------------------

    qc_file = fullfile( ...
        subject_dir, ...
        [subject '_VBM_QC.txt']);

    fid = fopen(qc_file,'w');

    fprintf(fid, 'NeuroFusion AI - External ds004302 VBM QC\n\n');
    fprintf(fid, 'Subject: %s\n\n', subject);

    fprintf(fid, 'Input:\n%s\n\n', working_nii);
    fprintf(fid, 'Native GM:\n%s\n\n', native_gm);
    fprintf(fid, 'Warped GM:\n%s\n\n', warped_gm);
    fprintf(fid, 'Modulated GM:\n%s\n\n', modulated_gm);
    fprintf(fid, 'Forward deformation:\n%s\n\n', deformation);
    fprintf(fid, 'Smoothed modulated GM:\n%s\n\n', smooth_gm);

    fprintf(fid, 'Modulated dimensions: [%s]\n', num2str(M.dim));
    fprintf(fid, 'Modulated finite voxels: %d / %d\n', ...
        finite_count, total_count);
    fprintf(fid, 'Modulated non-zero voxels: %d\n', nonzero_count);
    fprintf(fid, 'Modulated minimum: %.8g\n', min(Y(:)));
    fprintf(fid, 'Modulated maximum: %.8g\n', max(Y(:)));
    fprintf(fid, 'Modulated mean: %.8g\n', mean(Y(:)));

    fprintf(fid, '\n');

    fprintf(fid, 'Smoothed dimensions: [%s]\n', num2str(S.dim));
    fprintf(fid, 'Smoothed finite voxels: %d / %d\n', ...
        sum(isfinite(Ys(:))), numel(Ys));
    fprintf(fid, 'Smoothed minimum: %.8g\n', min(Ys(:)));
    fprintf(fid, 'Smoothed maximum: %.8g\n', max(Ys(:)));
    fprintf(fid, 'Smoothed mean: %.8g\n', mean(Ys(:)));

    fclose(fid);

    % ---------------------------------------------------------------------
    % SUCCESS
    % ---------------------------------------------------------------------

    fprintf('\nSTATUS: PASS\n');
    fprintf('\nmwc1:\n%s\n', modulated_gm);
    fprintf('\nSmoothed mwc1:\n%s\n', smooth_gm);
    fprintf('\nQC:\n%s\n', qc_file);

    successful = successful + 1;

end

% =========================================================================
% FINAL SUMMARY
% =========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('EXTERNAL ds004302 VBM SUMMARY\n');
fprintf('====================================================================\n');

fprintf('\nSubjects found : %d\n', numel(subjects));
fprintf('Successful     : %d\n', successful);
fprintf('Failed         : %d\n', failed);

if failed == 0
    fprintf('\nSTATUS: EXTERNAL VBM COMPLETED SUCCESSFULLY\n');
else
    fprintf('\nSTATUS: EXTERNAL VBM COMPLETED WITH FAILURES\n');
end

fprintf('\nOutput root:\n%s\n', OUTPUT_ROOT);

fprintf('\n');
fprintf('====================================================================\n');