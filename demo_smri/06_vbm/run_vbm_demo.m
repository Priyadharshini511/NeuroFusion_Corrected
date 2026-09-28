% ================================================================
% NeuroFusion AI - sMRI DEMO VBM
%
% DEMO ONLY - ONE SUBJECT
% Subject: A00024372
%
% Pipeline:
% N4 RMS MPRAGE
%       ->
% SPM25 tissue segmentation
%       ->
% native GM (c1)
%       ->
% warped + Jacobian-modulated GM (mwc1)
%       ->
% 8 mm smoothing
%
% IMPORTANT:
% This script does NOT use the 119-subject VBM worker.
% This script does NOT modify files under src/.
% ================================================================

clear;
clc;

fprintf('\n');
fprintf('======================================================================\n');
fprintf('NeuroFusion sMRI DEMO - VBM / SPM25\n');
fprintf('======================================================================\n');

% ----------------------------------------------------------------
% PROJECT PATHS
% ----------------------------------------------------------------

project_root = 'C:\Users\User\Documents\neurofusion';

input_gz = fullfile( ...
    project_root, ...
    'processed_data', ...
    'smri', ...
    'demo_single_subject', ...
    '04_n4', ...
    'A00024372_sMRI_n4.nii.gz');

vbm_root = fullfile( ...
    project_root, ...
    'processed_data', ...
    'smri', ...
    'demo_single_subject', ...
    '06_vbm');

work_dir = vbm_root;

if ~exist(work_dir,'dir')
    mkdir(work_dir);
end

% ----------------------------------------------------------------
% VERIFY INPUT
% ----------------------------------------------------------------

fprintf('\nInput:\n%s\n', input_gz);
fprintf('\nOutput directory:\n%s\n', work_dir);

if ~exist(input_gz,'file')
    error('Demo N4 input not found:\n%s', input_gz);
end

% ----------------------------------------------------------------
% UNCOMPRESS N4 IMAGE INTO DEMO VBM DIRECTORY
%
% Original N4 file remains untouched.
% ----------------------------------------------------------------

input_nii = fullfile( ...
    work_dir, ...
    'A00024372_sMRI_n4.nii');

if ~exist(input_nii,'file')

    fprintf('\nUncompressing demo N4 image...\n');

    gunzip(input_gz,work_dir);

end

if ~exist(input_nii,'file')
    error('Uncompressed N4 image was not created.');
end

fprintf('\nUncompressed input:\n%s\n',input_nii);

% ----------------------------------------------------------------
% SPM INITIALIZATION
% ----------------------------------------------------------------

fprintf('\nInitializing SPM25...\n');

spm('Defaults','fmri');
spm_jobman('initcfg');

fprintf('SPM initialization PASS\n');

% ----------------------------------------------------------------
% TPM
% ----------------------------------------------------------------

tpm = fullfile( ...
    spm('Dir'), ...
    'tpm', ...
    'TPM.nii');

if ~exist(tpm,'file')
    error('SPM TPM file not found:\n%s',tpm);
end

fprintf('\nSPM TPM:\n%s\n',tpm);

% ----------------------------------------------------------------
% SPM TISSUE SEGMENTATION
% ----------------------------------------------------------------

fprintf('\n');
fprintf('======================================================================\n');
fprintf('Starting SPM25 tissue segmentation\n');
fprintf('Subject: A00024372\n');
fprintf('======================================================================\n');

clear matlabbatch;

matlabbatch{1}.spm.spatial.preproc.channel.vols = ...
    {[input_nii ',1']};

matlabbatch{1}.spm.spatial.preproc.channel.biasreg = ...
    0.001;

matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = ...
    60;

matlabbatch{1}.spm.spatial.preproc.channel.write = ...
    [0 1];

% Tissue Gaussian components
ngaus = [1 1 2 3 4 2];

for k = 1:6

    matlabbatch{1}.spm.spatial.preproc.tissue(k).tpm = ...
        {[tpm ',' num2str(k)]};

    matlabbatch{1}.spm.spatial.preproc.tissue(k).ngaus = ...
        ngaus(k);

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

% ----------------------------------------------------------------
% NONLINEAR DEFORMATION SETTINGS
% ----------------------------------------------------------------

matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;

matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;

matlabbatch{1}.spm.spatial.preproc.warp.reg = ...
    [0 0.001 0.5 0.05 0.2];

matlabbatch{1}.spm.spatial.preproc.warp.affreg = ...
    'mni';

matlabbatch{1}.spm.spatial.preproc.warp.fwhm = ...
    0;

matlabbatch{1}.spm.spatial.preproc.warp.samp = ...
    3;

% Write deformation fields
matlabbatch{1}.spm.spatial.preproc.warp.write = ...
    [1 1];

% ----------------------------------------------------------------
% RUN SEGMENTATION
% ----------------------------------------------------------------

fprintf('\nRunning SPM25 segmentation...\n');

spm_jobman('run',matlabbatch);

fprintf('\nSPM25 segmentation completed.\n');

% ----------------------------------------------------------------
% LOCATE GM OUTPUTS
% ----------------------------------------------------------------

native_gm = fullfile( ...
    work_dir, ...
    'c1A00024372_sMRI_n4.nii');

warped_gm = fullfile( ...
    work_dir, ...
    'wc1A00024372_sMRI_n4.nii');

modulated_gm = fullfile( ...
    work_dir, ...
    'mwc1A00024372_sMRI_n4.nii');

fprintf('\nChecking segmentation outputs...\n');

fprintf('Native GM:\n%s\n',native_gm);
fprintf('Warped GM:\n%s\n',warped_gm);
fprintf('Modulated GM:\n%s\n',modulated_gm);

if ~exist(native_gm,'file')
    error('Native GM output was not created.');
end

if ~exist(modulated_gm,'file')
    error('Modulated GM output (mwc1) was not created.');
end

fprintf('\nSegmentation outputs PASS\n');

% ----------------------------------------------------------------
% 8 MM SMOOTHING OF MODULATED GM
% ----------------------------------------------------------------

fprintf('\n');
fprintf('======================================================================\n');
fprintf('Starting 8 mm smoothing\n');
fprintf('Input: mwc1 GM\n');
fprintf('======================================================================\n');

clear matlabbatch;

matlabbatch{1}.spm.spatial.smooth.data = ...
    {[modulated_gm ',1']};

matlabbatch{1}.spm.spatial.smooth.fwhm = ...
    [8 8 8];

matlabbatch{1}.spm.spatial.smooth.dtype = ...
    0;

matlabbatch{1}.spm.spatial.smooth.im = ...
    0;

matlabbatch{1}.spm.spatial.smooth.prefix = ...
    's8_';

spm_jobman('run',matlabbatch);

% ----------------------------------------------------------------
% VERIFY SMOOTHED OUTPUT
% ----------------------------------------------------------------

smoothed_gm = fullfile( ...
    work_dir, ...
    's8_mwc1A00024372_sMRI_n4.nii');

fprintf('\nChecking smoothed VBM image...\n');
fprintf('Expected output:\n%s\n',smoothed_gm);

if ~exist(smoothed_gm,'file')
    error('8 mm smoothed VBM image was not created.');
end

% ----------------------------------------------------------------
% FINAL STATUS
% ----------------------------------------------------------------

fprintf('\n');
fprintf('======================================================================\n');
fprintf('STATUS: DEMO VBM PASS\n');
fprintf('======================================================================\n');

fprintf('\nOnly this demo subject was processed:\n');
fprintf('  A00024372\n');

fprintf('\nVBM outputs:\n');
fprintf('  Native GM:\n  %s\n',native_gm);
fprintf('\n  Modulated GM:\n  %s\n',modulated_gm);
fprintf('\n  8 mm smoothed GM:\n  %s\n',smoothed_gm);

fprintf('\nNo 119-subject batch processing was performed.\n');
fprintf('No files under src/ were modified.\n');

fprintf('\n======================================================================\n');

