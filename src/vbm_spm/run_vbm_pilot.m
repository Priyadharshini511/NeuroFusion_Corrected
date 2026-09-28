% ================================================================
% NeuroFusion AI - SPM25 VBM Pilot
% Subject: A00000368
%
% Pipeline:
% N4 RMS MPRAGE
%       ->
% SPM Segmentation
%       ->
% native GM (c1)
%       ->
% warped GM + Jacobian modulation (mwc1)
%       ->
% 8 mm smoothing
%
% ================================================================

clear;
spm('Defaults','fmri');
spm_jobman('initcfg');

project_root = 'C:\Users\User\Documents\neurofusion';

input_gz = fullfile( ...
    project_root, ...
    'processed_data', ...
    'smri', ...
    'n4', ...
    'A00000368_rms_n4.nii.gz');

vbm_root = fullfile( ...
    project_root, ...
    'processed_data', ...
    'smri', ...
    'vbm_spm');

work_dir = fullfile(vbm_root,'A00000368');

if ~exist(work_dir,'dir')
    mkdir(work_dir);
end

if ~exist(input_gz,'file')
    error('Input file not found: %s',input_gz);
end

% ---------------------------------------------------------------
% Uncompress N4 image into the NEW VBM directory.
% Original N4 file remains untouched.
% ---------------------------------------------------------------

gunzip(input_gz,work_dir);

input_nii = fullfile( ...
    work_dir, ...
    'A00000368_rms_n4.nii');

if ~exist(input_nii,'file')
    error('Uncompressed input was not created.');
end

% ---------------------------------------------------------------
% SPM initialization
% ---------------------------------------------------------------

spm('Defaults','fmri');
spm_jobman('initcfg');

% ---------------------------------------------------------------
% SPM tissue segmentation
%
% native  = native-space GM/WM/CSF
% warped  = normalized tissue classes
%
% warp.write = [1 1]
%   inverse deformation + forward deformation
% ---------------------------------------------------------------

tpm = fullfile( ...
    spm('Dir'), ...
    'tpm', ...
    'TPM.nii');

clear matlabbatch;

matlabbatch{1}.spm.spatial.preproc.channel.vols = ...
    {[input_nii ',1']};

matlabbatch{1}.spm.spatial.preproc.channel.biasreg = ...
    0.001;

matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = ...
    60;

matlabbatch{1}.spm.spatial.preproc.channel.write = ...
    [0 1];


% ---------------------------------------------------------------
% Tissue classes
% ---------------------------------------------------------------

ngauss = [1 1 2 3 4 2];

for k = 1:6

    matlabbatch{1}.spm.spatial.preproc.tissue(k).tpm = ...
        {[tpm ',' num2str(k)]};

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

    % Request warped tissue classes for GM/WM/CSF.
    if k <= 3
        matlabbatch{1}.spm.spatial.preproc.tissue(k).warped = ...
            [1 1];
    else
        matlabbatch{1}.spm.spatial.preproc.tissue(k).warped = ...
            [0 0];
    end

end


% ---------------------------------------------------------------
% Nonlinear deformation settings
% ---------------------------------------------------------------

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


% ---------------------------------------------------------------
% Run segmentation
% ---------------------------------------------------------------

fprintf('\n');
fprintf('=============================================\n');
fprintf('Starting SPM25 VBM segmentation\n');
fprintf('Subject: A00000368\n');
fprintf('=============================================\n\n');

spm_jobman('run',matlabbatch);


% ---------------------------------------------------------------
% Locate generated GM outputs
% ---------------------------------------------------------------

native_gm = fullfile( ...
    work_dir, ...
    'c1A00000368_rms_n4.nii');

warped_gm = fullfile( ...
    work_dir, ...
    'wc1A00000368_rms_n4.nii');

modulated_gm = fullfile( ...
    work_dir, ...
    'mwc1A00000368_rms_n4.nii');

deformation = fullfile( ...
    work_dir, ...
    'y_A00000368_rms_n4.nii');


% ---------------------------------------------------------------
% Verify outputs
% ---------------------------------------------------------------

if ~exist(native_gm,'file')
    error('Native GM output was not created.');
end

if ~exist(warped_gm,'file')
    error('Warped GM output was not created.');
end

if ~exist(modulated_gm,'file')
    error('Modulated warped GM output was not created.');
end

if ~exist(deformation,'file')
    error('Forward deformation field was not created.');
end


fprintf('\nSPM segmentation completed successfully.\n');


% ---------------------------------------------------------------
% 8-mm smoothing of modulated GM
% ---------------------------------------------------------------

smooth_dir = fullfile(work_dir,'smooth');

if ~exist(smooth_dir,'dir')
    mkdir(smooth_dir);
end

% Put the modulated GM into the smooth directory.
smooth_input = fullfile( ...
    smooth_dir, ...
    'mwc1A00000368_rms_n4.nii');

copyfile(modulated_gm,smooth_input);


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


fprintf('\n');
fprintf('Starting 8-mm smoothing...\n');

spm_jobman('run',matlabbatch);


smooth_gm = fullfile( ...
    smooth_dir, ...
    's8mwc1A00000368_rms_n4.nii');


if ~exist(smooth_gm,'file')
    error('Smoothed modulated GM was not created.');
end


% ---------------------------------------------------------------
% Basic numerical QC
% ---------------------------------------------------------------

V = spm_vol(modulated_gm);
Y = spm_read_vols(V);

Vs = spm_vol(smooth_gm);
Ys = spm_read_vols(Vs);

qc_file = fullfile( ...
    work_dir, ...
    'A00000368_VBM_QC.txt');

fid = fopen(qc_file,'w');

fprintf(fid,'NeuroFusion AI - SPM25 VBM Pilot QC\n\n');

fprintf(fid,'Subject: A00000368\n\n');

fprintf(fid,'Input:\n%s\n\n',input_nii);

fprintf(fid,'Native GM:\n%s\n\n',native_gm);

fprintf(fid,'Warped GM:\n%s\n\n',warped_gm);

fprintf(fid,'Modulated GM:\n%s\n\n',modulated_gm);

fprintf(fid,'Forward deformation:\n%s\n\n',deformation);

fprintf(fid,'Smoothed modulated GM:\n%s\n\n',smooth_gm);

fprintf(fid,'Modulated dimensions: [%s]\n', ...
    num2str(V.dim));

fprintf(fid,'Modulated finite voxels: %d / %d\n', ...
    sum(isfinite(Y(:))),numel(Y));

fprintf(fid,'Modulated minimum: %.8g\n', ...
    min(Y(:)));

fprintf(fid,'Modulated maximum: %.8g\n', ...
    max(Y(:)));

fprintf(fid,'Modulated mean: %.8g\n', ...
    mean(Y(:)));

fprintf(fid,'\n');

fprintf(fid,'Smoothed dimensions: [%s]\n', ...
    num2str(Vs.dim));

fprintf(fid,'Smoothed finite voxels: %d / %d\n', ...
    sum(isfinite(Ys(:))),numel(Ys));

fprintf(fid,'Smoothed minimum: %.8g\n', ...
    min(Ys(:)));

fprintf(fid,'Smoothed maximum: %.8g\n', ...
    max(Ys(:)));

fprintf(fid,'Smoothed mean: %.8g\n', ...
    mean(Ys(:)));

fclose(fid);


% ---------------------------------------------------------------
% Final report
% ---------------------------------------------------------------

fprintf('\n');
fprintf('====================================================\n');
fprintf('NEUROFUSION SPM25 VBM PILOT COMPLETED SUCCESSFULLY\n');
fprintf('====================================================\n');

fprintf('Subject:\nA00000368\n\n');

fprintf('Native GM:\n%s\n\n',native_gm);

fprintf('Warped GM:\n%s\n\n',warped_gm);

fprintf('MODULATED GM:\n%s\n\n',modulated_gm);

fprintf('SMOOTHED MODULATED GM:\n%s\n\n',smooth_gm);

fprintf('QC:\n%s\n',qc_file);

fprintf('====================================================\n');
