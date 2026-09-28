from pathlib import Path
import csv
import math
import numpy as np
import SimpleITK as sitk

ROOT = Path('processed_data') / 'smri'
INPUT_DIR = ROOT / 'legacy_normalized'
MNI_PATH = ROOT / 'mni' / 'mni152_template_2mm.nii.gz'
OUTPUT_DIR = ROOT / 'legacy_registered'
TRANSFORM_DIR = ROOT / 'legacy_transforms'
REPORT_FILE = OUTPUT_DIR / 'legacy_registration_report.csv'

LEGACY_SUBJECTS = '''A00000300 A00002198 A00003150 A00004087 A00007409 A00009946 A00010150 A00010684 A00011265 A00011725 A00012995 A00013140 A00013363 A00013816 A00014120 A00014225 A00014522 A00014839 A00014898 A00015759 A00015826 A00017294 A00022729 A00022773 A00022810 A00022835 A00022837 A00022915 A00023095 A00023120 A00023131 A00023143 A00023330 A00023337 A00023730 A00023800 A00023848 A00023866 A00024160 A00024301 A00024372 A00024446 A00024535 A00027487 A00028406 A00028409 A00029226 A00029452 A00031478 A00031764 A00033214 A00037564 A00037665'''.split()

ATOL = 1e-6
TOL = 1e-5

def geometry_ok(a, b):
    return (a.GetDimension() == b.GetDimension() and a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=ATOL, rtol=0)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=ATOL, rtol=0)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=ATOL, rtol=0))

def valid_output(p, fixed):
    if not p.exists(): return False, 'missing_output'
    try: im = sitk.ReadImage(str(p))
    except Exception as e: return False, f'read_error:{type(e).__name__}'
    if not geometry_ok(im, fixed): return False, 'grid_mismatch'
    x = sitk.GetArrayViewFromImage(im)
    if x.size == 0 or not np.isfinite(x).all(): return False, 'empty_or_nonfinite'
    if float(x.min()) < -TOL or float(x.max()) > 1 + TOL: return False, 'intensity_out_of_range'
    return True, 'valid'

def valid_transform(p):
    if not p.exists(): return False, 'missing_transform'
    try: t = sitk.ReadTransform(str(p))
    except Exception as e: return False, f'read_error:{type(e).__name__}'
    if t.GetDimension() != 3 or t.GetName() != 'AffineTransform': return False, 'not_standalone_affine'
    q = np.asarray(t.GetParameters(), dtype=float)
    if q.size != 12 or not np.isfinite(q).all(): return False, 'invalid_parameters'
    return True, 'valid'

def register(moving, fixed):
    tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.20)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([2, 1, 0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetInitialTransform(tx, inPlace=True)
    final_tx = r.Execute(fixed, moving)
    if final_tx.GetName() != 'AffineTransform':
        raise RuntimeError(f'unexpected transform: {final_tx.GetName()}')
    out = sitk.Resample(moving, fixed, final_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    out.CopyInformation(fixed)
    metric = float(r.GetMetricValue())
    if not math.isfinite(metric): raise RuntimeError('non-finite registration metric')
    return out, final_tx, metric

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)
    if not MNI_PATH.exists(): raise FileNotFoundError(MNI_PATH)
    fixed = sitk.ReadImage(str(MNI_PATH), sitk.sitkFloat32)
    print('=' * 70)
    print('LEGACY sMRI -> MNI REGISTRATION')
    print(f'Expected: {len(LEGACY_SUBJECTS)}')
    print(f'MNI grid: {fixed.GetSize()} spacing={fixed.GetSpacing()}')
    rows=[]; passed=skipped=failed=0
    for i, sid in enumerate(LEGACY_SUBJECTS, 1):
        inp=INPUT_DIR/f'{sid}_legacy_n4_normalized.nii.gz'
        outp=OUTPUT_DIR/f'{sid}_legacy_n4_mni.nii.gz'
        txp=TRANSFORM_DIR/f'{sid}_legacy_to_mni.tfm'
        row={'subject_id':sid,'input':str(inp),'output':str(outp),'transform':str(txp),'status':'','metric_value':'','message':''}
        print(f'[{i}/{len(LEGACY_SUBJECTS)}] {sid}')
        if not inp.exists():
            row.update(status='FAILED', message='missing_input'); failed+=1; rows.append(row); print('  FAILED: missing input'); continue
        vo,mo=valid_output(outp,fixed); vt,mt=valid_transform(txp)
        if vo and vt:
            row.update(status='SKIPPED_VALID', message='existing_output_and_transform_valid'); skipped+=1; rows.append(row); print('  SKIP: existing valid pair'); continue
        if outp.exists(): outp.unlink()
        if txp.exists(): txp.unlink()
        try:
            moving=sitk.ReadImage(str(inp),sitk.sitkFloat32)
            x=sitk.GetArrayViewFromImage(moving)
            if moving.GetDimension()!=3: raise RuntimeError('input_not_3d')
            if x.size==0 or not np.isfinite(x).all(): raise RuntimeError('input_empty_or_nonfinite')
            if float(x.min()) < -TOL or float(x.max()) > 1+TOL: raise RuntimeError('input_not_[0,1]')
            print(f'  Input: {moving.GetSize()} {moving.GetSpacing()}')
            reg,tx,metric=register(moving,fixed)
            y=sitk.GetArrayViewFromImage(reg)
            if not np.isfinite(y).all(): raise RuntimeError('registered_nonfinite')
            if float(y.min()) < -TOL or float(y.max()) > 1+TOL: raise RuntimeError('registered_intensity_out_of_range')
            sitk.WriteImage(reg,str(outp),useCompression=True)
            sitk.WriteTransform(tx,str(txp))
            vo,mo=valid_output(outp,fixed); vt,mt=valid_transform(txp)
            if not vo: raise RuntimeError(f'output_validation:{mo}')
            if not vt: raise RuntimeError(f'transform_validation:{mt}')
            row.update(status='PASS',metric_value=f'{metric:.10g}',message='registered_and_validated')
            passed+=1; print(f'  PASS: metric={metric:.10g}')
        except Exception as e:
            row.update(status='FAILED',message=f'{type(e).__name__}: {e}')
            failed+=1; print(f'  FAILED: {type(e).__name__}: {e}')
        rows.append(row)
    with REPORT_FILE.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['subject_id','input','output','transform','status','metric_value','message'])
        w.writeheader(); w.writerows(rows)
    print('\n'+'='*70)
    print(f'Expected: {len(LEGACY_SUBJECTS)} | Passed: {passed} | Skipped: {skipped} | Failed: {failed}')
    print(f'Report: {REPORT_FILE}')
    print('FINAL STATUS: PASS' if failed==0 and passed+skipped==len(LEGACY_SUBJECTS) else 'FINAL STATUS: REVIEW REQUIRED')

if __name__ == '__main__': main()
