"""
MSD Liver 10개 케이스에 TotalSegmentator total 태스크 적용.
liver, tumor Dice score 계산 후 텍스트 파일 저장.

total 태스크 결과:
    liver.nii.gz → GT label==1 (liver)와 Dice 비교
    liver_tumor.nii.gz 없음 → liver_vessels 태스크로 tumor Dice 비교
"""

import os
import subprocess
import numpy as np
import nibabel as nib

IMG_DIR    = '/home/rintern10/talaria/datasets/MSD_Liver/imagesTr'
LBL_DIR    = '/home/rintern10/talaria/datasets/MSD_Liver/labelsTr'
OUTPUT_DIR = '/home/rintern10/TotalSegmentator/msd_eval'
RESULT_TXT = '/home/rintern10/TotalSegmentator/msd_eval_results.txt'
N_CASES    = 10


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float('nan')
    return round(float(2 * intersection / denom), 4)


def run_totalseg(input_path, output_dir, task=None):
    os.makedirs(output_dir, exist_ok=True)
    cmd = ['TotalSegmentator', '-i', input_path, '-o', output_dir]
    if task is None:
        cmd += ['--fast']
    else:
        cmd += ['-ta', task]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    return ret.returncode == 0


def main():
    cases = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.nii.gz')])[:N_CASES]
    results = []

    for fname in cases:
        case_id  = fname.replace('.nii.gz', '')
        img_path = os.path.join(IMG_DIR, fname)
        lbl_path = os.path.join(LBL_DIR, fname)

        if not os.path.exists(lbl_path):
            print(f'[{case_id}] GT 없음, 스킵')
            continue

        print(f'\n[{case_id}] 처리 중...')

        # GT 로드
        gt       = nib.load(lbl_path).get_fdata()
        gt_liver = (gt == 1).astype(np.uint8)
        gt_tumor = (gt == 2).astype(np.uint8)

        result = {
            'case':            case_id,
            'gt_liver_voxels': int(gt_liver.sum()),
            'gt_tumor_voxels': int(gt_tumor.sum()),
        }

        # 1. total 태스크 → liver Dice
        total_dir  = os.path.join(OUTPUT_DIR, case_id, 'total')
        liver_path = os.path.join(total_dir, 'liver.nii.gz')

        if not os.path.exists(liver_path):
            print(f'  [total] 실행 중...')
            run_totalseg(img_path, total_dir, task=None)

        if os.path.exists(liver_path):
            pred_liver = nib.load(liver_path).get_fdata().astype(np.uint8)
            result['pred_liver_voxels'] = int(pred_liver.sum())
            result['liver_dice']        = dice_score(pred_liver, gt_liver)
            print(f'  liver Dice: {result["liver_dice"]}')
        else:
            result['pred_liver_voxels'] = -1
            result['liver_dice']        = 'FAIL'

        # 2. liver_vessels 태스크 → tumor Dice
        vessels_dir = os.path.join(OUTPUT_DIR, case_id, 'liver_vessels')
        tumor_path  = os.path.join(vessels_dir, 'liver_tumor.nii.gz')

        if not os.path.exists(tumor_path):
            print(f'  [liver_vessels] 실행 중...')
            run_totalseg(img_path, vessels_dir, task='liver_vessels')

        if os.path.exists(tumor_path):
            pred_tumor = nib.load(tumor_path).get_fdata().astype(np.uint8)
            result['pred_tumor_voxels'] = int(pred_tumor.sum())
            result['tumor_dice']        = dice_score(pred_tumor, gt_tumor) if gt_tumor.sum() > 0 else 'no_gt'
            print(f'  tumor Dice: {result["tumor_dice"]}')
        else:
            result['pred_tumor_voxels'] = -1
            result['tumor_dice']        = 'FAIL'

        results.append(result)

    # 결과 저장
    os.makedirs(os.path.dirname(RESULT_TXT), exist_ok=True)
    with open(RESULT_TXT, 'w') as f:
        f.write('=== MSD Liver TotalSegmentator 평가 결과 ===\n')
        f.write(f'케이스 수: {len(results)}\n\n')
        f.write(f'{"Case":<12} {"GT_liver":>10} {"Pred_liver":>10} {"Liver_Dice":>10} '
                f'{"GT_tumor":>10} {"Pred_tumor":>10} {"Tumor_Dice":>10}\n')
        f.write('-' * 80 + '\n')

        liver_dices = []
        tumor_dices = []
        for r in results:
            f.write(f'{r["case"]:<12} '
                    f'{r["gt_liver_voxels"]:>10} '
                    f'{str(r.get("pred_liver_voxels","N/A")):>10} '
                    f'{str(r.get("liver_dice","N/A")):>10} '
                    f'{r["gt_tumor_voxels"]:>10} '
                    f'{str(r.get("pred_tumor_voxels","N/A")):>10} '
                    f'{str(r.get("tumor_dice","N/A")):>10}\n')
            if isinstance(r.get('liver_dice'), float):
                liver_dices.append(r['liver_dice'])
            if isinstance(r.get('tumor_dice'), float):
                tumor_dices.append(r['tumor_dice'])

        f.write('-' * 80 + '\n')
        if liver_dices:
            f.write(f'평균 Liver Dice: {round(sum(liver_dices)/len(liver_dices), 4)}\n')
        if tumor_dices:
            f.write(f'평균 Tumor Dice: {round(sum(tumor_dices)/len(tumor_dices), 4)}\n')

    print(f'\n>>> 결과 저장: {RESULT_TXT}')
    with open(RESULT_TXT) as f:
        print(f.read())


if __name__ == '__main__':
    main()