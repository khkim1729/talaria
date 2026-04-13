"""
MSD Liver 전체 케이스에 TotalSegmentator --ml 적용.

Usage:
    python run_totalseg_msd.py
"""

import os
import subprocess
import sys

IMG_DIR    = '/home/rintern10/talaria/datasets/MSD_Liver/imagesTr'
OUTPUT_DIR = '/home/rintern10/TotalSegmentator/msd_ml'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cases = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.nii.gz')])
    print(f'>>> 총 {len(cases)}개 케이스 발견\n')

    for i, fname in enumerate(cases):
        case_id     = fname.replace('.nii.gz', '')
        input_path  = os.path.join(IMG_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, f'{case_id}_ml.nii.gz')

        # 이미 완료된 케이스 스킵
        if os.path.exists(output_path):
            print(f'[{i+1}/{len(cases)}] {case_id} — 이미 존재, 스킵')
            continue

        print(f'[{i+1}/{len(cases)}] {case_id} 처리 중...')
        print(f'    input:  {input_path}')
        print(f'    output: {output_path}')

        cmd = ['TotalSegmentator', '-i', input_path, '-o', output_path, '--ml']
        ret = subprocess.run(cmd, capture_output=True, text=True)

        if ret.returncode == 0:
            print(f'    ✅ 완료')
        else:
            print(f'    ❌ 실패')
            print(f'    stderr: {ret.stderr[:300]}')

    print(f'\n>>> 전체 완료. 결과 디렉토리: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()