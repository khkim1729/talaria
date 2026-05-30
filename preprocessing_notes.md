# HECKTOR 2025 Preprocessing Pipeline Summary
### Multi-stage Multimodal Progressive Learning for Coordinated Segmentation, Diagnosis, and Prognosis in Head and Neck Cancer

## 1. 전처리 파이프라인 흐름
1. **Geometry Audit**: CT와 PET의 Origin, Spacing, Direction 일치 여부 검사
2. **Resampling**: 교차 영역 기준, B-spline 보간법을 이용해 1.0mm Isotropic 공간으로 통합
3. **Intensity Scaling**:
   - CT: [-1024, 1024] HU로 자른 뒤 [-1, 1]로 정규화
   - PET: Non-zero 복셀 기준 전체 z-score 정규화
4. **ROI Center Detection**: Cranial 25% 영역 내 PET 상위 복셀 3D Connected Component 중심점 탐색
5. **Crop**: 
   - Inference Target Patch: 200 x 200 x 310
   - Training Patch: Random 128^3 sub-patch

## 2. 다음 파이프라인(Segmentation Evaluation)과의 연결 규격
- 최종 출력 포맷: NIfTI (.nii.gz)
- 예측 모델 인풋 shape: (Batch, 2, 128, 128, 128) -> [CT channel, PET channel]
- 레이블 shape: (Batch, 2, 128, 128, 128) -> [GTVp mask, GTVn mask]
