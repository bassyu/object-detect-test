import * as tf from '@tensorflow/tfjs-node-gpu';

export interface DetectedObject {
  bbox: [number, number, number, number]; // [x, y, width, height]
  label: string;
  score: number;
}

interface ImageSlice {
  imageTensor: tf.Tensor3D; // tensor 슬라이스
  offsetX: number; // 전체 이미지에서의 x 좌표
  offsetY: number; // 전체 이미지에서의 y 좌표
}

export class SlicingModel {
  private sliceSize: number;
  private baseModel: any;
  private overlapRatio: number;
  private nmsThreshold: number;

  constructor(parms: {
    sliceSize: number;
    baseModel: any;
    overlapRatio?: number;
    nmsThreshold?: number;
  }) {
    this.sliceSize = parms.sliceSize;
    this.baseModel = parms.baseModel;
    this.overlapRatio = parms.overlapRatio ?? 0.2;
    this.nmsThreshold = parms.nmsThreshold ?? 0.5;
  }

  async detect(imageTensor: tf.Tensor3D): Promise<DetectedObject[]> {
    // 1. 이미지 슬라이싱
    const slices = await this.sliceImage(imageTensor);
    console.log('length: ', slices.length);

    // 2. 각 슬라이스에 모델 적용
    const allDetections = await this.detectOnSlices(slices);

    // 3. NMS로 중복 제거
    const finalDetections = this.mergeDetections(allDetections);

    return finalDetections;
  }

  private async sliceImage(imageTensor: tf.Tensor3D): Promise<ImageSlice[]> {
    const [imgHeight, imgWidth, channels] = imageTensor.shape;

    // 오버랩 계산
    const overlap = Math.floor(this.sliceSize * this.overlapRatio);
    const step = this.sliceSize - overlap;

    const slices: ImageSlice[] = [];

    for (let y = 0; y < imgHeight; y += step) {
      for (let x = 0; x < imgWidth; x += step) {
        const sliceWidth = Math.min(this.sliceSize, imgWidth - x);
        const sliceHeight = Math.min(this.sliceSize, imgHeight - y);
        const sliceTensor = imageTensor.slice([y, x, 0], [sliceHeight, sliceWidth, channels]);

        const paddingHeight = this.sliceSize - sliceHeight;
        const paddingWidth = this.sliceSize - sliceWidth;
        const paddedTensor = tf.pad(sliceTensor, [
          [0, paddingHeight], // height 패딩
          [0, paddingWidth], // width 패딩
          [0, 0], // channel 패딩 없음
        ]);

        slices.push({
          imageTensor: paddedTensor,
          offsetX: x,
          offsetY: y,
        });
      }
    }

    return slices;
  }

  private async detectOnSlices(slices: ImageSlice[]): Promise<DetectedObject[]> {
    const allDetections: DetectedObject[] = [];

    for (const slice of slices) {
      try {
        // 기존 모델로 탐지
        const detections = await this.baseModel.detect(slice.imageTensor, 2000);

        // 좌표를 전체 이미지 좌표계로 변환
        const convertedDetections = detections.map((det: DetectedObject) => ({
          ...det,
          bbox: [
            det.bbox[0] + slice.offsetX,
            det.bbox[1] + slice.offsetY,
            det.bbox[2],
            det.bbox[3],
          ] as [number, number, number, number],
        }));

        allDetections.push(...convertedDetections);
      } catch (error) {
        console.log('Slice detection error:', error);
      } finally {
        slice.imageTensor.dispose();
      }
    }

    return allDetections;
  }

  private mergeDetections(detections: DetectedObject[]): DetectedObject[] {
    if (detections.length === 0) return [];

    // 라벨별로 그룹화
    const detectionsByLabel = new Map<string, DetectedObject[]>();

    for (const detection of detections) {
      if (!detectionsByLabel.has(detection.label)) {
        detectionsByLabel.set(detection.label, []);
      }
      detectionsByLabel.get(detection.label)!.push(detection);
    }

    // 각 라벨별로 NMS 적용
    const finalDetections: DetectedObject[] = [];

    for (const [label, labelDetections] of detectionsByLabel) {
      const nmsResults = this.applyNMS(labelDetections, this.nmsThreshold);
      finalDetections.push(...nmsResults);
    }

    return finalDetections;
  }

  private applyNMS(detections: DetectedObject[], threshold: number): DetectedObject[] {
    if (detections.length === 0) return [];

    // 점수별로 내림차순 정렬
    const sortedDetections = [...detections].sort((a, b) => b.score - a.score);

    const kept: DetectedObject[] = [];
    const suppressed = new Set<number>();

    for (let i = 0; i < sortedDetections.length; i++) {
      if (suppressed.has(i)) continue;

      const current = sortedDetections[i];
      kept.push(current);

      // 현재 detection과 겹치는 것들 제거
      for (let j = i + 1; j < sortedDetections.length; j++) {
        if (suppressed.has(j)) continue;

        const other = sortedDetections[j];
        const iou = this.calculateIoU(current.bbox, other.bbox);

        if (iou > threshold) {
          suppressed.add(j);
        }
      }
    }

    return kept;
  }

  private calculateIoU(
    box1: [number, number, number, number],
    box2: [number, number, number, number],
  ): number {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;

    // 박스의 끝점 계산
    const x1_max = x1 + w1;
    const y1_max = y1 + h1;
    const x2_max = x2 + w2;
    const y2_max = y2 + h2;

    // 교집합 영역 계산
    const intersect_x1 = Math.max(x1, x2);
    const intersect_y1 = Math.max(y1, y2);
    const intersect_x2 = Math.min(x1_max, x2_max);
    const intersect_y2 = Math.min(y1_max, y2_max);

    // 교집합이 없는 경우
    if (intersect_x2 <= intersect_x1 || intersect_y2 <= intersect_y1) {
      return 0;
    }

    // 교집합 면적
    const intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1);

    // 각 박스의 면적
    const box1_area = w1 * h1;
    const box2_area = w2 * h2;

    // 합집합 면적
    const union_area = box1_area + box2_area - intersect_area;

    return intersect_area / union_area;
  }
}
