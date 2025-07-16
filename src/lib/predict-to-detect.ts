import { DetectedObject } from '../coco-ssd/coco-ssd';

export const predictToDetect = (prediction: DetectedObject) => ({
  label: prediction.label,
  score: prediction.score,
  bbox: {
    x: prediction.bbox[0],
    y: prediction.bbox[1],
    width: prediction.bbox[2],
    height: prediction.bbox[3],
  },
});

export const detectToPredict = (detection: {
  label: string;
  score: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}): DetectedObject => ({
  ...detection,
  label: detection.label,
  score: detection.score,
  bbox: [detection.bbox.x, detection.bbox.y, detection.bbox.width, detection.bbox.height],
});
