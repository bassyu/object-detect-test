import * as tf from '@tensorflow/tfjs-node-gpu';
// import * as tf from '@tensorflow/tfjs-node';

import { CLASSES as WALDO_CLASSES } from './classes';

// export const WALDO_SZIE = 416;
// const MODEL_PATH = './models/WALDO30_yolov8n_416x416_saved_model';

export const WALDO_SZIE = 640;
const MODEL_PATH = './models/WALDO30_yolov8n_640x640_saved_model';

export const CLASSES = WALDO_CLASSES;

export interface DetectedObject {
  bbox: [number, number, number, number]; // [x, y, width, height]
  label: string;
  score: number;
}

export interface ModelConfig {
  modelUrl?: string;
  gpuMemoryGrowth?: boolean;
}

export async function loadWaldo(config: ModelConfig = {}) {
  const objectDetection = new WaldoObjectDetection(config.modelUrl, config.gpuMemoryGrowth);
  await objectDetection.load();
  return objectDetection;
}

export class WaldoObjectDetection {
  private modelPath: string;
  private model: any | tf.GraphModel | null = null;
  private gpuMemoryGrowth: boolean;

  constructor(modelUrl?: string, gpuMemoryGrowth = true) {
    this.modelPath = modelUrl ?? MODEL_PATH;
    this.gpuMemoryGrowth = gpuMemoryGrowth;
  }

  async load() {
    try {
      // CUDA GPU 백엔드 설정
      await tf.ready();

      // GPU 메모리 증가 설정 (선택사항)
      if (this.gpuMemoryGrowth) {
        const gpuBackend = tf.backend();
        if (gpuBackend && 'gpuKernelBackend' in gpuBackend) {
          // GPU 메모리 설정은 TensorFlow.js Node.js에서 자동으로 관리됨
          console.log('GPU backend initialized');
        }
      }

      // this.model = await tf.loadGraphModel(tf.io.fileSystem(this.modelPath));
      // this.model = await tf.loadGraphModel(
      //   tf.io.fileSystem('./src/waldo/WALDO30_yolov8n_416x416_web_model/model.json'),
      // );
      this.model = await tf.node.loadSavedModel(this.modelPath);

      // 백엔드 정보 출력
      console.log('Current backend:', tf.getBackend());
      console.log('Memory info:', tf.memory());
    } catch (error) {
      console.error('Error loading model:', error);
      throw error;
    }
  }

  private preprocessImage(imageTensor: tf.Tensor3D): tf.Tensor4D {
    return tf.tidy(() => {
      // 이미지 크기 조정
      let processedTensor = tf.image.resizeBilinear(imageTensor, [WALDO_SZIE, WALDO_SZIE]);

      // 정규화 (0-255 -> 0-1)
      processedTensor = tf.div(processedTensor, 255.0);

      // 배치 차원 추가
      const batchTensor = tf.expandDims(processedTensor, 0) as tf.Tensor4D;

      return batchTensor;
    });
  }

  private nms(boxes: number[][], scores: number[], iouThreshold: number = 0.4): number[] {
    const indices = Array.from(Array(scores.length).keys()).sort((a, b) => scores[b] - scores[a]);
    const keep: number[] = [];

    while (indices.length > 0) {
      const current = indices.shift() ?? 0;
      keep.push(current);

      const currentBox = boxes[current];
      indices.splice(
        0,
        indices.length,
        ...indices.filter((idx) => {
          const box = boxes[idx];
          const iou = this.calculateIoU(currentBox, box);
          return iou <= iouThreshold;
        }),
      );
    }

    return keep;
  }

  private calculateIoU(box1: number[], box2: number[]): number {
    const [x1_1, y1_1, x2_1, y2_1] = box1;
    const [x1_2, y1_2, x2_2, y2_2] = box2;

    const intersectionArea =
      Math.max(0, Math.min(x2_1, x2_2) - Math.max(x1_1, x1_2)) *
      Math.max(0, Math.min(y2_1, y2_2) - Math.max(y1_1, y1_2));

    const box1Area = (x2_1 - x1_1) * (y2_1 - y1_1);
    const box2Area = (x2_2 - x1_2) * (y2_2 - y1_2);
    const unionArea = box1Area + box2Area - intersectionArea;

    return intersectionArea / unionArea;
  }

  private postprocess(
    output: tf.Tensor,
    originalWidth: number,
    originalHeight: number,
    maxNumBoxes: number,
    minScore: number,
  ): DetectedObject[] {
    const data = output.dataSync() as Float32Array;
    const [, numClasses, numBoxes] = output.shape;

    const boxes: number[][] = [];
    const scores: number[] = [];
    const classIds: number[] = [];

    for (let i = 0; i < numBoxes; i++) {
      const cx = data[i];
      const cy = data[numBoxes + i];
      const w = data[2 * numBoxes + i];
      const h = data[3 * numBoxes + i];

      let maxScore = 0;
      let maxClassId = 0;

      for (let j = 4; j < numClasses; j++) {
        const score = data[j * numBoxes + i];
        if (score > maxScore) {
          maxScore = score;
          maxClassId = j - 4;
        }
      }

      if (maxScore >= minScore) {
        const x1 = ((cx - w / 2) * originalWidth) / WALDO_SZIE;
        const y1 = ((cy - h / 2) * originalHeight) / WALDO_SZIE;
        const x2 = ((cx + w / 2) * originalWidth) / WALDO_SZIE;
        const y2 = ((cy + h / 2) * originalHeight) / WALDO_SZIE;

        boxes.push([x1, y1, x2, y2]);
        scores.push(maxScore);
        classIds.push(maxClassId);
      }
    }

    const keepIndices = this.nms(boxes, scores);
    const results: DetectedObject[] = [];

    for (let i = 0; i < Math.min(keepIndices.length, maxNumBoxes); i++) {
      const idx = keepIndices[i];
      const [x1, y1, x2, y2] = boxes[idx];

      results.push({
        bbox: [x1, y1, x2 - x1, y2 - y1],
        label: CLASSES[classIds[idx]].displayName,
        score: scores[idx],
      });
    }

    return results;
  }

  async detect(
    imageTensor: tf.Tensor3D,
    maxNumBoxes = Infinity,
    minScore = 0.5,
  ): Promise<DetectedObject[]> {
    if (!this.model) {
      throw new Error('Model not loaded. Call load() first.');
    }

    try {
      // console.time('detect');

      const originalHeight = imageTensor.shape[0];
      const originalWidth = imageTensor.shape[1];

      const inputTensor = this.preprocessImage(imageTensor);

      // console.time('execute');
      // const prediction = this.model!.execute(inputTensor) as tf.Tensor;
      const prediction = this.model!.predict(inputTensor) as tf.Tensor;
      // console.timeEnd('execute');

      const results = this.postprocess(
        prediction,
        originalWidth,
        originalHeight,
        maxNumBoxes,
        minScore,
      );

      // console.timeEnd('detect');

      inputTensor.dispose();
      inputTensor.dispose();
      prediction.dispose();

      return results;
    } catch (error) {
      console.error('Error during detection:', error);
      throw error;
    }
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    // 메모리 정리
    tf.disposeVariables();
  }
}
