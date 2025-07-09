/**
 * WALDO Object Detection using TensorFlow.js for Node.js with CUDA GPU
 * Adapted from browser version to work with base64 image inputs
 */

import * as tf from "@tensorflow/tfjs-node-gpu";
// import * as tf from "@tensorflow/tfjs-node";

import { CLASSES as WALDO_CLASSES } from "./classes";

const MODEL_PATH =  "./models/WALDO30_yolov8n_640x640/model.json";
const SIZE = 640;

export const CLASSES = WALDO_CLASSES;

export interface DetectedObject {
  bbox: [number, number, number, number]; // [x, y, width, height]
  class: string;
  score: number;
}

export interface ModelConfig {
  modelUrl?: string;
  gpuMemoryGrowth?: boolean;
}

export async function load(config: ModelConfig = {}) {
  const objectDetection = new WaldoObjectDetection(config.modelUrl, config.gpuMemoryGrowth);
  await objectDetection.load();
  return objectDetection;
}

export class WaldoObjectDetection {
  private modelPath: string;
  private model: tf.GraphModel | null = null;
  private gpuMemoryGrowth: boolean;

  constructor(modelUrl?: string, gpuMemoryGrowth = true) {
    this.modelPath = modelUrl ?? MODEL_PATH;
    this.gpuMemoryGrowth = gpuMemoryGrowth;
  }

  async load() {
    try {
      // await tf.setBackend('cpu');

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

      this.model = await tf.loadGraphModel(tf.io.fileSystem(this.modelPath) );
      
      // 백엔드 정보 출력
      console.log('Current backend:', tf.getBackend());
      console.log('Memory info:', tf.memory());
    } catch (error) {
      console.error('Error loading model:', error);
      throw error;
    }
  }

  private base64ToImageTensor(base64String: string): tf.Tensor3D {
    try {
      // base64 문자열에서 데이터 URL 프리픽스 제거
      const base64Data = base64String.replace(/^data:image\/[a-z]+;base64,/, '');
      // base64를 Buffer로 변환
      const imageBuffer = Buffer.from(base64Data, 'base64');
      // Buffer를 이미지 텐서로 디코딩
      const imageTensor = tf.node.decodeImage(imageBuffer, 3) as tf.Tensor3D;
      
      return imageTensor;
    } catch (error) {
      console.error('Error processing base64 image:', error);
      throw new Error('Failed to process base64 image');
    }
  }

  private preprocessImage(imageTensor: tf.Tensor3D ): tf.Tensor4D {
    return tf.tidy(() => {
      // 이미지 크기 조정
      imageTensor = tf.image.resizeBilinear(imageTensor, [SIZE, SIZE]);
      
      // 정규화 (0-255 -> 0-1)
      imageTensor = tf.div(imageTensor, 255.0);
      
      // 배치 차원 추가
      const batchTensor = tf.expandDims(imageTensor, 0) as tf.Tensor4D;
      
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
        })
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
    minScore: number
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
        const x1 = ((cx - w / 2) * originalWidth) / SIZE;
        const y1 = ((cy - h / 2) * originalHeight) / SIZE;
        const x2 = ((cx + w / 2) * originalWidth) / SIZE;
        const y2 = ((cy + h / 2) * originalHeight) / SIZE;

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
        class: CLASSES[classIds[idx]].displayName,
        score: scores[idx],
      });
    }

    return results;
  }

  async detect(
    base64Image: string,
    maxNumBoxes = Infinity,
    minScore = 0.5
  ): Promise<DetectedObject[]> {
    if (!this.model) {
      throw new Error("Model not loaded. Call load() first.");
    }

    try {

      let results: DetectedObject[] = [];

      // tf.tidy를 사용하여 메모리 누수 방지
      tf.tidy(() => {
        console.time('detect')
        console.group('detect')

        // 원본 이미지 크기 획득을 위한 임시 텐서 생성
        console.time('image2tensor')
        const originalImageTensor = this.base64ToImageTensor(base64Image);
        const originalHeight = originalImageTensor.shape[0];
        const originalWidth = originalImageTensor.shape[1];
        console.timeEnd('image2tensor')

        // console.time('preprocess')
        const inputTensor = this.preprocessImage(originalImageTensor);
        // console.timeEnd('preprocess')
      
        console.time('predict')
        const prediction = this.model!.predict(inputTensor) as tf.Tensor;
        console.timeEnd('predict')
    
        // console.time('postprocess')
        results = this.postprocess(prediction, originalWidth, originalHeight, maxNumBoxes, minScore);
        // console.timeEnd('postprocess')
      
        console.groupEnd()
        console.timeEnd('detect')
        console.log()

        // 텐서들은 tf.tidy에 의해 자동으로 정리됨
      });

      return results;
    } catch (error) {
      console.error('Error during detection:', error);
      throw error;
    }
  }

  async detectBatch(
    base64Images: string[],
    maxNumBoxes = Infinity,
    minScore = 0.5
  ): Promise<DetectedObject[][]> {
    if (!this.model) {
      throw new Error("Model not loaded. Call load() first.");
    }

    const results: DetectedObject[][] = [];
    
    for (const base64Image of base64Images) {
      try {
        const detection = await this.detect(base64Image, maxNumBoxes, minScore);
        results.push(detection);
      } catch (error) {
        console.error('Error detecting image in batch:', error);
        results.push([]); // 에러 발생 시 빈 배열 추가
      }
    }

    return results;
  }

  getModelInfo(): object {
    if (!this.model) {
      throw new Error("Model not loaded. Call load() first.");
    }

    return {
      modelPath: this.modelPath,
      backend: tf.getBackend(),
      memory: tf.memory(),
      inputShape: this.model.inputs[0].shape,
      outputShape: this.model.outputs[0].shape,
    };
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
