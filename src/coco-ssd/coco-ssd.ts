import * as tf from '@tensorflow/tfjs-node-gpu';
// import * as tf from '@tensorflow/tfjs-node';
import * as tfconv from '@tensorflow/tfjs-converter';

import { CLASSES as COCO_CLASSES } from './classes';

const BASE_PATH = 'https://storage.googleapis.com/tfjs-models/savedmodel/';

export const CLASSES = COCO_CLASSES;

/** @docinline */
export type ObjectDetectionBaseModel = 'mobilenet_v1' | 'mobilenet_v2' | 'lite_mobilenet_v2';

export interface DetectedObject {
  bbox: [number, number, number, number]; // [x, y, width, height]
  label: string;
  score: number;
}

/**
 * Coco-ssd model loading is configurable using the following config dictionary.
 */
export interface ModelConfig {
  /**
   * It determines which object detection architecture to load. The supported
   * architectures are: 'mobilenet_v1', 'mobilenet_v2' and 'lite_mobilenet_v2'.
   * It is default to 'lite_mobilenet_v2'.
   */
  base?: ObjectDetectionBaseModel;
  /**
   *
   * An optional string that specifies custom url of the model. This is useful
   * for area/countries that don't have access to the model hosted on GCP.
   */
  modelUrl?: string;
  /**
   * Optional GPU memory growth setting
   */
  gpuMemoryGrowth?: boolean;
}

export async function loadCocoSsd(config: ModelConfig = {}) {
  if (tf == null) {
    throw new Error(
      `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`,
    );
  }
  const base = config.base ?? 'lite_mobilenet_v2';
  const modelUrl = config.modelUrl;
  const gpuMemoryGrowth = config.gpuMemoryGrowth ?? true;

  if (!['mobilenet_v1', 'mobilenet_v2', 'lite_mobilenet_v2'].includes(base)) {
    throw new Error(
      `ObjectDetection constructed with invalid base model ` +
        `${base}. Valid names are 'mobilenet_v1',` +
        ` 'mobilenet_v2' and 'lite_mobilenet_v2'.`,
    );
  }

  const objectDetection = new ObjectDetection(base, modelUrl, gpuMemoryGrowth);
  await objectDetection.load();
  return objectDetection;
}

export class ObjectDetection {
  private modelPath: string;
  private model: tfconv.GraphModel | null = null;
  private gpuMemoryGrowth: boolean;

  constructor(base: ObjectDetectionBaseModel, modelUrl?: string, gpuMemoryGrowth = true) {
    this.modelPath = modelUrl ?? `${BASE_PATH}${this.getPrefix(base)}/model.json`;
    this.gpuMemoryGrowth = gpuMemoryGrowth;
  }

  private getPrefix(base: ObjectDetectionBaseModel) {
    return base === 'lite_mobilenet_v2' ? `ssd${base}` : `ssd_${base}`;
  }

  async load() {
    try {
      // GPU 백엔드 설정
      await tf.ready();

      // GPU 메모리 증가 설정 (선택사항)
      if (this.gpuMemoryGrowth) {
        const gpuBackend = tf.backend();
        if (gpuBackend && 'gpuKernelBackend' in gpuBackend) {
          // GPU 메모리 설정은 TensorFlow.js Node.js에서 자동으로 관리됨
          console.log('GPU backend initialized');
        }
      }

      this.model = await tfconv.loadGraphModel(this.modelPath);

      // 백엔드 정보 출력
      console.log('Current backend:', tf.getBackend());
      console.log('Memory info:', tf.memory());

      // 모델 워밍업
      const zeroTensor = tf.zeros([1, 300, 300, 3], 'int32');
      const result = (await this.model.executeAsync(zeroTensor)) as tf.Tensor[];
      await Promise.all(result.map((t) => t.data()));
      result.map((t) => t.dispose());
      zeroTensor.dispose();
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

  private async infer(
    base64Image: string,
    maxNumBoxes: number,
    minScore: number,
  ): Promise<DetectedObject[]> {
    try {
      // console.time('detect');
      // console.group('detect');

      // console.time('image2tensor');
      const imageTensor = this.base64ToImageTensor(base64Image);
      // console.timeEnd('image2tensor');

      const batched = tf.expandDims(imageTensor, 0);

      const height = batched.shape[1];
      const width = batched.shape[2];

      // console.time('predict');
      const result = (await this.model!.executeAsync(batched)) as tf.Tensor[];
      // console.timeEnd('predict');

      const scores = result[0].dataSync() as Float32Array;
      const boxes = result[1].dataSync() as Float32Array;

      batched.dispose();
      imageTensor.dispose();

      const [maxScores, classes] = this.calculateMaxScores(
        scores,
        result[0].shape[1] ?? 0,
        result[0].shape[2] ?? 0,
      );

      const indexTensor = tf.tidy(() => {
        const boxes2 = tf.tensor2d(boxes, [result[1].shape[1] ?? 0, result[1].shape[3] ?? 0]);
        return tf.image.nonMaxSuppression(boxes2, maxScores, maxNumBoxes, minScore, minScore);
      });

      const indexes = indexTensor.dataSync() as Float32Array;
      indexTensor.dispose();

      const results = this.buildDetectedObjects(
        width ?? 0,
        height ?? 0,
        boxes,
        maxScores,
        indexes,
        classes,
      );

      // console.groupEnd();
      // console.timeEnd('detect');
      // console.log();

      // result 텐서들 정리
      result.forEach((tensor) => tensor.dispose());

      return results;
    } catch (error) {
      console.error('Error during inference:', error);
      throw error;
    }
  }

  private buildDetectedObjects(
    width: number,
    height: number,
    boxes: Float32Array,
    scores: number[],
    indexes: Float32Array,
    classes: number[],
  ): DetectedObject[] {
    const count = indexes.length;
    const objects: DetectedObject[] = [];
    for (let i = 0; i < count; i++) {
      const bbox: [number, number, number, number] = [0, 0, 0, 0];
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j];
      }
      const minY = bbox[0] * height;
      const minX = bbox[1] * width;
      const maxY = bbox[2] * height;
      const maxX = bbox[3] * width;
      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;
      objects.push({
        bbox,
        label: CLASSES[classes[indexes[i]] + 1].displayName,
        score: scores[indexes[i]],
      });
    }
    return objects;
  }

  private calculateMaxScores(
    scores: Float32Array,
    numBoxes: number,
    numClasses: number,
  ): [number[], number[]] {
    const maxes: number[] = [];
    const classes: number[] = [];
    for (let i = 0; i < numBoxes; i++) {
      let max = Number.MIN_VALUE;
      let index = -1;
      for (let j = 0; j < numClasses; j++) {
        if (scores[i * numClasses + j] > max) {
          max = scores[i * numClasses + j];
          index = j;
        }
      }
      maxes[i] = max;
      classes[i] = index;
    }
    return [maxes, classes];
  }

  async detect(base64Image: string, maxNumBoxes = 20, minScore = 0.5): Promise<DetectedObject[]> {
    return this.infer(base64Image, maxNumBoxes, minScore);
  }

  dispose() {
    if (this.model != null) {
      this.model.dispose();
    }
    // 메모리 정리
    tf.disposeVariables();
  }
}
