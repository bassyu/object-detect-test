import * as tf from '@tensorflow/tfjs-node-gpu';

export function base64ToTensor(base64String: string): tf.Tensor3D | null {
  try {
    // base64 문자열에서 데이터 URL 프리픽스 제거
    const base64Data = base64String.replace(/^data:image\/[a-z]+;base64,/, '');
    // base64를 Buffer로 변환
    const imageBuffer = Buffer.from(base64Data, 'base64');
    // Buffer를 이미지 텐서로 디코딩
    const imageTensor = tf.node.decodeImage(imageBuffer, 3) as tf.Tensor3D;

    return imageTensor;
  } catch (error) {
    console.error('Why:', error);
    return null;
  }
}
