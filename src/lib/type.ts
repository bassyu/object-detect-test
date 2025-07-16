export interface DetectedObject {
  bbox: [number, number, number, number]; // [x, y, width, height]
  label: string;
  score: number;
}
