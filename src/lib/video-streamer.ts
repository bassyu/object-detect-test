// videoStream.js
import { throttle } from 'es-toolkit';
import ffmpeg from 'fluent-ffmpeg';
import { PassThrough, Writable } from 'stream';

const OUTPUT_OPTIONS = [
  '-vcodec mjpeg',
  '-f image2pipe',
  '-vf scale=1200:900', // 리사이즈로 성능 최적화
];

export class VideoStreamer {
  videoPath;
  targetFps;
  currentStream: null | PassThrough | Writable;
  frame: any = null;

  constructor(videoPath: string, targetFps = 15) {
    this.videoPath = videoPath;
    this.targetFps = targetFps;
    this.currentStream = null;
  }

  createVideoStream() {
    const stream = new PassThrough();

    this.currentStream = ffmpeg(this.videoPath)
      .fps(this.targetFps)
      .format('image2pipe')
      .outputOptions(OUTPUT_OPTIONS)
      .on('error', (err) => {
        console.error('FFmpeg 오류:', err);
      })
      .on('end', () => {
        console.log('비디오 스트림 종료, 재시작...');
        // 비디오 끝나면 처음부터 재시작
        setTimeout(() => this.createVideoStream(), 1000);
      })
      .pipe(stream)
      // TODO: CPU 성능 개선 필요
      .on('data', (chunk) => {
        this.frame = chunk;
      });

    return stream;
  }

  stop() {
    if (this.currentStream) {
      this.currentStream.destroy();
    }
  }
}
