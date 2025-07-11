import WebSocket from 'ws';
import express from 'express';
import http from 'http';

import { loadWaldo } from './waldo/waldo';
import { VideoStreamer } from './lib/video-streamer';
import { SlicingModel } from './lib/slicing-model';
import { base64ToTensor } from './lib/base64-to-tensor';

const PORT = 8346;

const VIDEO_PATH = 'https://videos.pexels.com/video-files/1263198/1263198-uhd_2560_1440_30fps.mp4';
const VIDEO_FPS = 240;

// const DETECT_SLICE_SIZE = 320;
// const DETECT_SLICE_SIZE = 480;
const DETECT_SLICE_SIZE = 640;

const videoStreamer = new VideoStreamer(VIDEO_PATH, VIDEO_FPS);
videoStreamer.createVideoStream();

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

wss.on('connection', async (ws) => {
  const baseModel = await loadWaldo();

  const model = new SlicingModel({
    baseModel,
    sliceSize: DETECT_SLICE_SIZE,
  });

  ws.on('message', async (message) => {
    const currFrame = videoStreamer.frame;
    if (!currFrame) {
      return;
    }

    const imageDataUrl = currFrame.toString('base64');
    const imageTensor = base64ToTensor(imageDataUrl);
    if (!imageTensor) {
      return;
    }

    console.time('all');
    const detections = await model.detect(imageTensor).catch((error) => {
      console.error('detect error');
      return [];
    });
    console.timeEnd('all');

    ws.send(
      JSON.stringify({
        frame: imageDataUrl,
        detections: detections,
        timestamp: Date.now(),
      }),
    );
  });

  ws.on('close', () => {
    console.log('클라이언트 연결 종료');
  });

  ws.on('error', (error) => {
    console.error('WebSocket 오류:', error);
  });
});

server.listen(PORT, () => {
  console.log(`서버가 포트 ${PORT}에서 실행 중입니다`);
});

process.on('SIGINT', () => {
  console.log('서버 종료 중...');

  process.exit(0);
});
