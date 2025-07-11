import fs from 'fs';
import WebSocket from 'ws';
import express from 'express';
import http from 'http';
// import { streamMp4ToWebSocket, currFrame } from './stream';
import { loadWaldo } from './waldo/waldo';
import { PassThrough } from 'stream';
import { VideoStreamer } from './lib/video-streamer';

const PORT = 8346;
const IMAGE_PATH = './mock-datas/mock1.png';
const VIDEO_PATH = 'https://videos.pexels.com/video-files/1263198/1263198-uhd_2560_1440_30fps.mp4';

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
const videoStreamer = new VideoStreamer(VIDEO_PATH, 300);

let currnetFrame: any = null;

wss.on('connection', async (ws) => {
  // const buf = fs.readFileSync(IMAGE_PATH);
  // const imageDataUrl = buf.toString('base64');

  const model = await loadWaldo();

  const videoStream = videoStreamer.createVideoStream();
  videoStream.on('data', (chunk) => {
    currnetFrame = chunk;
  });

  ws.on('message', async (message) => {
    console.log('message', message);
    if (!currnetFrame) {
      return;
    }

    const imageDataUrl = currnetFrame.toString('base64');

    const detections = await model.detect(imageDataUrl, 2000).catch(() => {
      console.log('detect error');
      return [];
    });

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
