import WebSocket from 'ws';
import express from 'express';
import http from 'http';

import { loadWaldo } from './waldo/waldo';
import { VideoStreamer } from './lib/video-streamer';
import { SlicingModel } from './lib/slicing-model';
import { base64ToTensor } from './lib/base64-to-tensor';

const PORT = 8346;

const VIDEO_PATH =
  //line
  // 'https://videos.pexels.com/video-files/2053855/2053855-uhd_2560_1440_30fps.mp4'; // 드론 밤 도로
  'https://videos.pexels.com/video-files/1263198/1263198-uhd_2560_1440_30fps.mp4'; // 드론 낮 도로
// 'https://videos.pexels.com/video-files/17845991/17845991-uhd_1440_2560_30fps.mp4'; // 드론 낮 세로 도로
// 'https://videos.pexels.com/video-files/3150502/3150502-uhd_2560_1440_30fps.mp4'; // 드론 낮 가로 도로

// "https://www.shutterstock.com/shutterstock/videos/1109631067/preview/stock-footage-car-crash-in-miami-injured-driver-waiting-for-ambulance-to-arrive-people-helping-accident-victim.webm"; // 도로 옆
// "https://videos.pexels.com/video-files/27375219/12127641_2320_1080_30fps.mp4"; // 신호등
// "https://nearthlab-public.s3.ap-northeast-2.amazonaws.com/ai-experience-zone/accident-back.mov"; // 도로 옆 뒤
// "https://videos.pexels.com/video-files/19216259/19216259-hd_1920_1080_30fps.mp4"; // 농구 1
// 'https://videos.pexels.com/video-files/30514788/13074046_1920_1080_30fps.mp4'; // 농구 2

const VIDEO_FPS = 240;

const videoStreamer = new VideoStreamer(VIDEO_PATH, VIDEO_FPS);
videoStreamer.createVideoStream();

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

wss.on('connection', async (ws) => {
  const baseModel = await loadWaldo();
  // const baseModel = await loadCocoSsd();

  const model = new SlicingModel({
    baseModel,
    sliceSize: 400,
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

    console.log('\non message');

    console.time('slice detect');
    const detections = await model.detect(imageTensor).catch((error) => {
      console.error('detect error', error);
      return [];
    });
    console.timeEnd('slice detect');

    ws.send(
      JSON.stringify({
        frame: imageDataUrl,
        detections,
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
