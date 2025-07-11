import fs from 'fs';
import { loadWaldo } from './waldo/waldo';
import { loadCocoSsd } from './coco-ssd/coco-ssd';
import { delay } from 'es-toolkit';

const PATH = './mock-datas/mock1.png';
const DETECT_DELAY = 1000;

async function main() {
  const buf = fs.readFileSync(PATH);
  const imageDataUrl = buf.toString('base64');

  const model = await loadWaldo();
  // const model = await loadCocoSsd();

  // console.time('all');
  // console.group('all');

  // for (let i = 0; i < 4; i += 1) {
  //   await model.detect(imageDataUrl);
  // }

  while (true) {
    await model.detect(imageDataUrl);

    await delay(DETECT_DELAY);
  }

  // await Promise.all(
  //   Array(200)
  //     .fill(null)
  //     .map(() => model.detect(imageDataUrl)),
  // );

  // console.groupEnd();
  // console.timeEnd('all');
}

main();
