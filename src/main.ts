import fs from 'fs';
import { loadWaldo } from './waldo/waldo';
import { loadCocoSsd } from './coco-ssd/coco-ssd';

const PATH = './mock-images/mock1.png';

async function main() {
  const buf = fs.readFileSync(PATH);
  const imageDataUrl = buf.toString('base64');

  const [waldo, cocoSsd] = await Promise.all([loadWaldo(), loadCocoSsd()]);

  console.time('all');
  console.group('all');
  for (let i = 0; i < 20; i += 1) {
    // console.time('function')
    // console.group('function')

    waldo.detect(imageDataUrl);
    // cocoSsd.detect(imageDataUrl);

    // console.groupEnd()
    // console.timeEnd('function')
  }
  console.groupEnd();
  console.timeEnd('all');
}

main();
