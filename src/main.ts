import fs from 'fs';
import { loadWaldo } from './waldo/waldo';
import { loadCocoSsd } from './coco-ssd/coco-ssd';

const PATH = './mock-images/mock1.png';

async function main() {
  const buf = fs.readFileSync(PATH);
  const imageDataUrl = buf.toString('base64');

  const model = await loadWaldo();
  // const model = await loadCocoSsd();

  console.time('all');
  console.group('all');

  for (let i = 0; i < 4; i += 1) {
    await model.detect(imageDataUrl);
  }

  // await Promise.all(
  //   Array(200)
  //     .fill(null)
  //     .map(() => model.detect(imageDataUrl)),
  // );

  console.groupEnd();
  console.timeEnd('all');
}

main();
