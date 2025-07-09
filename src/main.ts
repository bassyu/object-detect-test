import fs from "fs";
import { load } from "./waldo/waldo";

const PATH = "./mock-images/mock1.png";

async function main() {
  const buf = fs.readFileSync(PATH);
  const imageDataUrl = buf.toString("base64");

  const waldo = await load();

  console.time("all");
  console.group("all");
  for (let i = 0; i < 20; i += 1) {
    // console.time('function')
    // console.group('function')
    waldo.detect(imageDataUrl);
    // console.groupEnd()
    // console.timeEnd('function')
  }
  console.groupEnd();
  console.timeEnd("all");
}

main();
