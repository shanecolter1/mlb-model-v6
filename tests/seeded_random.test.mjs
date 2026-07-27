import { strict as assert } from "node:assert";
import { createSeededRandom, seedFromGameId } from "../src/model/seeded_random.js";

const seed = seedFromGameId("12345", 1);
const a = createSeededRandom(seed);
const b = createSeededRandom(seed);
for (let i = 0; i < 20; i += 1) assert.equal(a(), b());
console.log("Seeded RNG tests passed.");
