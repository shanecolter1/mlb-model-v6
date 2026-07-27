import { promises as fs } from "node:fs";
import path from "node:path";

export class SnapshotStore {
  constructor(rootDir) {
    this.rootDir = rootDir;
  }

  async save(snapshot) {
    const date = snapshot.game_date;
    const dir = path.join(this.rootDir, date);
    await fs.mkdir(dir, { recursive: true });
    const safeTimestamp = snapshot.prediction_timestamp.replaceAll(":", "-");
    const file = path.join(dir, `${snapshot.game_id}_${safeTimestamp}.json`);
    await fs.writeFile(file, JSON.stringify(snapshot, null, 2), "utf8");
    return file;
  }

  async list(date) {
    const dir = path.join(this.rootDir, date);
    try {
      return (await fs.readdir(dir))
        .filter((name) => name.endsWith(".json"))
        .map((name) => path.join(dir, name));
    } catch (error) {
      if (error.code === "ENOENT") return [];
      throw error;
    }
  }

  async read(file) {
    return JSON.parse(await fs.readFile(file, "utf8"));
  }
}
