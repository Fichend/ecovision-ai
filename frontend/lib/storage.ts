import type { StoredAnalysis } from "./types";

const LEGACY_KEY = "ecovision.latest-analysis";
const DATABASE_NAME = "ecovision";
const DATABASE_VERSION = 1;
const STORE_NAME = "analyses";
const LATEST_KEY = "latest";

let memoryAnalysis: StoredAnalysis | null = null;
let databasePromise: Promise<IDBDatabase> | null = null;

function openDatabase(): Promise<IDBDatabase> {
  if (!databasePromise) {
    databasePromise = new Promise((resolve, reject) => {
      if (typeof indexedDB === "undefined") {
        reject(new Error("IndexedDB is unavailable."));
        return;
      }

      const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains(STORE_NAME)) {
          database.createObjectStore(STORE_NAME);
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("Could not open browser storage."));
    });
  }
  return databasePromise;
}

function removeLegacyCopy(): void {
  try {
    sessionStorage.removeItem(LEGACY_KEY);
  } catch {
    // Storage may be disabled. IndexedDB or the in-memory copy still works.
  }
}

export async function saveAnalysis(analysis: StoredAnalysis): Promise<void> {
  memoryAnalysis = analysis;
  removeLegacyCopy();

  try {
    const database = await openDatabase();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).put(analysis, LATEST_KEY);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("Could not save the analysis."));
      transaction.onabort = () => reject(transaction.error ?? new Error("Saving the analysis was aborted."));
    });
  } catch {
    // Navigation still works in this tab through the in-memory copy. This also
    // keeps private-browsing or storage-disabled sessions usable.
  }
}

export async function loadAnalysis(): Promise<StoredAnalysis | null> {
  if (memoryAnalysis) return memoryAnalysis;

  try {
    const database = await openDatabase();
    const analysis = await new Promise<StoredAnalysis | null>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readonly");
      const request = transaction.objectStore(STORE_NAME).get(LATEST_KEY);
      request.onsuccess = () => resolve((request.result as StoredAnalysis | undefined) ?? null);
      request.onerror = () => reject(request.error ?? new Error("Could not load the analysis."));
    });
    memoryAnalysis = analysis;
    return analysis;
  } catch {
    return null;
  }
}
