import type { ChatResponse, ClassificationResult } from "./types";
import { IS_STATIC_DEMO, STATIC_DEMO_MESSAGE } from "./deployment";

const API_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "";

async function parseError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : "The request could not be completed.";
  } catch {
    return "The service returned an unreadable response.";
  }
}

export async function classifyImage(file: File): Promise<ClassificationResult> {
  if (IS_STATIC_DEMO) throw new Error(STATIC_DEMO_MESSAGE);
  const body = new FormData();
  body.append("image", file);
  const response = await fetch(`${API_URL}/api/classify`, { method: "POST", body });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function askAssistant(payload: object): Promise<ChatResponse> {
  if (IS_STATIC_DEMO) throw new Error(STATIC_DEMO_MESSAGE);
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}
