export function cameraErrorMessage(error: unknown): string {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Camera access was blocked. Allow camera access for this site, then try again.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No camera was found on this device. Choose an existing image instead.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "The camera is busy or unavailable. Close other camera apps and try again.";
  }
  return "The camera could not be started. Try another browser or choose an existing image.";
}

export function fitCameraFrame(
  width: number,
  height: number,
  maxEdge = 1600,
): { width: number; height: number } {
  const scale = Math.min(1, maxEdge / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}
