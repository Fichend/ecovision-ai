import { describe, expect, it } from "vitest";
import { cameraErrorMessage, fitCameraFrame } from "./camera";

describe("camera helpers", () => {
  it("scales a camera frame without changing its aspect ratio", () => {
    expect(fitCameraFrame(4000, 3000)).toEqual({ width: 1600, height: 1200 });
    expect(fitCameraFrame(640, 480)).toEqual({ width: 640, height: 480 });
  });

  it("gives a useful permission error", () => {
    expect(cameraErrorMessage(new DOMException("denied", "NotAllowedError"))).toContain(
      "Allow camera access",
    );
  });
});
