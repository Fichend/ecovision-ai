import { describe, expect, it } from "vitest";
import { formatPercent, titleCase } from "./format";

describe("format helpers", () => {
  it("formats model probabilities", () => expect(formatPercent(0.9233)).toBe("92.3%"));
  it("humanizes class names", () => expect(titleCase("mixed_plastic")).toBe("Mixed Plastic"));
});
