import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConfidenceBadge } from "./ConfidenceBadge";

describe("ConfidenceBadge", () => {
  it("announces the confidence level", () => {
    render(<ConfidenceBadge level="medium" />);
    expect(screen.getByText(/medium confidence/i)).toBeInTheDocument();
  });
});
