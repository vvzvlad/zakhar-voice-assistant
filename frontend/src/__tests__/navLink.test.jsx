// @vitest-environment jsdom
// The panel's one in-panel link (primitives.jsx NavLink): its text then "→", the shared
// .z-link look, a click to its own handler, and without one a click that reaches the
// enclosing clickable element (a dashboard tile).
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { NavLink } from "../components/primitives.jsx";

afterEach(cleanup);

describe("NavLink", () => {
  it("reads as its text and an arrow, in the shared link style, and calls its handler", () => {
    const go = vi.fn();
    render(<NavLink onClick={go}>Manage devices</NavLink>);
    const link = screen.getByRole("button", { name: "Manage devices →" });
    expect(link).toHaveClass("z-link");
    expect(link).toHaveAttribute("type", "button");
    fireEvent.click(link);
    expect(go).toHaveBeenCalledTimes(1);
  });

  it("without a handler leaves the click to the element around it", () => {
    const open = vi.fn();
    render(<div onClick={open}><NavLink>Configure</NavLink></div>);
    fireEvent.click(screen.getByRole("button", { name: "Configure →" }));
    expect(open).toHaveBeenCalledTimes(1);
  });
});
