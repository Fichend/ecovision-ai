"use client";

import { Leaf, Menu, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const links = [
  ["Recognize", "/recognize"],
  ["Assistant", "/assistant"],
  ["Evaluation", "/evaluation"]
] as const;

export function Navigation() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-50 border-b border-forest-900/10 bg-cream/90 backdrop-blur-xl">
      <div className="page-shell flex min-h-20 items-center justify-between">
        <Link className="flex items-center gap-3 font-bold text-forest-900" href="/">
          <span className="grid size-10 place-items-center rounded-2xl bg-forest-800 text-white">
            <Leaf aria-hidden="true" size={20} />
          </span>
          <span>EcoVision AI</span>
        </Link>
        <nav className="hidden items-center gap-2 md:flex" aria-label="Primary navigation">
          {links.map(([label, href]) => (
            <Link
              key={href}
              className={`rounded-full px-4 py-2 text-sm font-semibold transition ${pathname === href ? "bg-forest-800 text-white" : "text-slate-600 hover:bg-white hover:text-forest-900"}`}
              href={href}
            >
              {label}
            </Link>
          ))}
        </nav>
        <button
          className="grid size-11 place-items-center rounded-full border border-forest-900/15 bg-white md:hidden"
          onClick={() => setOpen((value) => !value)}
          aria-label="Toggle navigation"
          aria-expanded={open}
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>
      {open && (
        <nav className="page-shell flex flex-col gap-2 border-t border-forest-900/10 py-4 md:hidden">
          {links.map(([label, href]) => (
            <Link key={href} className="rounded-xl px-4 py-3 font-semibold hover:bg-white" href={href} onClick={() => setOpen(false)}>
              {label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
