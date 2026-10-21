import { AlertTriangle, CheckCircle2, CircleHelp } from "lucide-react";

type Level = "high" | "medium" | "low";

const styles: Record<Level, string> = {
  high: "bg-emerald-100 text-emerald-800",
  medium: "bg-amber-100 text-amber-800",
  low: "bg-rose-100 text-rose-800"
};

export function ConfidenceBadge({ level }: { level: Level }) {
  const Icon = level === "high" ? CheckCircle2 : level === "medium" ? CircleHelp : AlertTriangle;
  return (
    <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wider ${styles[level]}`}>
      <Icon size={14} aria-hidden="true" /> {level} confidence
    </span>
  );
}
