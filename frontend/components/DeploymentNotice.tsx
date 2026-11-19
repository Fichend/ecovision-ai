import { AlertTriangle } from "lucide-react";
import { IS_STATIC_DEMO, STATIC_DEMO_MESSAGE } from "@/lib/deployment";

export function DeploymentNotice() {
  if (!IS_STATIC_DEMO) return null;

  return (
    <div className="border-b border-amber-300 bg-amber-50 text-amber-950">
      <div className="page-shell flex items-start gap-3 py-3 text-sm leading-6">
        <AlertTriangle className="mt-0.5 shrink-0 text-amber-700" size={18} aria-hidden="true" />
        <p><strong>Static demonstration:</strong> {STATIC_DEMO_MESSAGE}</p>
      </div>
    </div>
  );
}
