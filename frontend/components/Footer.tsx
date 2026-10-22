import { ExternalLink, Leaf } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-24 border-t border-forest-900/10 bg-white/70">
      <div className="page-shell grid gap-8 py-10 sm:grid-cols-2 sm:items-end">
        <div>
          <div className="mb-3 flex items-center gap-2 font-bold text-forest-900"><Leaf size={18} /> EcoVision AI</div>
          <p className="max-w-lg text-sm leading-6 text-slate-600">
            Visual classification is an estimate. Always confirm material composition and local collection rules before disposal.
          </p>
        </div>
        <a className="flex items-center gap-2 text-sm font-semibold text-forest-800 sm:justify-self-end" href="https://github.com/Fichend/ecovision-ai/blob/main/PROJECT_BUILD_GUIDE_12CLASS.md" target="_blank" rel="noreferrer">
          Data sources and model card <ExternalLink size={15} />
        </a>
      </div>
    </footer>
  );
}
