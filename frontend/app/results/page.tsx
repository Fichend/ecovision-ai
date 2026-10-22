"use client";

import { ArrowRight, Eye, Gauge, RotateCcw, Timer } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { formatPercent, titleCase } from "@/lib/format";
import { loadAnalysis } from "@/lib/storage";
import type { StoredAnalysis } from "@/lib/types";

export default function ResultsPage() {
  const [analysis, setAnalysis] = useState<StoredAnalysis | null | undefined>(undefined);
  useEffect(() => {
    let active = true;
    void loadAnalysis().then((stored) => {
      if (active) setAnalysis(stored);
    });
    return () => { active = false; };
  }, []);
  if (analysis === undefined) return <div className="page-shell py-24 text-center">Loading result…</div>;
  if (!analysis) return <section className="page-shell py-24 text-center"><h1 className="text-4xl font-black text-forest-900">No analysis yet</h1><p className="mt-3 text-slate-600">Upload an image to create a result.</p><Link className="button-primary mt-7" href="/recognize">Analyze an item</Link></section>;
  const { result, imageDataUrl } = analysis;
  const rejected = result.accepted === false || result.predicted_class === "unknown";
  return (
    <section className="page-shell py-14 sm:py-20">
      <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end"><div><p className="eyebrow">Recognition result</p><h1 className="mt-3 text-4xl font-black tracking-tight text-forest-900 sm:text-5xl">{rejected ? "Unable to identify reliably" : `Likely ${titleCase(result.predicted_class)}`}</h1>{rejected && result.raw_predicted_class && <p className="mt-3 text-sm text-slate-600">Best visual match: {titleCase(result.raw_predicted_class)}. This guess was rejected by the confidence gate.</p>}</div><ConfidenceBadge level={result.confidence_level} /></div>
      {rejected && <div className="mt-7 rounded-2xl border border-amber-300 bg-amber-50 p-5 text-sm leading-6 text-amber-950"><strong>No disposal category was assigned.</strong> Photograph one item in brighter light against a plain background, or describe its material to the assistant. Do not rely on the best visual match alone.</div>}
      <div className="mt-9 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="panel grid gap-4 p-5 sm:grid-cols-2">
          <figure><figcaption className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-500">Original image</figcaption><div className="overflow-hidden rounded-2xl bg-slate-100"><img className="aspect-square w-full object-contain" src={imageDataUrl} alt="Uploaded item" /></div></figure>
          <figure><figcaption className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500"><Eye size={15} /> Grad-CAM attention</figcaption><div className="overflow-hidden rounded-2xl bg-slate-100">{result.gradcam_data_url ? <img className="aspect-square w-full object-contain" src={result.gradcam_data_url} alt="Grad-CAM model attention heatmap" /> : <div className="grid aspect-square place-items-center p-6 text-center text-sm text-slate-500">No heatmap was produced.</div>}</div></figure>
          <p className="sm:col-span-2 text-xs leading-5 text-slate-500">The heatmap shows where the model focused. It does not prove that those regions caused a correct prediction.</p>
        </div>
        <div className="grid gap-6">
          <div className="panel p-7"><div className="flex items-center justify-between"><h2 className="text-xl font-black text-forest-900">Top predictions</h2><Gauge className="text-forest-600" /></div><div className="mt-6 space-y-5">{result.top_predictions.map((prediction) => <div key={prediction.label}><div className="mb-2 flex justify-between text-sm"><span className="font-bold text-forest-900">{titleCase(prediction.label)}</span><span>{formatPercent(prediction.probability)}</span></div><div className="h-2 overflow-hidden rounded-full bg-forest-100"><div className="h-full rounded-full bg-forest-700" style={{ width: `${prediction.probability * 100}%` }} /></div></div>)}</div></div>
          <div className="panel p-7"><div className="flex items-center gap-2 text-sm font-bold text-forest-800"><Timer size={17} /> {result.inference_time_ms.toFixed(0)} ms · {result.model_version}</div><p className="mt-4 text-sm leading-6 text-slate-600">{result.uncertainty_message}</p></div>
        </div>
      </div>
      <div className="mt-8 flex flex-col gap-3 sm:flex-row"><Link className={rejected ? "button-secondary" : "button-primary"} href="/assistant">Ask the recycling assistant <ArrowRight size={18} /></Link><Link className={rejected ? "button-primary" : "button-secondary"} href="/recognize"><RotateCcw size={18} /> {rejected ? "Retake or choose another image" : "Analyze another item"}</Link></div>
    </section>
  );
}
