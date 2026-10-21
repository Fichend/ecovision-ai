import { AlertTriangle, CheckCircle2, Database, FlaskConical, Gauge, Microscope } from "lucide-react";

const metrics = [
  ["Accuracy", 0.9233], ["Macro-F1", 0.9011], ["Macro precision", 0.8947], ["Macro recall", 0.9123]
] as const;
const classes = [
  ["Cardboard", 0.9833, 60], ["Glass", 0.9078, 75], ["Metal", 0.9219, 61], ["Paper", 0.96, 89], ["Plastic", 0.9103, 72], ["Trash", 0.7234, 21]
] as const;
const matrix = [[59,0,0,0,1,0],[0,64,4,1,6,0],[0,0,59,0,1,1],[0,0,0,84,0,5],[0,2,2,0,66,2],[0,0,1,2,1,17]];
const labels = ["C", "G", "M", "Pa", "Pl", "T"];

export default function EvaluationPage() {
  return (
    <section className="page-shell py-14 sm:py-20">
      <div className="max-w-3xl"><p className="eyebrow">Model evaluation</p><h1 className="mt-4 text-4xl font-black tracking-tight text-forest-900 sm:text-5xl">Performance, without the victory lap.</h1><p className="mt-5 text-lg leading-8 text-slate-600">Results use the best validation-selected checkpoint and a held-out test split where same-label near-duplicates are kept together.</p></div>
      <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">{metrics.map(([label, value]) => <div className="panel p-6" key={label}><p className="text-sm font-semibold text-slate-500">{label}</p><p className="mt-3 text-4xl font-black text-forest-900">{(value * 100).toFixed(1)}%</p></div>)}</div>
      <div className="mt-6 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <article className="panel p-7 sm:p-9"><div className="flex items-center gap-3"><Gauge className="text-forest-700" /><h2 className="text-2xl font-black text-forest-900">Per-class F1</h2></div><div className="mt-7 space-y-5">{classes.map(([label, score, support]) => <div key={label}><div className="mb-2 flex justify-between text-sm"><span className="font-bold text-forest-900">{label} <span className="font-normal text-slate-400">n={support}</span></span><span>{(score * 100).toFixed(1)}%</span></div><div className="h-3 overflow-hidden rounded-full bg-forest-100"><div className={`h-full rounded-full ${score < 0.8 ? "bg-amber-500" : "bg-forest-700"}`} style={{ width: `${score * 100}%` }} /></div></div>)}</div></article>
        <article className="panel p-7 sm:p-9"><div className="flex items-center gap-3"><Microscope className="text-forest-700" /><h2 className="text-2xl font-black text-forest-900">Confusion matrix</h2></div><div className="mt-7 overflow-x-auto"><table className="mx-auto border-separate border-spacing-1 text-center text-xs"><thead><tr><th className="p-2" /><th colSpan={6} className="p-2 text-slate-500">Predicted</th></tr><tr><th className="p-2 text-slate-500">True</th>{labels.map((label) => <th className="p-2" key={label}>{label}</th>)}</tr></thead><tbody>{matrix.map((row, rowIndex) => <tr key={labels[rowIndex]}><th className="p-2">{labels[rowIndex]}</th>{row.map((value, columnIndex) => <td className={`size-10 rounded-lg ${rowIndex === columnIndex ? "bg-forest-700 font-bold text-white" : value ? "bg-amber-100 text-amber-900" : "bg-slate-50 text-slate-300"}`} key={columnIndex}>{value}</td>)}</tr>)}</tbody></table></div><p className="mt-5 text-xs leading-5 text-slate-500">C cardboard · G glass · M metal · Pa paper · Pl plastic · T trash</p></article>
      </div>
      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <article className="panel p-7"><Database className="text-forest-700" /><h2 className="mt-5 text-xl font-black text-forest-900">Dataset controls</h2><ul className="mt-4 space-y-3 text-sm leading-6 text-slate-600"><li>2,521 usable images after conflict removal</li><li>1,765 / 378 / 378 split</li><li>Seed 42 and recorded manifest</li><li>Zero dHash≤4 pairs crossing splits</li></ul></article>
        <article className="panel p-7"><FlaskConical className="text-forest-700" /><h2 className="mt-5 text-xl font-black text-forest-900">Model comparison</h2><div className="mt-4 space-y-4 text-sm"><div className="rounded-2xl bg-forest-50 p-4"><span className="font-bold text-forest-900">Group-aware split</span><p className="mt-1 text-slate-600">92.33% accuracy · primary result</p></div><div className="rounded-2xl bg-slate-50 p-4"><span className="font-bold text-slate-700">Ordinary random split</span><p className="mt-1 text-slate-500">91.03% accuracy · rejected for 39 cross-split near-duplicate pairs</p></div></div></article>
        <article className="rounded-3xl border border-amber-200 bg-amber-50 p-7"><AlertTriangle className="text-amber-700" /><h2 className="mt-5 text-xl font-black text-amber-950">Known limitation</h2><p className="mt-4 text-sm leading-6 text-amber-900">TrashNet is small, imbalanced, and mostly photographed on white backgrounds. Performance on mixed waste and real collection scenes may be lower. External-photo validation is still required.</p></article>
      </div>
      <div className="mt-6 flex items-start gap-3 rounded-3xl border border-emerald-200 bg-emerald-50 p-6 text-sm leading-6 text-emerald-900"><CheckCircle2 className="mt-0.5 shrink-0" size={20} /><p><strong>Reproducibility:</strong> model weights, preprocessing, class mapping, seed, training history, classification report, and confusion matrix are saved with the run artifacts.</p></div>
    </section>
  );
}
