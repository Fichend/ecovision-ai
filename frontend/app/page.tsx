import { ArrowRight, BrainCircuit, Camera, ChartNoAxesCombined, ScanSearch, ShieldCheck } from "lucide-react";
import Link from "next/link";

const workflow = [
  { icon: Camera, step: "01", title: "Capture", text: "Upload a photo or use your device camera. JPEG, PNG, and WebP are supported." },
  { icon: ScanSearch, step: "02", title: "Classify", text: "A MobileNetV3 CNN compares 12 waste categories and rejects uncertain matches instead of forcing an answer." },
  { icon: BrainCircuit, step: "03", title: "Understand", text: "Grad-CAM shows model attention while the assistant grounds advice in official sources." }
];

export default function Home() {
  return (
    <>
      <section className="page-shell grid min-h-[78vh] items-center gap-12 py-20 lg:grid-cols-[1.05fr_0.95fr]">
        <div>
          <p className="eyebrow mb-5">Explainable recycling intelligence</p>
          <h1 className="max-w-3xl text-5xl font-black leading-[0.98] tracking-[-0.05em] text-forest-900 sm:text-6xl lg:text-7xl">
            See waste.<br />Understand the next move.
          </h1>
          <p className="mt-7 max-w-2xl text-lg leading-8 text-slate-600">
            EcoVision AI combines a trained image classifier, visual explanations, and source-grounded guidance for Hong Kong and mainland China.
          </p>
          <div className="mt-9 flex flex-col gap-3 sm:flex-row">
            <Link className="button-primary" href="/recognize">Analyze an item <ArrowRight size={18} /></Link>
            <Link className="button-secondary" href="/evaluation">Inspect model evidence</Link>
          </div>
        </div>
        <div className="relative mx-auto aspect-square w-full max-w-xl">
          <div className="absolute inset-0 rounded-[3rem] border border-forest-900/10 bg-forest-800" />
          <div className="absolute inset-7 grid place-items-center rounded-[2.25rem] border border-white/20 bg-forest-700 px-10 text-center text-white sm:inset-12">
            <div>
              <div className="mx-auto mb-7 grid size-24 place-items-center rounded-full border border-white/20 bg-white/10"><ScanSearch size={42} /></div>
              <p className="text-sm font-bold uppercase tracking-[0.2em] text-forest-100">Twelve waste categories</p>
              <p className="mt-3 text-3xl font-black">89.7% held-out accuracy</p>
              <p className="mt-3 text-sm leading-6 text-forest-100">Multi-source, group-aware test result. Calibrated confidence and an explicit unknown state reduce forced guesses; new real-world performance may be lower.</p>
            </div>
          </div>
          <div className="absolute -bottom-5 -left-2 rounded-2xl border border-forest-900/10 bg-white px-5 py-4 shadow-soft">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Device</span>
            <p className="mt-1 font-bold text-forest-900">Apple MPS verified</p>
          </div>
        </div>
      </section>

      <section className="border-y border-forest-900/10 bg-white/75 py-20">
        <div className="page-shell">
          <div className="mb-10 max-w-2xl">
            <p className="eyebrow">One transparent workflow</p>
            <h2 className="mt-3 text-3xl font-black tracking-tight text-forest-900 sm:text-4xl">From image to responsible action</h2>
          </div>
          <div className="grid gap-5 md:grid-cols-3">
            {workflow.map(({ icon: Icon, step, title, text }) => (
              <article className="panel p-7" key={step}>
                <div className="flex items-center justify-between"><span className="grid size-12 place-items-center rounded-2xl bg-forest-100 text-forest-800"><Icon /></span><span className="text-sm font-black text-forest-300">{step}</span></div>
                <h3 className="mt-7 text-xl font-black text-forest-900">{title}</h3>
                <p className="mt-3 text-sm leading-6 text-slate-600">{text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="page-shell grid gap-6 py-20 lg:grid-cols-2">
        <div className="panel p-8 sm:p-10">
          <ShieldCheck className="text-forest-700" size={30} />
          <h2 className="mt-6 text-3xl font-black text-forest-900">Designed around uncertainty</h2>
          <p className="mt-4 leading-7 text-slate-600">The assistant never silently replaces the CNN result. Confidence, alternative classes, source links, and limitations stay visible.</p>
        </div>
        <div className="panel p-8 sm:p-10">
          <ChartNoAxesCombined className="text-forest-700" size={30} />
          <h2 className="mt-6 text-3xl font-black text-forest-900">Evidence before claims</h2>
          <p className="mt-4 leading-7 text-slate-600">The evaluation page exposes class imbalance, per-class metrics, confusion patterns, source-domain bias, and the limits of a small specialist-class sample.</p>
        </div>
      </section>
    </>
  );
}
