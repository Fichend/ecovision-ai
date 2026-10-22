"use client";

import { Bot, ExternalLink, LoaderCircle, MapPin, Send, UserRound } from "lucide-react";
import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";
import { askAssistant } from "@/lib/api";
import { formatPercent, titleCase } from "@/lib/format";
import { loadAnalysis } from "@/lib/storage";
import type { ChatResponse, StoredAnalysis } from "@/lib/types";

type Message = { role: "user" | "assistant"; content: string };

export default function AssistantPage() {
  const [analysis, setAnalysis] = useState<StoredAnalysis | null | undefined>(undefined);
  const [location, setLocation] = useState<"hong_kong" | "mainland_china">("hong_kong");
  const [city, setCity] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<ChatResponse["sources"]>([]);
  const [provider, setProvider] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    void loadAnalysis().then((stored) => {
      if (active) setAnalysis(stored);
    });
    return () => { active = false; };
  }, []);
  if (analysis === undefined) return <div className="page-shell py-24 text-center">Loading assistant…</div>;
  if (!analysis) return <section className="page-shell py-24 text-center"><Bot className="mx-auto text-forest-700" size={40} /><h1 className="mt-5 text-4xl font-black text-forest-900">Analyze an item first</h1><p className="mt-3 text-slate-600">The assistant needs a structured CNN result to ground the conversation.</p><Link className="button-primary mt-7" href="/recognize">Open recognition</Link></section>;
  const result = analysis.result;
  const rejected = result.accepted === false || result.predicted_class === "unknown";

  async function submit(event: FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;
    const nextMessages = [...messages, { role: "user" as const, content: trimmed }];
    setMessages(nextMessages); setQuestion(""); setLoading(true); setError(null);
    try {
      const response = await askAssistant({ conversation_history: messages.slice(-8), classification: result, question: trimmed, location, city: location === "mainland_china" ? city || null : null });
      setMessages([...nextMessages, { role: "assistant", content: response.response }]);
      setSources(response.sources); setProvider(response.provider);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "The assistant failed to respond."); }
    finally { setLoading(false); }
  }

  return (
    <section className="page-shell py-14 sm:py-20">
      <div className="grid gap-6 lg:grid-cols-[0.72fr_1.28fr]">
        <aside className="space-y-6">
          <div className="panel p-6"><p className="eyebrow">Current context</p><div className="mt-5 flex gap-4"><img className="size-24 rounded-2xl bg-slate-100 object-cover" src={analysis.imageDataUrl} alt="Current classified item" /><div><h1 className="text-2xl font-black text-forest-900">{rejected ? "Unknown item" : titleCase(result.predicted_class)}</h1><p className="mt-1 text-sm text-slate-500">{formatPercent(result.confidence)} raw confidence</p><p className="mt-3 text-xs leading-5 text-slate-500">{rejected ? `The classifier rejected this result${result.raw_predicted_class ? `; its best visual match was ${titleCase(result.raw_predicted_class)}` : ""}. The assistant will ask for more detail before giving item-specific advice.` : "The assistant cannot overwrite this accepted CNN output."}</p></div></div></div>
          <div className="panel p-6"><label className="flex items-center gap-2 text-sm font-bold text-forest-900" htmlFor="location"><MapPin size={17} /> Guidance location</label><select id="location" className="field mt-3" value={location} onChange={(event) => setLocation(event.target.value as typeof location)}><option value="hong_kong">Hong Kong</option><option value="mainland_china">Mainland China</option></select>{location === "mainland_china" && <input className="field mt-3" value={city} onChange={(event) => setCity(event.target.value)} placeholder="City, e.g. Shenzhen" aria-label="Mainland China city" />}</div>
          {sources.length > 0 && <div className="panel p-6"><h2 className="font-black text-forest-900">Official sources</h2><div className="mt-4 space-y-3">{sources.map((source) => <a className="block rounded-2xl border border-forest-900/10 p-4 text-sm hover:bg-forest-50" key={source.url} href={source.url} target="_blank" rel="noreferrer"><span className="flex items-start justify-between gap-3 font-bold text-forest-800">{source.title}<ExternalLink size={15} /></span><span className="mt-1 block text-xs text-slate-500">{source.publisher}</span></a>)}</div></div>}
        </aside>
        <div className="panel flex min-h-[650px] flex-col overflow-hidden">
          <div className="border-b border-forest-900/10 p-6"><div className="flex items-center gap-3"><span className="grid size-11 place-items-center rounded-2xl bg-forest-800 text-white"><Bot size={21} /></span><div><h2 className="font-black text-forest-900">Recycling assistant</h2><p className="text-xs text-slate-500">{provider === "deepseek" ? "DeepSeek · curated sources" : provider ? "Knowledge-base fallback · add API key for DeepSeek" : "Source-grounded guidance"}</p></div></div></div>
          <div className="flex-1 space-y-5 overflow-y-auto bg-cream/50 p-6" aria-live="polite">
            {messages.length === 0 && <div className="mx-auto mt-16 max-w-md text-center"><Bot className="mx-auto text-forest-600" size={36} /><h3 className="mt-4 text-xl font-black text-forest-900">Ask what to do next</h3><p className="mt-2 text-sm leading-6 text-slate-500">Try “Should I rinse this?”, “Where can I take it?”, or “What if it contains mixed materials?”</p></div>}
            {messages.map((message, index) => <div className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`} key={`${message.role}-${index}`}><div className={`max-w-[82%] rounded-3xl p-4 text-sm leading-6 ${message.role === "user" ? "bg-forest-800 text-white" : "border border-forest-900/10 bg-white text-slate-700"}`}><span className="mb-2 flex items-center gap-2 text-xs font-bold uppercase tracking-wider opacity-70">{message.role === "user" ? <UserRound size={13} /> : <Bot size={13} />}{message.role}</span>{message.content}</div></div>)}
            {loading && <div className="flex items-center gap-2 text-sm text-slate-500"><LoaderCircle className="animate-spin" size={17} /> Checking grounded guidance…</div>}
          </div>
          <form className="border-t border-forest-900/10 p-4 sm:p-6" onSubmit={submit}><div className="flex gap-3"><input className="field" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about this item…" aria-label="Question for recycling assistant" /><button className="button-primary shrink-0 px-5" disabled={loading || !question.trim()} aria-label="Send question"><Send size={18} /></button></div>{error && <p className="mt-3 text-sm font-semibold text-rose-700" role="alert">{error}</p>}</form>
        </div>
      </div>
    </section>
  );
}
