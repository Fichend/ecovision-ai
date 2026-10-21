"use client";

import { Camera, FileImage, ImagePlus, LoaderCircle, Trash2, UploadCloud, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import { classifyImage } from "@/lib/api";
import { cameraErrorMessage, fitCameraFrame } from "@/lib/camera";
import { saveAnalysis } from "@/lib/storage";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_BYTES = 10 * 1024 * 1024;

export default function RecognizePage() {
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const cameraStream = useRef<MediaStream | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);

  useEffect(() => () => { if (preview?.startsWith("blob:")) URL.revokeObjectURL(preview); }, [preview]);

  useEffect(() => {
    if (!cameraOpen) return;
    let active = true;
    setCameraLoading(true);
    setCameraReady(false);

    void navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    }).then(async (stream) => {
      if (!active) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      cameraStream.current = stream;
      if (video.current) {
        video.current.srcObject = stream;
        await video.current.play();
        setCameraReady(true);
      }
    }).catch((caught: unknown) => {
      if (active) {
        setError(cameraErrorMessage(caught));
        setCameraOpen(false);
      }
    }).finally(() => {
      if (active) setCameraLoading(false);
    });

    return () => {
      active = false;
      cameraStream.current?.getTracks().forEach((track) => track.stop());
      cameraStream.current = null;
      if (video.current) video.current.srcObject = null;
    };
  }, [cameraOpen]);

  function selectFile(candidate?: File) {
    setError(null);
    if (!candidate) return;
    if (!ACCEPTED_TYPES.includes(candidate.type)) return setError("Choose a JPEG, PNG, or WebP image.");
    if (candidate.size > MAX_BYTES) return setError("The image must be 10 MB or smaller.");
    if (preview?.startsWith("blob:")) URL.revokeObjectURL(preview);
    setFile(candidate);
    setPreview(URL.createObjectURL(candidate));
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files[0]);
  }

  function openCamera() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Live camera is unavailable in this browser. Open EcoVision AI in Safari or Chrome, or choose an existing image.");
      return;
    }
    setCameraOpen(true);
  }

  function capturePhoto() {
    const source = video.current;
    if (!source || !source.videoWidth || !source.videoHeight) {
      setError("The camera is not ready yet. Wait a moment and try again.");
      return;
    }

    const dimensions = fitCameraFrame(source.videoWidth, source.videoHeight);
    const canvas = document.createElement("canvas");
    canvas.width = dimensions.width;
    canvas.height = dimensions.height;
    const context = canvas.getContext("2d");
    if (!context) {
      setError("The camera frame could not be captured.");
      return;
    }
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setError("The camera frame could not be captured.");
        return;
      }
      const captured = new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" });
      selectFile(captured);
      setCameraOpen(false);
    }, "image/jpeg", 0.9);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const result = await classifyImage(file);
      const imageDataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(new Error("Could not read the selected image."));
        reader.readAsDataURL(file);
      });
      await saveAnalysis({ result, imageDataUrl, fileName: file.name });
      router.push("/results");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  function clear() {
    if (preview?.startsWith("blob:")) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null); setError(null);
  }

  return (
    <section className="page-shell py-14 sm:py-20">
      <div className="mx-auto max-w-4xl text-center">
        <p className="eyebrow">Image recognition</p>
        <h1 className="mt-4 text-4xl font-black tracking-tight text-forest-900 sm:text-5xl">What are you holding?</h1>
        <p className="mx-auto mt-4 max-w-2xl leading-7 text-slate-600">Use one clear item, good lighting, and a simple background when possible. The model recognizes visual categories, not chemical composition.</p>
      </div>

      <div className="panel mx-auto mt-10 max-w-4xl overflow-hidden p-5 sm:p-8">
        {!preview ? (
          <div
            className={`grid min-h-[360px] place-items-center rounded-3xl border-2 border-dashed p-8 text-center transition ${dragging ? "border-forest-500 bg-forest-50" : "border-forest-900/20 bg-cream/55"}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <div>
              <span className="mx-auto grid size-20 place-items-center rounded-full bg-forest-100 text-forest-800"><UploadCloud size={34} /></span>
              <h2 className="mt-6 text-2xl font-black text-forest-900">Drop an image here</h2>
              <p className="mt-2 text-sm text-slate-500">JPEG, PNG or WebP · maximum 10 MB</p>
              <div className="mt-7 flex flex-col justify-center gap-3 sm:flex-row">
                <button className="button-primary" onClick={() => fileInput.current?.click()}><ImagePlus size={18} /> Choose image</button>
                <button className="button-secondary" onClick={openCamera}><Camera size={18} /> Use camera</button>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid gap-7 lg:grid-cols-[1fr_0.7fr]">
            <div className="overflow-hidden rounded-3xl bg-slate-100"><img className="h-full max-h-[520px] w-full object-contain" src={preview} alt="Selected waste item preview" /></div>
            <div className="flex flex-col justify-between rounded-3xl bg-forest-900 p-7 text-white">
              <div><FileImage className="text-forest-300" /><p className="mt-6 text-xs font-bold uppercase tracking-wider text-forest-300">Ready to analyze</p><h2 className="mt-2 break-words text-xl font-black">{file?.name}</h2><p className="mt-3 text-sm leading-6 text-forest-100">The model will return the top three categories and a Grad-CAM attention map.</p></div>
              <div className="mt-8 grid gap-3"><button className="button-primary bg-white text-forest-900 hover:bg-forest-50" disabled={loading} onClick={analyze}>{loading ? <><LoaderCircle className="animate-spin" size={18} /> Analyzing…</> : <>Analyze image</>}</button><button className="inline-flex items-center justify-center gap-2 rounded-full px-4 py-3 text-sm font-bold text-forest-100 hover:bg-white/10" onClick={clear}><Trash2 size={17} /> Remove image</button></div>
            </div>
          </div>
        )}
        {error && <div className="mt-5 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm font-semibold text-rose-800" role="alert">{error}</div>}
      </div>
      {cameraOpen && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-forest-950/80 p-4" role="dialog" aria-modal="true" aria-labelledby="camera-title">
          <div className="w-full max-w-3xl overflow-hidden rounded-3xl bg-white shadow-2xl">
            <div className="flex items-center justify-between border-b border-forest-900/10 px-5 py-4">
              <div>
                <p className="eyebrow">Live camera</p>
                <h2 id="camera-title" className="mt-1 text-xl font-black text-forest-900">Center one item in the frame</h2>
              </div>
              <button className="grid size-10 place-items-center rounded-full text-slate-600 hover:bg-slate-100" onClick={() => setCameraOpen(false)} aria-label="Close camera"><X size={20} /></button>
            </div>
            <div className="relative aspect-[4/3] bg-slate-950">
              <video ref={video} className="size-full object-contain" autoPlay muted playsInline aria-label="Live camera preview" />
              {cameraLoading && <div className="absolute inset-0 grid place-items-center text-white"><span className="flex items-center gap-2"><LoaderCircle className="animate-spin" size={20} /> Starting camera…</span></div>}
            </div>
            <div className="flex flex-col-reverse gap-3 p-5 sm:flex-row sm:justify-end">
              <button className="button-secondary" onClick={() => setCameraOpen(false)}>Cancel</button>
              <button className="button-primary" disabled={!cameraReady} onClick={capturePhoto}><Camera size={18} /> Take photo</button>
            </div>
          </div>
        </div>
      )}
      <input ref={fileInput} className="hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event: ChangeEvent<HTMLInputElement>) => selectFile(event.target.files?.[0])} />
    </section>
  );
}
