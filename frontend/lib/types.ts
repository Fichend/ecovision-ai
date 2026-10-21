export type ConfidenceLevel = "high" | "medium" | "low";

export interface PredictionScore {
  label: string;
  probability: number;
}

export interface ClassificationResult {
  predicted_class: string;
  top_predictions: PredictionScore[];
  confidence: number;
  confidence_level: ConfidenceLevel;
  uncertainty_message: string;
  inference_time_ms: number;
  model_version: string;
  gradcam_available: boolean;
  gradcam_data_url: string | null;
}

export interface SourceReference {
  title: string;
  publisher: string;
  url: string;
}

export interface ChatResponse {
  response: string;
  sources: SourceReference[];
  uncertainty_information: string;
  provider: "deepseek" | "grounded_fallback";
  model: string | null;
}

export interface StoredAnalysis {
  result: ClassificationResult;
  imageDataUrl: string;
  fileName: string;
}
