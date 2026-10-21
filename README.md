# EcoVision AI

EcoVision AI is a local-first, full-stack recycling assistant for Hong Kong and
mainland China. It combines a reproducible six-class TrashNet training pipeline,
a FastAPI inference service, Grad-CAM explanations, an English recycling
knowledge base, an optional DeepSeek integration, and a responsive Next.js
interface. The model is MobileNetV2 with ImageNet transfer learning and the
classes `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`.

For the complete Chinese build, training, architecture, frontend, DeepSeek, and
delivery record, see [`PROJECT_BUILD_GUIDE.md`](PROJECT_BUILD_GUIDE.md).

## Included deliverables

- reproducible data validation, de-duplication, grouped splitting, training, and evaluation
- trained model artifacts with 92.33% grouped test accuracy and 90.11% macro-F1
- FastAPI health, image classification, Grad-CAM, and recycling-chat endpoints
- English guidance grounded in official Hong Kong and mainland China sources
- deterministic knowledge-base fallback until a DeepSeek API key is configured
- responsive recognition, results, assistant, and evaluation screens
- external-photo evaluation command for an independent real-world dataset
- backend and frontend automated tests plus a verified production build

## Environment

```bash
conda activate ecovision
python -m pip install -e .
```

## Prepare the data

Place the official TrashNet resized dataset at
`data/raw/dataset-resized/<class>/*.jpg`, then run:

```bash
python -m ecovision_ml.prepare_data \
  --source data/raw/dataset-resized \
  --output data/processed
```

Preparation validates every image, removes exact duplicate content, excludes
all copies of byte-identical images carrying conflicting labels, groups
same-label near-duplicates using a 64-bit dHash threshold, produces a seeded
group-aware split targeting 70/15/15, and records every sample in
`manifest.csv`.

## Train and evaluate

```bash
python -m ecovision_ml.train \
  --manifest data/processed/manifest.csv \
  --output artifacts/mobilenet_v2_baseline \
  --epochs 10 \
  --freeze-epochs 2 \
  --batch-size 32
```

The best checkpoint is selected on validation macro-F1. The test set is used
once after model selection. Outputs include class mapping, training history,
test metrics, a classification report, and a confusion matrix.

## Run the application locally

The API loads the trained checkpoint lazily and uses Apple MPS when available:

```bash
conda activate ecovision
cp .env.example .env
python -m uvicorn ecovision_api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
conda activate ecovision
cd frontend
pnpm install
pnpm dev
```

Open `http://127.0.0.1:3000`. Until `DEEPSEEK_API_KEY` is added to `.env`, the
assistant deliberately returns a labelled, deterministic response from the
curated knowledge base.

## Verify the project

From the project root:

```bash
conda activate ecovision
python -m pytest -q
python -m pip check
cd frontend
pnpm typecheck
pnpm test -- --run
pnpm build
```

## Real-world photo evaluation

Place unlabelled photos directly in `data/external`, or labelled photos under
folders such as `data/external/glass` and `data/external/paper`. Keep these
photos separate from TrashNet, then run:

```bash
python -m ecovision_ml.evaluate_external \
  --source data/external \
  --output artifacts/external_evaluation
```

The command saves every prediction and, when folder names are valid labels,
computes external accuracy and macro-F1. No real-world score is claimed until
independent photos are supplied.

## Dataset attribution

TrashNet was created by Gary Thung and Mindy Yang for a Stanford CS229 final
project. The official repository reports 2,527 resized images across six
classes and requests citation when the dataset is used:

- https://github.com/garythung/trashnet
- https://huggingface.co/datasets/garythung/trashnet

The source repository is MIT licensed. Dataset limitations include strong class
imbalance and mostly clean, white backgrounds, so reported test results should
not be interpreted as real-world recycling-bin performance.
