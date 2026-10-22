# EcoVision AI

EcoVision AI is a local-first, full-stack recycling assistant for Hong Kong and
mainland China. It combines a reproducible 12-class, multi-source training
pipeline, a FastAPI inference service, Grad-CAM explanations, an English
recycling knowledge base, an optional DeepSeek integration, and a responsive
Next.js interface. The production model uses ImageNet transfer learning with a
resource-conscious MobileNetV3 backbone.

For the current Chinese build, training, architecture, frontend, DeepSeek, and
delivery record, see [`PROJECT_BUILD_GUIDE_12CLASS.md`](PROJECT_BUILD_GUIDE_12CLASS.md).
The older [`PROJECT_BUILD_GUIDE.md`](PROJECT_BUILD_GUIDE.md) remains as the
historical six-class baseline record.

## Included deliverables

- reproducible multi-source curation, validation, de-duplication, grouped splitting, training, and evaluation
- 12 useful waste categories plus an explicit `unknown` rejection state
- temperature-calibrated confidence and a validation-selected acceptance gate
- 89.71% held-out test accuracy and 88.82% macro-F1 on the group-aware split
- 90.92% accuracy across the 97.67% of test images accepted by the confidence gate
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

## Classes

`paper`, `cardboard`, `glass`, `metal`, `plastic_container`,
`plastic_bag_film`, `food_organic`, `battery`, `electronic_waste`, `textile`,
`foam_disposable_tableware`, and `general_waste`.

`unknown` is not a trained thirteenth class. It is returned when the calibrated
top probability or the margin over the runner-up is too small.

## Prepare the data

The v2 pipeline streams a bounded sample from source archives, resizes selected
images to at most 384 px, and avoids fully expanding the source ZIP. The exact
source mappings and commands used for the delivered model are documented in
[`PROJECT_BUILD_GUIDE_12CLASS.md`](PROJECT_BUILD_GUIDE_12CLASS.md).

```bash
python -m ecovision_ml.prepare_multisource \
  --garbage12-zip /path/to/garbage_classification.zip \
  --trashnet-dir data/raw/dataset-resized \
  --additional-dir /path/to/curated-specialist-images \
  --curated-dir /path/to/temporary-curated-images \
  --output-dir /path/to/prepared-manifest \
  --max-per-source-class 800
```

Preparation validates and normalizes images, records provenance, removes exact
duplicates and label conflicts, groups same-source object crops and same-label
near-duplicates, then produces a seeded group-aware 70/15/15 split.

## Train and evaluate

```bash
python -m ecovision_ml.train \
  --manifest /path/to/prepared-manifest/manifest.csv \
  --output artifacts/ecovision_12class_v1 \
  --architecture mobilenet_v3_large \
  --epochs 12 \
  --freeze-epochs 2 \
  --batch-size 8 \
  --workers 0 \
  --cpu-threads 2
```

The best checkpoint is selected on validation macro-F1 with early stopping. The
validation logits fit a scalar temperature and acceptance thresholds; the test
set is used once afterward. Outputs include the checkpoint, class mapping,
audit manifest without local image paths, training history, test metrics,
rejection metrics, classification report, and confusion matrix.

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

The delivered 12-class model combines a bounded selection from the public
Garbage Classification dataset, the existing TrashNet copy, TACO object crops,
and a bounded CC BY 4.0 GIZ E-Waste Database sample. See
[`PROJECT_BUILD_GUIDE_12CLASS.md`](PROJECT_BUILD_GUIDE_12CLASS.md) for URLs,
licenses, mappings, counts, limitations, and reproducibility details. Training
images are not committed; the trained weights and non-image audit artifacts are.
