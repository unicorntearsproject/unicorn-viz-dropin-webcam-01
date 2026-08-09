# Third-party model — `selfie_segmenter.tflite`

| | |
| --- | --- |
| **Source** | Google MediaPipe — Image Segmenter, selfie-segmentation task, `float16` variant |
| **Origin URL** | `https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite` |
| **Model Card** | `https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf` |
| **Authors** | Tingbo Hou, Siargey Pisarchyk, Karthik Raveendran (Google), May 6, 2021 |
| **License** | Apache License, Version 2.0 |
| **Copyright** | Google LLC |

## Why this file exists

Unlike every other optional drop-in dependency in the project (see
`docs/audits/2026-08-08-licensing-audit.md`), this is a binary asset
committed directly to this repository rather than a package the operator
installs themselves. Apache 2.0 permits redistribution — including
binary redistribution — freely, provided the license and copyright notice
travel with the file. This notice satisfies that.

Apache 2.0 imposes no share-alike or field-of-use restriction, so
committing this file carries no obligation beyond attribution and no
constraint on how unicorn-viz itself is licensed or distributed —
unlike, for example, the Demucs pretrained weights used by
`dj-mixer-01`, which are CC BY-NC 4.0 and restrict *use* (see Finding 9
in the licensing audit).

The `mediapipe` Python package (Apache 2.0) that loads this model is a
separate, operator-installed optional dependency declared in this
drop-in's own `requirements.txt` — not bundled, not committed, and
covered by the audit's existing advisory-table treatment for optional
drop-in dependencies.

Full Apache License 2.0 text: https://www.apache.org/licenses/LICENSE-2.0
