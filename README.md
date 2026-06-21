# DooMoo (Black Pig Project)

A **contactless, smartphone-based system** that estimates pig weight from a single photo — built for remote farming communities in northern Thailand.

---

## The Problem

Karen and Muser farmers in Chiang Mai raise black pigs for ceremonies, food, and income. A single 2–3 month old piglet sells for **2,000–3,500 baht**, making accurate weight measurement critical for fair pricing.

But in remote mountain villages, weighing scales are rare — they're expensive, hard to transport, and pigs resist being handled. Farmers rely on visual judgment or bamboo sticks, which are inconsistent. Young piglets are especially sensitive: excessive handling causes stress, stunts growth, and can even lead mothers to reject them.

There is no reliable, accessible, and stress-free way to weigh a pig.

---

## The Solution

A mobile app that turns a smartphone into a pig weighing tool — **no scale, no contact, no internet**.

1. **Take a photo** of the pig from above
2. AI isolates the pig from the background and measures its body length
3. Physical length is calculated using camera geometry and a simple calibration method (laser tool or reference object)
4. A regression model predicts weight from the measured length

The farmer gets a reliable weight estimate in seconds — using only their phone.

---

## Why It Matters

- **For the farmer** — fair pricing, better income, data-driven decisions on feeding and selling
- **For the pig** — no stress, no handling, no risk of injury or maternal rejection
- **For the community** — accessible technology that works offline on basic Android phones, no expensive equipment needed

---

## Key Features

- **Fully offline** — works in mountain villages with no internet
- **Runs on low-end phones** — Helio G85 / Snapdragon 680+, 3–4 GB RAM
- **Two calibration methods** — external measurement tool (ex. laser meter) or a reference pig estimated length
- **AI-powered** — computer vision for pig detection + segmentation, ML for weight prediction
- **Built for farmers** — simple interface, step-by-step guidance, results in Thai language

---

## How It Works

```
Smartphone photo → Pig detection → Segmentation → Length measurement
                                                       ↓
                                              Weight prediction (kg)
                                                       ↓
                                              Result displayed in app
```

---

## Repository Structure

```
├── DooMoo/             # Flutter Android app (the mobile application)
├── pig_segmentation/   # AI model training notebooks
├── measure_length/     # Measurement algorithms
├── transform_model/    # Weight regression model
├── depth-estimation/   # Depth estimation experiments
└── transform/          # Perspective correction
```

---

## Technical Stack

| Layer | Technology |
|---|---|
| Mobile app | Flutter / Dart |
| Detection | YOLOv8 (ONNX) |
| Segmentation | RF-DETR (ONNX) |
| Weight model | Linear regression |
| All AI runs on-device | No server, no cloud |

---
