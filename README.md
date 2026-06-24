# PPE Detection and Safety Monitoring

## Introduction

The following implementation is developed and tested on Windows.

PPE Detection and Safety Monitoring is a Python-based computer vision application that uses YOLO models for:

* Worker detection and tracking
* Personal Protective Equipment (PPE) compliance monitoring
* Safety violation reporting
* Worker counting
* Unsafe proximity detection between workers and machinery/vehicles
* Automatic violation snapshot generation

The system processes a video file, tracks individual workers using ByteTrack, checks PPE compliance, and generates detailed CSV reports for further analysis.

***The models used are pre-trained models***

---

## Features

### Worker Detection and Tracking

* Detects workers in video streams.
* Assigns a unique tracking ID to each worker using ByteTrack.
* Tracks workers across frames.

### PPE Compliance Monitoring

Detects:

* Hardhat (Helmet)
* Safety Vest

For each tracked worker, the system determines:

* Helmet Present / Missing
* Safety Vest Present / Missing

### Violation Detection

Generates violations when:

* Helmet is missing
* Safety Vest is missing

Violation events are logged and saved to reports.

### Unsafe Proximity Detection

Detects workers operating too close to:

* Machinery
* Vehicles

When a worker is within the configured proximity threshold, an unsafe proximity event is recorded.

### Automatic Snapshot Capture

When a violation is detected, the system automatically saves a snapshot image containing:

* Timestamp
* Worker ID
* Violation Type

Snapshots are stored in the `violations/` directory.

### Report Generation

The system automatically generates CSV reports containing:

* Worker counts
* PPE compliance status
* Safety violations
* Unsafe proximity events

---

## Prerequisites

* Python 3.10 or higher
* Windows OS
* NVIDIA GPU (recommended for real-time performance)

---

## Installation

Create and activate the virtual environment:

```bash
.\setup_env.bat
.\ppedet\Scripts\activate
```

---

## Model Files

Place the trained models in the following locations:

```text
weights/
├── ppe.pt # download from https://github.com/Vinayakmane47/PPE_detection_YOLO/tree/main/YOLO-Weights
└── person.onnx
```

Where:

* `ppe.pt` → PPE detection model
* `person.onnx` → Person detection and tracking model

---

## Usage

Place the input video as:

```text
test.mp4
```

Run the application:

```bash
python main.py
```

Press **ESC** at any time to stop processing.

---

## Output

### Live Monitoring Window

The application displays:

* Worker bounding boxes
* Worker IDs
* Helmet status
* Vest status
* Total worker count
* Unsafe proximity alerts

Color coding:

* Green → PPE compliant
* Red → PPE violation detected

---

### Violation Snapshots

Snapshots are automatically saved in:

```text
violations/
```

Example:

```text
violations/
├── 12_3_No_Helmet.jpg
├── 18_5_No_Vest.jpg
```

---

### Reports

Generated in:

```text
reports/
```

#### Worker Count Report

```text
worker_count.csv
```

Contains:

| timestamp | worker_count |
| --------- | ------------ |
| 1.25      | 4            |
| 1.29      | 4            |

---

#### PPE Compliance Report

```text
ppe_compliance.csv
```

Contains:

| timestamp | worker_id | helmet | vest  |
| --------- | --------- | ------ | ----- |
| 5.23      | 3         | True   | False |

---

#### Violation Report

```text
violations.csv
```

Contains:

| timestamp | worker_id | violation |
| --------- | --------- | --------- |
| 5.23      | 3         | No Vest   |

---

#### Unsafe Proximity Report

```text
unsafe_proximity.csv
```

Contains:

| timestamp | worker_id | event            |
| --------- | --------- | ---------------- |
| 7.15      | 2         | Unsafe Proximity |

---

## Configuration

The following parameters can be modified directly in `main.py`:

```python
SCALE = 4
CONF_PERSON = 0.5
CONF_PPE = 0.3
PROXIMITY_THRESHOLD = 150
```

| Parameter           | Description                                    |
| ------------------- | ---------------------------------------------- |
| SCALE               | Video downscaling factor                       |
| CONF_PERSON         | Person detection confidence threshold          |
| CONF_PPE            | PPE detection confidence threshold             |
| PROXIMITY_THRESHOLD | Distance threshold for unsafe proximity alerts |

---

## PPE Classes

The PPE model supports detection of:

* Hardhat
* Mask
* NO-Hardhat
* NO-Mask
* NO-Safety Vest
* Person
* Safety Cone
* Safety Vest
* Machinery
* Vehicle

Current compliance checks are performed using:

* Hardhat
* Safety Vest

---

## Directory Structure

```text
project/
├── main.py
├── test.mp4
├── weights/
│   ├── ppe.pt
│   └── person.onnx
├── violations/
├── reports/
└── setup_env.bat
```

