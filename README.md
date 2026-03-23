## 🏙️ Smart City 2030: AI Digital Twin

An advanced AI-powered digital twin designed to simulate urban dynamics, optimize traffic flow, and monitor environmental impact. This project integrates geospatial engineering, reinforcement learning, and natural language processing to create a living model of a modern metropolis.

## 🚀 Overview

Smart City 2030 leverages real-time data and predictive modeling to solve three core urban challenges:

Congestion: Reducing travel time via RL-based traffic light synchronization.

Sustainability: Mapping pollution levels and predicting high-emission zones.

Efficiency: Forecasting energy demand and optimizing infrastructure placement.

## 🏗️ System architecture
The platform is divided into five specialized layers, ensuring a modular and scalable deployment.

1. Data Engineering (Spatial Backbone)
Role: Ingests and normalizes complex geospatial and temporal datasets.

Key Tech: PostGIS, GeoPandas, OSMNX.

Focus: Building ETL pipelines that transform raw OpenStreetMap data into graph structures for simulation.


## 📂  Data Description: Digital Twins

### 🚦 Mobility & infrastructure
* **`simulation_telemetry.csv`**: Time-series telemetry (15-min intervals). Contains `time_step`, `sensor_id`, and `speed_mph`.
* **`road_network_adjacency.csv`**: Adjacency matrix ($W$) representing physical road connectivity between sensors for Graph Neural Network (GNN) modeling.
* **`california_traffic_sensors.geojson`**: Geospatial coordinates and properties (Freeway, Direction) for Mapbox/Leaflet visualization.
* **`traffic_velocity_california.csv`**: The raw 2024 speed matrix used to generate the telemetry.

### 🌬️ Atmosphere & environment
* **`weather_hourly_dim.csv`**: Hourly meteorological data (Temp, Humidity, Wind Speed, Wind Direction). Use to join with telemetry on timestamps.
* **`ca_pollution_silver_2024.csv`**: Filtered 2024 EPA annual concentration baselines ($NO_2$, $PM_{2.5}$, Ozone) for California.
* **`sensor_pollutant_links_2024.csv`**: Haversine distance mapping between each traffic sensor and its nearest air quality monitor.
* **`california_air_monitor_markers.csv`**: Metadata for official EPA monitoring sites.

### 🆔 Metadata & mapping
* **`sensor_metadata.csv`**: Static sensor attributes including Freeway ID, Direction, and mapped 2024 $NO_2$ baseline. Join key: `sensor_id`.
* **`sensor_to_pollution_mapping.csv`**: Cross-reference file 


> **Team:** 1 Data Engineer · 1 Backend · 1 ML Engineer · 1 NLP Engineer · 1 Frontend  
> **Stack:** Kafka · Faust · FastAPI · TimescaleDB · PostGIS · PyTorch · Gymnasium · LangChain · BERT · React · Mapbox GL · Redis · Docker  

## Scope Summary

| Epic | Title | Owner | Days | Target |
|------|-------|--------|-------|------|--------|
| [EP-01](#ep-01--real-time-sensor-ingest) | Real-time sensor ingest | Data Eng + Backend | 5 days | Days 1–3 |
| [EP-02](#ep-02--live-traffic-heatmap) | Live traffic heatmap | Frontend + Backend | 4 days | Days 1–3 |
| [EP-03](#ep-03--traffic-light-simulation) | Traffic light simulation | ML Eng + Backend + Frontend | 5 days | Days 2–4 |
| [EP-04](#ep-04--24h-congestion-forecast) | 24h congestion forecast | ML Eng + Backend + Frontend | 4 days | Days 3–5 |
| [EP-05](#ep-05--nlp-query--bert-policy-engine) | NLP query + BERT policy engine | NLP Eng + Backend + Frontend | 6 days | Days 1–5 |
| [EP-06](#ep-06--pollution-overlay) | Pollution overlay | Frontend | 1 day | Day 5 |
| [EP-07](#ep-07--historical-simulation-playback) | Historical playback (hardcoded run) | Frontend + Backend | 2 days | Day 5 |
| [CC](#cc--jwt-auth-skeleton--docker--readme) | JWT skeleton + Docker + README | All | 2 days | Day 6 |

---

## Day-by-Day Plan

| Day | Date | Data Eng | Backend | ML Eng | NLP Eng | Frontend |
|-----|------|----------|---------|--------|---------|----------|
| 1 | Sat Mar 22 | Kafka/Redpanda + replay script | FastAPI skeleton + WebSocket | Synthetic data generation script — run overnight | BERT training data generation — 500 labelled examples via script | Mapbox canvas + road network |
| 2 | Sun Mar 23 | Faust processor + TimescaleDB + PostGIS | Node detail API | Gymnasium env (4 intersections) — LSTM training running overnight | BERT fine-tuning running overnight | WebSocket client + Zustand + heatmap colours |
| 3 | Mon Mar 24 | Done — support integration | Simulation API + JWT middleware skeleton | PPO training on 4-intersection env | SHAP scores wired to BERT + FastAPI endpoint | Road detail panel + simulation overlay |
| 4 | Tue Mar 25 | — | Forecast API + playback frames endpoint | LSTM inference endpoint | LangChain router wiring BERT to simulation + lookup | NLP chat panel + forecast chart |
| 5 | Wed Mar 26 | — | Integration fixes | Integration fixes | Integration fixes | Playback scrubber + pollution toggle + integration fixes |
| 6 | Thu Mar 27 | Docker Compose verify | Docker Compose verify | Docker Compose verify | Docker Compose verify | README + architecture diagram |

---

## Critical Path

```
Day 1:  Synthetic data generation script (ML) ──► overnight run produces 30 days of history
Day 2:  LSTM training starts (ML) ──────────────► overnight run
        BERT fine-tuning starts (NLP) ──────────► overnight run
Day 3:  LSTM model available for inference
        BERT model available for inference
        Simulation API ready (Backend) ─────────► unblocks Frontend simulation overlay
Day 4:  Forecast API ready (Backend) ───────────► unblocks Frontend forecast chart
        NLP endpoint ready (NLP) ───────────────► unblocks Frontend chat panel
Day 5:  All features integrated end to end
Day 6:  Ship
```

> **Biggest risk:** LSTM or BERT training does not converge overnight. Mitigation: if model performance is poor, ship it anyway with the limitation documented in the README. A real training pipeline with a weak model is more impressive than no training pipeline.

---

## EP-01 — Real-Time Sensor Ingest

**Goal:** Kafka receives sensor events, Faust validates and enriches them, TimescaleDB stores history, PostGIS holds current road state. Runs entirely in Docker.  
**Owner:** Data Engineer (lead) + Backend  
**Priority:** P0 — blocks every other epic  
**Status:** ❌ Not Started  
**Target:** Done by end of Day 3

| Story ID | Title | Owner | Days | Status |
|----------|-------|-------|------|--------|
| M-101 | Kafka + Faust local setup | Data Eng | 1 day | ❌ Not Started |
| M-102 | Sensor replay script | Data Eng | 1 day | ❌ Not Started |
| M-103 | Faust processor — validate + enrich | Data Eng | 1 day | ❌ Not Started |
| M-104 | TimescaleDB + PostGIS schema + writer | Data Eng | 1 day | ❌ Not Started |
| M-105 | WebSocket broadcast on ingest | Backend | 1 day | ❌ Not Started |

---

### M-101 — Kafka + Faust Local Setup

**Type:** Task | **Owner:** Data Engineer | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** None

**Description:**  
Stand up Redpanda (Kafka-compatible, single binary, no Zookeeper) via Docker Compose. Define topics. Confirm producer and consumer exchange messages. Faust app boots and connects.

> **Use Redpanda, not Kafka.** Kafka requires Zookeeper, a separate container, and significant configuration time. Redpanda is one container, API-compatible with Kafka, and starts in under 30 seconds. On a 6-day timeline this is not a preference, it is the correct engineering decision.

**Topics:**

| Topic | Purpose |
|-------|---------|
| `city.traffic.raw` | Raw traffic sensor events |
| `city.pollution.raw` | Raw pollution sensor events |
| `city.traffic.clean` | Validated, enriched events |
| `city.traffic.dlq` | Failed validation — dead letter queue |

```yaml
# docker-compose.yml excerpt
redpanda:
  image: redpandadata/redpanda:latest
  command: redpanda start --overprovisioned --smp 1 --memory 512M
  ports:
    - "9092:9092"
    - "9644:9644"
```

**Acceptance Criteria:**
- `docker compose up` starts broker with zero manual steps
- All topics created on startup via config file in `infra/redpanda/topics.yaml`
- Faust app boots and logs "Connected to broker" within 10 seconds
- `python ingest/test_producer.py` publishes 10 events, consumer reads all 10

**Success Criteria:** `docker compose up && python ingest/test_producer.py` runs clean on a fresh clone with zero errors.

---

### M-102 — Sensor Replay Script

**Type:** Task | **Owner:** Data Engineer | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-101

**Description:**  
Python script that reads a pre-committed sample CSV and publishes traffic and pollution events to Kafka at configurable replay speed. The CSV covers one real city district generated via OSMNX. Replay loops continuously so the demo never runs out of data. A high-speed mode (1440×) generates 30 days of synthetic history overnight for LSTM training.

**Generating the dataset:**
```python
import osmnx as ox
import pandas as pd
import numpy as np

G = ox.graph_from_place("Fitzrovia, London, UK", network_type="drive")
nodes, edges = ox.graph_to_gdfs(G)
edges['speed_kmh'] = np.random.normal(30, 15, len(edges)).clip(0, 120)
edges['occupancy_pct'] = np.random.normal(50, 25, len(edges)).clip(0, 100)
edges.to_csv('data/sample_traffic.csv')
```

**Temporal patterns for realistic training data:**
```python
def synthetic_occupancy(hour, day_of_week):
    base = 20
    if day_of_week < 5:
        if 7 <= hour <= 9:     base += 60  # morning peak
        elif 17 <= hour <= 19: base += 55  # evening peak
        elif 10 <= hour <= 16: base += 30  # daytime
    else:
        if 11 <= hour <= 15:   base += 25  # weekend midday
    return base + np.random.normal(0, 8)
```

**Acceptance Criteria:**
- `python ingest/replay.py --speed 10` runs at 10× real time, loops continuously
- `python ingest/generate_history.py --days 30 --speed 1440` generates 30 days of history into TimescaleDB overnight
- At least 200 unique road segments in sample dataset
- `data/sample_traffic.csv` and `data/sample_pollution.csv` committed — zero external downloads at demo time
- Pollution events include `no2_ugm3`, `pm25_ugm3`, `aqi`

**Success Criteria:** Replay runs for 10 minutes continuously with zero errors. History generation script completes overnight and produces 30 days of rows with realistic temporal patterns.

---

### M-103 — Faust Processor — Validate + Enrich

**Type:** Task | **Owner:** Data Engineer | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-101, M-102

**Description:**  
Faust agent validates each event against hard rules, enriches with canonical `road_id`, routes to clean topic or DLQ.

**Validation rules:**

| Field | Rule | Action |
|-------|------|--------|
| `speed_kmh` | 0–200 | Route to DLQ |
| `occupancy_pct` | 0–100 | Route to DLQ |
| Any required field | Not null | Route to DLQ |
| `timestamp` | Not > 60s in future | Route to DLQ |

**Acceptance Criteria:**
- `ingest/processor/app.py` consuming `city.traffic.raw` and `city.pollution.raw`
- Valid events → clean topic with `road_id` confirmed against PostGIS
- Invalid events → DLQ with `failed_rule` field appended
- Processing lag < 500ms
- `pytest ingest/tests/test_processor.py` — 10 valid pass, 5 invalid route to DLQ

**Success Criteria:** 100 replay events processed with correct routing. `pytest` passes clean.

---

### M-104 — TimescaleDB + PostGIS Schema + Writer

**Type:** Task | **Owner:** Data Engineer | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-103

**Description:**  
Hypertables in TimescaleDB for time-series history. Upsert current state into PostGIS on every clean event.

```sql
-- TimescaleDB
CREATE TABLE traffic_readings (
  time          TIMESTAMPTZ NOT NULL,
  road_id       TEXT NOT NULL,
  speed_kmh     FLOAT,
  vehicle_count INT,
  occupancy_pct FLOAT
);
SELECT create_hypertable('traffic_readings', 'time');
CREATE INDEX ON traffic_readings (road_id, time DESC);

CREATE TABLE pollution_readings (
  time      TIMESTAMPTZ NOT NULL,
  road_id   TEXT NOT NULL,
  no2_ugm3  FLOAT,
  pm25_ugm3 FLOAT,
  aqi       INT
);
SELECT create_hypertable('pollution_readings', 'time');
CREATE INDEX ON pollution_readings (road_id, time DESC);

-- PostGIS
CREATE TABLE road_nodes (
  road_id       TEXT PRIMARY KEY,
  geom          GEOMETRY(LineString, 4326),
  speed_kmh     FLOAT,
  occupancy_pct FLOAT,
  no2_ugm3      FLOAT,
  aqi           INT,
  last_updated  TIMESTAMPTZ
);
CREATE INDEX ON road_nodes USING GIST (geom);
```

**Acceptance Criteria:**
- Migrations in `db/migrations/` run automatically on `docker compose up`
- Road network seeded from `db/seeds/roads.sql` on first start
- Faust writer flushes to TimescaleDB every 500ms via asyncpg batch insert
- PostGIS upserted via `ON CONFLICT (road_id) DO UPDATE` on every clean event
- After 60s of replay: `SELECT COUNT(*) FROM traffic_readings` > 500 rows

**Success Criteria:** Both databases have live data 60 seconds after `docker compose up`.

---

### M-105 — WebSocket Broadcast on Ingest

**Type:** Task | **Owner:** Backend | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-104

**Description:**  
FastAPI WebSocket endpoint broadcasts road update payload to all connected clients on every PostGIS upsert.

**Payload:**
```json
{
  "type": "road_update",
  "road_id": "way/12345",
  "speed_kmh": 23.4,
  "occupancy_pct": 67.2,
  "no2_ugm3": 41.2,
  "aqi": 87,
  "last_updated": "2026-03-22T09:14:22Z"
}
```

**Acceptance Criteria:**
- Endpoint at `ws://localhost:8000/ws/city-feed`
- No auth on WebSocket — JWT applied to REST endpoints only (CC-01)
- Broadcasts within 200ms of PostGIS upsert
- Handles 5 concurrent connections without dropping events
- Frontend auto-reconnects on disconnect with exponential backoff: 1s → 2s → 4s → max 30s

**Success Criteria:** Three browser tabs all receive the same road update within 500ms of Kafka publish.

---

---

## EP-02 — Live Traffic Heatmap

**Goal:** Mapbox GL map renders the demo district. Roads colour in real time from WebSocket events. Road click opens detail panel with live metrics and action buttons wiring to all other features.  
**Owner:** Frontend (lead) + Backend  
**Priority:** P0  
**Status:** ❌ Not Started  
**Target:** Done by end of Day 3

| Story ID | Title | Owner | Days | Status |
|----------|-------|-------|------|--------|
| M-201 | Mapbox GL canvas + road network | Frontend | 1 day | ❌ Not Started |
| M-202 | WebSocket client + Zustand store | Frontend | 1 day | ❌ Not Started |
| M-203 | Heatmap colour layer | Frontend | 1 day | ❌ Not Started |
| M-204 | Road node detail panel | Frontend + Backend | 1 day | ❌ Not Started |

---

### M-201 — Mapbox GL Canvas + Road Network

**Type:** Task | **Owner:** Frontend | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-104

**Acceptance Criteria:**
- React + TypeScript + Vite app in `frontend/src/`
- Mapbox GL JS in `src/components/CityMap.tsx`
- `GET /api/roads` queries PostGIS ST_MakeEnvelope on demo bbox, returns GeoJSON FeatureCollection
- Roads render as thin gray lines on Mapbox Dark v11
- Map centred on demo district — hardcoded bbox
- Mapbox token from `VITE_MAPBOX_TOKEN` env var — `.env.example` committed

**Success Criteria:** `docker compose up` → `localhost:3000` → road network visible within 3 seconds, zero console errors.

---

### M-202 — WebSocket Client + Zustand Store

**Type:** Task | **Owner:** Frontend | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-201, M-105

**Why Zustand not Redux:** Events arrive at 50+/sec. Zustand's partial merge is O(1) per update. Redux action dispatch cycle creates unnecessary overhead at this event frequency.

**Acceptance Criteria:**
- Hook in `src/hooks/useRoadFeed.ts` — connects on mount, cleans up on unmount
- Store in `src/store/roadStore.ts` — shape: `Record<string, RoadUpdate>`
- Partial merge per event: `set(s => ({ roads: { ...s.roads, [e.road_id]: e } }))`
- Connection status dot: green / amber / red — top right corner of map
- Reconnect on disconnect: 1s → 2s → 4s → max 30s backoff

**Success Criteria:** 50 events/sec for 60 seconds with no memory growth. No full tree re-renders confirmed in React DevTools.

---

### M-203 — Heatmap Colour Layer

**Type:** User Story | **Owner:** Frontend | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-202

**User Story:** As a reviewer opening the demo, I want to see roads visually changing colour in real time so the system's core value is immediately obvious.

**Colour scale:**

| Occupancy | Colour | Meaning |
|-----------|--------|---------|
| 0–30% | `#1D9E75` | Free flow |
| 31–60% | `#EF9F27` | Moderate |
| 61–80% | `#D85A30` | Heavy |
| 81–100% | `#E24B4A` | Gridlock |
| Stale > 60s | `#444441` | Unknown |

**Acceptance Criteria:**
- Mapbox `line` layer colours from Zustand store via GeoJSON expression
- Only changed feature updated via `setData()` — never full GeoJSON replacement
- Line width: 2px city zoom, 5px street zoom via zoom expression
- Stale roads fade to gray after 60s
- Colour transitions: 300ms smooth via `line-color-transition`

**Success Criteria:** Roads visibly change colour within 500ms of replay publishing. Map looks alive within 10 seconds.

---

### M-204 — Road Node Detail Panel

**Type:** User Story | **Owner:** Frontend + Backend | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-203

**User Story:** As a reviewer, I want to click any road and see its live metrics so I can verify the full data pipeline end to end.

**Panel contents:**
1. Road name + last updated ("2s ago")
2. Speed (km/h), vehicle count, occupancy % with colour badge
3. NO2, PM2.5, AQI with WHO category label
4. Action buttons: "Run simulation" → EP-03 · "Forecast" → EP-04 · "Ask AI" → EP-05

**Backend:** `GET /api/node/:road_id` — single PostGIS row lookup, < 50ms.

**Acceptance Criteria:**
- Panel opens as right-side drawer on road click, closes on Escape or X
- Reads from Zustand store first — instant render, no API latency on basic metrics
- Backend called only for road name and OSM metadata not in store
- All action buttons pre-fill their target panels with current `road_id`

**Success Criteria:** Panel fully populated within 200ms of road click.

---

---

## EP-03 — Traffic Light Simulation

**Goal:** Gymnasium environment models 4 real intersections. PPO agent returns optimal timing plan. Result overlays on map showing per-intersection wait time and emission deltas.  
**Owner:** ML Engineer (lead) + Backend + Frontend  
**Priority:** P0  
**Status:** ❌ Not Started  
**Target:** Done by end of Day 4

**Scope decision:** 4 intersections trains in hours on CPU. Architecturally identical to a full city — only the graph query changes. This is the correct scope for 6 days.

| Story ID | Title | Owner | Days | Status |
|----------|-------|-------|------|--------|
| M-301 | Gymnasium environment — 4 intersections | ML Eng | 2 days | ❌ Not Started |
| M-302 | PPO agent train + ONNX export | ML Eng | 1 day | ❌ Not Started |
| M-303 | Simulation API endpoint | Backend | 1 day | ❌ Not Started |
| M-304 | Simulation result overlay on map | Frontend | 1 day | ❌ Not Started |

---

### M-301 — Gymnasium Environment — 4 Intersections

**Type:** Task | **Owner:** ML Engineer | **Priority:** P0 | **Estimate:** 2 days | **Depends on:** M-104

**State space:** 4 intersections × 4 approaches = 16 occupancy values + 4 current phases = 20-dimensional observation.  
**Action space:** `MultiDiscrete([4, 4, 4, 4])` — phase duration per intersection: 15s / 30s / 45s / 60s.  
**Reward:** `−mean(queue_lengths) − 0.1 × co2_estimate`

**Acceptance Criteria:**
- `ml/envs/city_traffic_env.py` inheriting `gymnasium.Env`
- `reset()` loads 4-intersection subgraph from PostGIS using hardcoded node IDs from demo district
- `step(action)` advances 30s simulation time, returns obs, reward, done, info
- Passes `gymnasium.utils.env_checker.check_env()` with no errors
- 1,000 random steps complete in < 5s on CPU
- Seeded with `np.random.seed(42)` for reproducible results

**Success Criteria:** `check_env()` passes. 1,000 steps in < 5s. No GPU required.

---

### M-302 — PPO Agent Train + ONNX Export

**Type:** Task | **Owner:** ML Engineer | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-301

**Acceptance Criteria:**
- `python ml/training/train_ppo.py` — no arguments
- Trains 50,000 timesteps — 1–3 hours on CPU
- MLflow experiment at `localhost:5000` — reward curve logged per episode
- Model exported to `ml/models/ppo_traffic_v1.onnx`
- `python ml/eval/eval_ppo.py` prints mean reward vs random baseline
- Agent beats random policy — if not converging after 2 hours, reduce env to 2 intersections

**Success Criteria:** Agent beats random baseline. ONNX inference < 100ms.

---

### M-303 — Simulation API Endpoint

**Type:** Task | **Owner:** Backend | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-302

**Cache key:** `sha256("sim:demo_district:{model_version}")`  
**Redis TTL:** 2 minutes

**Request:**
```json
POST /api/simulate/run
{ "area": "demo_district" }
```

**Response:**
```json
{
  "run_id": "uuid",
  "intersections": [
    { "road_id": "node/111", "current_phase_s": 60, "recommended_phase_s": 35, "wait_time_delta_s": -14.2 },
    { "road_id": "node/222", "current_phase_s": 60, "recommended_phase_s": 50, "wait_time_delta_s": -8.1 },
    { "road_id": "node/333", "current_phase_s": 60, "recommended_phase_s": 45, "wait_time_delta_s": -11.4 },
    { "road_id": "node/444", "current_phase_s": 60, "recommended_phase_s": 40, "wait_time_delta_s": -9.7 }
  ],
  "mean_wait_delta_s": -10.9,
  "emission_delta_pct": -8.3,
  "cached": false,
  "model_version": "ppo_v1"
}
```

**Acceptance Criteria:**
- ONNX model loaded once on startup — not per request
- Cache hit < 20ms, cache miss < 2s
- Result written to `db/simulation_runs` table for EP-07 playback
- Returns 503 with clear message if ONNX file missing — never crashes with 500

**Success Criteria:** First call < 2s. Second call < 20ms. Response matches schema exactly.

---

### M-304 — Simulation Result Overlay

**Type:** User Story | **Owner:** Frontend | **Priority:** P0 | **Estimate:** 1 day | **Depends on:** M-303, M-204

**Acceptance Criteria:**
- "Run simulation" in detail panel triggers endpoint with loading spinner
- 4 intersection nodes shown as coloured circles: green (wait reduced), red (increased)
- Summary: "Mean wait time reduced by 10.9s · Estimated −8.3% emissions"
- Hover tooltip: "Current: 60s → Recommended: 35s · −14.2s wait"
- Toggle: "Show optimised" / "Show current"

**Success Criteria:** Click road → "Run simulation" → see intersection markers in < 5 seconds total.

---

---

## EP-04 — 24h Congestion Forecast

**Goal:** LSTM model trained on 30 days of synthetic sensor history forecasts congestion up to 24h ahead with p10/p50/p90 confidence bands. This is the feature trading firms specifically evaluate.  
**Owner:** ML Engineer (lead) + Backend + Frontend  
**Priority:** P1  
**Status:** ❌ Not Started  
**Target:** Done by end of Day 5

> **How we get 30 days of training data in 6 days:** The replay script (M-102) generates synthetic readings with realistic temporal patterns. Run at 1440× speed overnight on Day 1 — 1 hour of compute produces 60 days of sensor history. LSTM training starts Day 2 morning.

| Story ID | Title | Owner | Days | Status |
|----------|-------|-------|------|--------|
| M-401 | Synthetic history generation — 30 days overnight | ML Eng | 0.5 days | ❌ Not Started |
| M-402 | LSTM model train + ONNX export | ML Eng | 1.5 days | ❌ Not Started |
| M-403 | Forecast API endpoint | Backend | 1 day | ❌ Not Started |
| M-404 | Forecast chart UI | Frontend | 1 day | ❌ Not Started |

---

### M-401 — Synthetic History Generation — 30 Days Overnight

**Type:** Task | **Owner:** ML Engineer | **Priority:** P1 | **Estimate:** 0.5 days | **Depends on:** M-102, M-104

**Description:**  
Extend the replay script to write 30 days of synthetic history into TimescaleDB at 1440× speed. Encode realistic morning/evening peaks so the LSTM has genuine patterns to learn. Start this at end of Day 1, runs overnight.

**Temporal pattern encoding:**
```python
def synthetic_occupancy(hour, day_of_week):
    base = 20
    if day_of_week < 5:  # weekday
        if 7 <= hour <= 9:     base += 60  # morning peak
        elif 17 <= hour <= 19: base += 55  # evening peak
        elif 10 <= hour <= 16: base += 30  # daytime moderate
    else:                    # weekend
        if 11 <= hour <= 15:   base += 25  # weekend midday
    return base + np.random.normal(0, 8)
```

**Acceptance Criteria:**
- `python ingest/generate_history.py --days 30 --speed 1440` runs and exits cleanly
- After run: `SELECT COUNT(DISTINCT date_trunc('day', time)) FROM traffic_readings` returns 30
- Morning and evening peaks visible in a time plot — patterns are non-trivial
- Completes before Day 2 standup

**Success Criteria:** 30 days of history in TimescaleDB by Day 2 morning with realistic temporal patterns confirmed visually.

---

### M-402 — LSTM Model Train + ONNX Export

**Type:** Task | **Owner:** ML Engineer | **Priority:** P1 | **Estimate:** 1.5 days | **Depends on:** M-401

**Description:**  
Train LSTM on 30-day synthetic history. Input: last 7 days of readings for a road. Output: 96 timesteps (24h at 15-min resolution) with p50, p10, p90 confidence bands.

**Input features per timestep:**
- `speed_kmh`, `occupancy_pct`, `vehicle_count`
- `sin(hour/24 × 2π)`, `cos(hour/24 × 2π)` — time of day encoding
- `sin(dow/7 × 2π)`, `cos(dow/7 × 2π)` — day of week encoding

**Architecture:**
```python
class CongestionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=128, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc_p50 = nn.Linear(128, 96)
        self.fc_p10 = nn.Linear(128, 96)
        self.fc_p90 = nn.Linear(128, 96)
```

**Acceptance Criteria:**
- `python ml/training/train_lstm.py` — no arguments
- 80/20 train/val split on 30-day history
- MLflow logs train loss, val loss, MAPE per epoch at `localhost:5000`
- Target MAPE ≤ 20% on val set (relaxed from production 15% — synthetic data)
- Exported to `ml/models/lstm_forecast_v1.onnx`
- Inference for one road < 200ms on CPU
- p90 must be meaningfully higher than p10 — non-trivial confidence bands

**Success Criteria:** Val MAPE ≤ 20%. Inference < 200ms. Confidence bands visually non-trivial.

---

### M-403 — Forecast API Endpoint

**Type:** Task | **Owner:** Backend | **Priority:** P1 | **Estimate:** 1 day | **Depends on:** M-402

**Cache key:**
```python
bucket = (int(time.time()) // 300) * 300  # floor to 5-min boundary
key = sha256(f"{road_id}:{horizon_hours}:{bucket}").hexdigest()
```

**Request:** `GET /api/forecast?road_id={id}&horizon=24`

**Response:**
```json
{
  "road_id": "way/12345",
  "horizon_hours": 24,
  "forecast": [
    { "timestamp": "2026-03-22T10:00:00Z", "p50": 62.3, "p10": 48.1, "p90": 74.5 },
    { "timestamp": "2026-03-22T10:15:00Z", "p50": 65.1, "p10": 51.2, "p90": 77.8 }
  ],
  "cached": false
}
```

**Acceptance Criteria:**
- TimescaleDB 7-day lookback < 100ms (index from M-104)
- Cache miss (DB + inference) < 1s
- Cache hit < 30ms
- Concurrent requests within same 5-min window share one inference call
- 404 if road has < 3 days of history with message: "Not enough history for this road yet"

**Success Criteria:** Cache miss < 1s. Cache hit < 30ms.

---

### M-404 — Forecast Chart UI

**Type:** User Story | **Owner:** Frontend | **Priority:** P1 | **Estimate:** 1 day | **Depends on:** M-403, M-204

**User Story:** As a reviewer from a trading firm, I want to see a 24h congestion forecast with confidence bands so I can verify the team understands time-series modelling beyond real-time streaming.

**Acceptance Criteria:**
- Panel opens from "Forecast" button in road detail panel (M-204)
- D3.js line chart: x-axis = 24h, y-axis = congestion score 0–100
- p50 in solid teal, p10/p90 as shaded confidence band
- Current time marked with vertical dashed line
- Horizon selector: 1h / 6h / 24h — cache hit on toggle < 100ms
- Loading skeleton — chart dimensions pre-set, no layout shift
- "Not enough data" state for roads with < 3 days history

**Success Criteria:** Chart renders within 500ms. Confidence band immediately clear to a reviewer. Horizon toggle feels instant on cache hit.

---

---

## EP-05 — NLP Query + BERT Policy Engine

**Goal:** LangChain classifies intent and extracts entities. What-if queries trigger simulation. Lookup queries hit TimescaleDB. Policy queries go through fine-tuned BERT with SHAP token-level explainability. This is the Anthropic signal.  
**Owner:** NLP Engineer (lead) + Backend + Frontend  
**Priority:** P1  
**Status:** ❌ Not Started  
**Target:** Done by end of Day 5

> **Why fine-tuned BERT, not just LangChain few-shot:** Few-shot prompting is wrapping someone else's model. It is a product engineering skill. Fine-tuned BERT with SHAP demonstrates understanding of training pipelines, model internals, and explainability methods — exactly what Anthropic evaluates. The two approaches are not equivalent signals.

| Story ID | Title | Owner | Days | Status |
|----------|-------|-------|------|--------|
| M-501 | BERT training data generation script | NLP Eng | 0.5 days | ❌ Not Started |
| M-502 | BERT fine-tuning + SHAP explainability | NLP Eng | 1.5 days | ❌ Not Started |
| M-503 | LangChain intent classifier + entity extraction | NLP Eng | 1 day | ❌ Not Started |
| M-504 | Query router + response composer | NLP Eng + Backend | 1 day | ❌ Not Started |
| M-505 | NLP chat panel UI | Frontend | 1 day | ❌ Not Started |

---

### M-501 — BERT Training Data Generation Script

**Type:** Task | **Owner:** NLP Engineer | **Priority:** P1 | **Estimate:** 0.5 days | **Depends on:** None

**Description:**  
Generate 500+ labelled training examples programmatically using templates with random substitutions. This is an afternoon of scripting, not manual annotation. Start overnight with BERT training at end of Day 1.

**Intervention categories:**
- `speed_limit_reduction`
- `lane_closure`
- `signal_retiming`
- `congestion_charge`
- `public_transport_increase`

**Generation approach:**
```python
templates = {
    "speed_limit_reduction": [
        "Reduce speed limit on {road} to {speed}mph",
        "Lower the speed limit near {area}",
        "Cut vehicle speeds on {road} during {time}",
    ],
    "lane_closure": [
        "Close {road} to traffic on {day}",
        "Shut the {direction} lane on {road}",
        "What if we remove a lane from {road}?",
    ],
    "signal_retiming": [
        "Adjust traffic light timing on {road}",
        "Change the signal phases at {intersection}",
        "Optimise the lights at {road} junction",
    ],
    "congestion_charge": [
        "Introduce a congestion charge on {road}",
        "Charge vehicles entering {area}",
        "Add a toll to {road} during peak hours",
    ],
    "public_transport_increase": [
        "Add more buses on the {road} route",
        "Increase train frequency through {area}",
        "Improve public transport near {road}",
    ],
}
roads = ["Main St", "Oxford Rd", "High Street", "Park Lane", "King St"]
# 100 examples per category via random template + substitution = 500 total
```

**Acceptance Criteria:**
- `python nlp/data/generate_training_data.py` produces `nlp/data/policy_labels.jsonl`
- At least 100 examples per category, balanced distribution
- Each record: `{ "text": str, "label": str }`
- Fixed random seed for reproducibility
- Runs in < 2 minutes

**Success Criteria:** 500+ examples in `policy_labels.jsonl`. Balanced distribution confirmed via label count.

---

### M-502 — BERT Fine-Tuning + SHAP Explainability

**Type:** Task | **Owner:** NLP Engineer | **Priority:** P1 | **Estimate:** 1.5 days | **Depends on:** M-501

**Description:**  
Fine-tune `bert-base-uncased` on the generated dataset. Integrate SHAP to produce token-level contribution scores for every prediction. Every API response includes which words drove the classification — this is what makes the NLP layer stand out to Anthropic reviewers.

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import shap

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5
)

training_args = TrainingArguments(
    output_dir="ml/models/bert_policy",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# SHAP explainability
explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(["Close Main St to traffic"])
# Returns token-level contributions to the classification decision
```

**Response schema from `POST /api/policy/classify`:**
```json
{
  "query": "Close Main St to traffic on weekdays",
  "prediction": "lane_closure",
  "confidence": 0.91,
  "shap_tokens": [
    { "token": "Close", "value": 0.34 },
    { "token": "traffic", "value": 0.28 },
    { "token": "weekdays", "value": 0.15 }
  ],
  "all_scores": {
    "lane_closure": 0.91,
    "signal_retiming": 0.05,
    "speed_limit_reduction": 0.02,
    "congestion_charge": 0.01,
    "public_transport_increase": 0.01
  }
}
```

**Acceptance Criteria:**
- `python nlp/training/train_bert.py` — no arguments, trains for 3 epochs
- 1–2 hours on CPU, ~20 min on GPU
- Validation accuracy ≥ 75% on held-out 20% split
- SHAP integrated — every inference returns `shap_tokens` with non-trivial values
- Model saved to `ml/models/bert_policy/`
- FastAPI endpoint at `POST /api/policy/classify`
- MLflow logs train loss, val accuracy per epoch

**Success Criteria:** Val accuracy ≥ 75%. Every response includes SHAP tokens. Inference < 500ms on CPU.

---

### M-503 — LangChain Intent Classifier + Entity Extraction

**Type:** Task | **Owner:** NLP Engineer | **Priority:** P1 | **Estimate:** 1 day | **Depends on:** None

**Description:**  
LangChain chain classifies free-text queries into routing intents and extracts road name and time window. Routes to BERT for policy, simulation API for what-if, TimescaleDB for lookups. Uses few-shot prompting — fast to build and handles open-vocabulary phrasing that a classifier would miss.

**Intent routing:**

| Intent | Routes to |
|--------|----------|
| `what_if_simulation` | Simulation API (EP-03) |
| `congestion_lookup` | TimescaleDB query |
| `policy_advice` | BERT classifier (M-502) |
| `unknown` | Graceful fallback message |

**Acceptance Criteria:**
- `nlp/chains/intent_classifier.py`
- Road name → `road_id` via PostGIS `similarity()` fuzzy match
- `unknown` never throws — always returns gracefully
- `pytest nlp/tests/test_classifier.py` — 5 queries per intent × 4 intents = 20 queries all correctly classified
- Works with Ollama if no API key: `ollama pull llama3.2:3b`
- Latency < 2s

**Success Criteria:** All 20 unit test queries correctly classified. No exceptions on any input.

---

### M-504 — Query Router + Response Composer

**Type:** Task | **Owner:** NLP Engineer + Backend | **Priority:** P1 | **Estimate:** 1 day | **Depends on:** M-502, M-503

**Routing logic:**
- `what_if_simulation` → `POST /api/simulate/run` → "Closing {road} would reduce mean wait time by {delta}s across {n} intersections"
- `congestion_lookup` → TimescaleDB 5-min average → "{road} is running at {speed}km/h with {occ}% occupancy — {WHO category}"
- `policy_advice` → `POST /api/policy/classify` → "Based on current conditions, {intervention} is recommended. Key factors: {top SHAP tokens}"
- `unknown` → "I can answer questions about traffic simulation, current congestion, and city policy. Try: 'What if we close Main St?' or 'How busy is Oxford Rd?'"

**Acceptance Criteria:**
- `POST /api/nlp/query` endpoint
- All three branches tested end to end before Day 5
- Response always includes `road_id` when resolved — frontend highlights road on map
- Policy response includes SHAP explanation as readable sentence
- SQL queries use parameterised statements — no string interpolation
- Ambiguous road name returns clarification prompt
- Latency: < 4s simulation path, < 500ms lookup, < 1s policy

**Success Criteria:** All three branches return complete grammatically correct answers. SHAP explanation appears in every policy response.

---

### M-505 — NLP Chat Panel UI

**Type:** User Story | **Owner:** Frontend | **Priority:** P1 | **Estimate:** 1 day | **Depends on:** M-504

**User Story:** As a reviewer from Anthropic, I want to type a question and see a plain-English answer with an explanation of the AI's reasoning — including which words drove the recommendation — so I can evaluate the depth of the NLP implementation.

**SHAP token visualisation:** Each token displayed as a coloured chip — teal for high positive contribution, gray for neutral, coral for negative. Contribution score shown on hover. Makes BERT explainability visually immediate.

**Acceptance Criteria:**
- Chat panel as left-side drawer, toggle button on map
- Loading skeleton while waiting — not spinner (shows confidence)
- `what_if` response: text + simulation overlay auto-triggered on map
- `policy_advice` response: text + SHAP token chips with contribution scores
- `congestion_lookup` response: text + road highlighted on map in blue
- `unknown` response: fallback text + 2 clickable example query chips
- Keyboard: Enter submits, Shift+Enter newline

**Success Criteria:** Full flow — type "What should we do about pollution near the school?" — see BERT prediction, confidence, SHAP chips, and plain English answer in < 2 seconds.

---

---

## EP-06 — Pollution Overlay (Basic)

**Goal:** Toggleable AQI layer reading from the existing WebSocket feed. No interpolation. One day frontend work.  
**Owner:** Frontend  
**Priority:** P2 | **Target:** Day 5 (parallel with integration)

| Story ID | Title | Owner | Days | Status |
|----------|-------|-------|------|--------|
| M-601 | Pollution layer + legend | Frontend | 1 day | ❌ Not Started |

---

### M-601 — Pollution Layer + Legend

**AQI colour scale (WHO):**

| AQI | Colour | Category |
|-----|--------|---------|
| 0–50 | `#1D9E75` | Good |
| 51–100 | `#EF9F27` | Moderate |
| 101–150 | `#D85A30` | Unhealthy (sensitive groups) |
| 151–200 | `#E24B4A` | Unhealthy |
| 200+ | `#534AB7` | Very unhealthy |

**Acceptance Criteria:**
- Second Mapbox `line` layer `roads-pollution` colours by `aqi` from Zustand store
- Layer control: Traffic and Pollution checkboxes — both simultaneously active
- Pollution layer at 70% opacity — road network visible underneath
- Legend shows AQI ramp with WHO category labels
- Road click while pollution active shows AQI + category in detail panel
- Toggle state persisted to `localStorage`

**Success Criteria:** Both layers composited correctly. Toggle < 100ms.

---

---

## EP-07 — Historical Simulation Playback (Hardcoded Run)

**Goal:** Playback scrubber loads the simulation result from EP-03 and plays it frame by frame on the map. No S3 infrastructure — reads directly from `db/simulation_runs`. Two days of work, not six.  
**Owner:** Frontend (lead) + Backend  
**Priority:** P2 | **Target:** Day 5

> **Scope decision:** Full S3 parquet playback is 6 days. Hardcoded playback against the one real simulation run we generated makes the same product point in 2 days. Judges see the scrubber working — that is what matters.

| Story ID | Title | Owner | Days | Status |
|----------|-------|-------|------|--------|
| M-701 | Playback frames endpoint | Backend | 0.5 days | ❌ Not Started |
| M-702 | Playback scrubber UI | Frontend | 1.5 days | ❌ Not Started |

---

### M-701 — Playback Frames Endpoint

**Type:** Task | **Owner:** Backend | **Priority:** P2 | **Estimate:** 0.5 days | **Depends on:** M-303

**Description:**  
Serve the stored simulation run as 24 interpolated frames representing the before/after transition.

**Acceptance Criteria:**
- `GET /api/simulations/history` returns list from `db/simulation_runs`
- `GET /api/simulations/:run_id/frames` returns 24 frames
- Each frame: `{ "frame_index": int, "timestamp_offset_s": int, "road_states": [{ "road_id": str, "occupancy_pct": float }] }`
- Frames in memory after first load — no DB call per frame
- Response < 200ms

**Success Criteria:** 24 frames returned in < 200ms with non-trivial road state changes between frames.

---

### M-702 — Playback Scrubber UI

**Type:** User Story | **Owner:** Frontend | **Priority:** P2 | **Estimate:** 1.5 days | **Depends on:** M-701

**Controls:** Play / Pause · Speed 1× / 4× / 16× · Scrubber timeline · Frame counter "Frame 12 / 24"

**Acceptance Criteria:**
- All frames pre-loaded into React ref on open — no per-frame API calls during playback
- `setInterval` drives frames: 1× = 500ms, 4× = 125ms, 16× = 31ms
- Only changed roads updated via Mapbox `setData()` per frame
- Scrubber seek updates map within one frame interval — instant from memory
- "Playback mode" banner — live WebSocket updates paused while playing
- Loops back to frame 0 at end

**Success Criteria:** Playback at 16× without dropped frames. Scrubber seek updates map within 32ms.

---

---

## CC — JWT Auth Skeleton + Docker + README

**Owner:** All | **Priority:** P0 | **Target:** Day 6

| Story ID | Title | Owner | Days | Status |
|----------|-------|-------|------|--------|
| M-901 | JWT auth middleware skeleton | Backend | 0.5 days | ❌ Not Started |
| M-902 | Docker Compose — full stack | Backend | 0.5 days | ❌ Not Started |

---

### M-901 — JWT Auth Middleware Skeleton

**Type:** Task | **Owner:** Backend | **Priority:** P0 | **Estimate:** 0.5 days | **Depends on:** None

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# POST /auth/token — issues JWT for demo credentials
# Applied to: /api/simulate/run, /api/forecast, /api/nlp/query
# Not applied to: /api/roads, /ws/city-feed (intentionally public for demo)
```

**Acceptance Criteria:**
- `POST /auth/token` with `{ "username": "demo", "password": "demo" }` returns JWT
- Protected endpoints return 401 without token, 200 with valid token
- Token expiry: 24 hours
- `JWT_SECRET_KEY` from env var — never hardcoded
- Comment on public endpoints: `# TODO: add JWT before production`

**Success Criteria:** Protected endpoints return 401 without JWT and 200 with it.

---

### M-902 — Docker Compose — Full Stack

**Type:** Task | **Owner:** Backend | **Priority:** P0 | **Estimate:** 0.5 days | **Depends on:** All epics

**Services:**

| Service | Image | Port |
|---------|-------|------|
| Redpanda | `redpandadata/redpanda:latest` | 9092 |
| TimescaleDB | `timescale/timescaledb:latest-pg15` | 5432 |
| PostGIS | `postgis/postgis:15-3.4` | 5433 |
| Redis | `redis:7-alpine` | 6379 |
| MLflow | `ghcr.io/mlflow/mlflow` | 5000 |
| FastAPI backend | `./backend` | 8000 |
| Faust processor | `./ingest` | — |
| React frontend | `./frontend` | 3000 |
| Sensor replay | `./ingest` (replay command) | — |

**Acceptance Criteria:**
- `docker compose up` starts everything in dependency order via health checks
- DB migrations and seeds run automatically on first start
- Sensor replay starts automatically — no manual command needed
- `.env.example` documents every required variable
- `docker compose down -v` cleans up completely
- Tested on a clean machine with only Docker installed

**Success Criteria:** Clone → copy `.env.example` to `.env` → add Mapbox token → `docker compose up` → working demo at `localhost:3000` within 90 seconds. Zero other steps.

---

## System Design

---

### Functional Requirements

1. Ingest traffic, pollution, and energy sensor data in real-time (< 1s end-to-end)
2. Serve a live road-level traffic heatmap updating from a WebSocket feed
3. Run RL-based traffic light optimization simulation on a road subgraph
4. Forecast congestion up to 24h ahead for any road segment with confidence bands
5. Accept natural language city queries and route to simulation, lookup, or BERT policy engine
6. Render a real-time pollution overlay (NO2, PM2.5, AQI) composited with the traffic layer
7. Play back a recorded simulation run frame-by-frame with scrubber controls

---

### Non-Functional Requirements

1. **Latency:** WebSocket road update delivered to browser < 500ms from sensor publish. Node detail panel < 200ms. Forecast API cache hit < 30ms, cache miss < 1s. NLP lookup path < 500ms, simulation path < 4s.
2. **Throughput:** Ingest pipeline handles 10,000 sensor events/sec sustained. WebSocket server handles 500 concurrent dashboard connections without dropping events.
3. **Model inference:** PPO agent inference < 100ms via ONNX. LSTM forecast inference < 200ms per road on CPU. BERT classification < 500ms on CPU.
4. **Consistency:** PostGIS road state always reflects the most recent clean sensor event. TimescaleDB is the source of truth for history. Redis is a disposable cache — eviction never causes data loss.
5. **Fault tolerance:** Bad sensor events never reach downstream consumers. DLQ captures every rejected event with full context. Sentry alerts on DLQ rate > 1% for 60s.
6. **Reproducibility:** All ML training runs logged to MLflow with hyperparameters, metrics, and model artifacts. Any run reproducible from its `run_id`.

---

### Assumptions

1. Sensor hardware publishes events via MQTT or HTTP webhook — the producer client wraps these into Kafka messages
2. A Mapbox API token is available — road network tiles served by Mapbox, not self-hosted
3. A single city district (approx 2km²) is used as the demo area — full city scale is a production concern
4. User authentication exists at the JWT layer — no full identity provider (Auth0, Cognito) for hackathon scope
5. Synthetic sensor data with realistic temporal patterns is sufficient to demonstrate the ML pipelines

---

### Capacity Estimations

**Data volume:**
```
Road segments in demo district:       200 segments
Sensor events per segment per minute: 2 events
Total events per minute:              200 × 2 = 400 events/min
Total events per day:                 400 × 60 × 24 = 576,000 events/day
Average event payload:                200 bytes
Daily ingest volume:                  576,000 × 200B = ~115 MB/day
30-day TimescaleDB storage:           3.5 GB (before compression)
TimescaleDB compression ratio:        10× on time-series data
Compressed storage:                   350 MB for 30 days
```

**WebSocket load:**
```
Concurrent dashboard users:           500
Events broadcast per second:          ~7 road updates/sec (400/min)
Payload per broadcast:                ~150 bytes
Total WebSocket bandwidth:            500 × 7 × 150B = ~525 KB/sec outbound
```

**ML inference load:**
```
Simulation requests per hour:         ~60 (1 per user per minute)
Redis cache hit rate target:          ~80% (same area, 2-min TTL)
Model inference calls per hour:       60 × 0.2 = 12 actual inference calls
ONNX inference time:                  < 100ms per call
Forecast requests per hour:           ~300 (road clicks)
Cache hit rate (5-min bucket):        ~70%
LSTM inference calls per hour:        300 × 0.3 = 90 inference calls
```

---

### Database Schema Design

#### TimescaleDB — Time-Series Sensor History

*Why TimescaleDB not PostgreSQL:* Hypertables partition data automatically by time chunk. A query for `road_id = X AND time > NOW() - 7 days` touches only the relevant chunks, not the full table. At 576,000 events/day, plain PostgreSQL sequential scans become prohibitively slow within weeks. TimescaleDB keeps the same query under 100ms indefinitely.

```
TABLE: traffic_readings (hypertable, partitioned by time, 1-hour chunks)
─────────────────────────────────────────────────────
time            TIMESTAMPTZ     NOT NULL   — partition key
road_id         TEXT            NOT NULL   — FK to road_nodes
speed_kmh       FLOAT                      — validated 0–200
vehicle_count   INT                        — validated ≥ 0
occupancy_pct   FLOAT                      — validated 0–100
sensor_id       TEXT                       — source sensor

INDEX: (road_id, time DESC)   — covers the 7-day lookback query in LSTM

TABLE: pollution_readings (hypertable, partitioned by time, 1-hour chunks)
─────────────────────────────────────────────────────
time            TIMESTAMPTZ     NOT NULL
road_id         TEXT            NOT NULL
sensor_id       TEXT
no2_ugm3        FLOAT                      — validated ≥ 0
pm25_ugm3       FLOAT                      — validated ≥ 0
aqi             INT                        — validated 0–500

INDEX: (road_id, time DESC)

RETENTION POLICY: auto-drop chunks older than 90 days
COMPRESSION: enabled after 7 days — ~10× size reduction
```

#### PostGIS — Current Road State (Live)

*Why PostGIS not a plain key-value store:* Road segments are geographic objects. Querying "all roads within the current map viewport" requires `ST_MakeEnvelope` + `ST_Intersects` — operations that only make sense on a spatial index. A key-value store like Redis cannot express spatial containment queries.

```
TABLE: road_nodes
─────────────────────────────────────────────────────
road_id         TEXT            PRIMARY KEY    — OSM way ID e.g. "way/12345"
geom            GEOMETRY        NOT NULL       — LineString, SRID 4326
                (LineString, 4326)
name            TEXT                           — OSM road name
speed_limit_kmh INT                            — from OSM maxspeed tag
lanes           INT                            — from OSM lanes tag
speed_kmh       FLOAT                          — current live value (upserted)
occupancy_pct   FLOAT                          — current live value (upserted)
vehicle_count   INT                            — current live value (upserted)
no2_ugm3        FLOAT                          — current live value (upserted)
pm25_ugm3       FLOAT                          — current live value (upserted)
aqi             INT                            — current live value (upserted)
last_updated    TIMESTAMPTZ                    — most recent sensor event time

INDEX: GIST(geom)   — enables ST_Intersects viewport queries
UPSERT: ON CONFLICT (road_id) DO UPDATE SET speed_kmh = EXCLUDED.speed_kmh, ...
```

#### PostgreSQL — Simulation Run Registry

*Why separate from TimescaleDB:* Simulation runs are records, not time-series. A plain relational table with a UUID primary key is the correct model. TimescaleDB is optimised for append-only time-series — it is the wrong tool for mutable records.

```
TABLE: simulation_runs
─────────────────────────────────────────────────────
run_id          UUID            PRIMARY KEY    — generated on each run
created_at      TIMESTAMPTZ     NOT NULL
area            TEXT            NOT NULL       — e.g. "demo_district"
model_version   TEXT            NOT NULL       — e.g. "ppo_v1"
mean_wait_delta FLOAT                          — summary metric
emission_delta  FLOAT                          — summary metric
result_json     JSONB           NOT NULL       — full intersection results
cached          BOOLEAN         DEFAULT FALSE
```

#### Redis — Cache Key Design

*Why TTL-based expiry not manual invalidation:* Manual cache invalidation requires every write path to know about every cache key that might be affected — a coupling that grows with the system. TTL expiry is decoupled: the writer doesn't know or care about the cache. For this system, simulation results are deterministic given their inputs, so a 2-minute TTL is the correct tradeoff between freshness and compute cost.

```
KEY PATTERNS:
─────────────────────────────────────────────────────
sim:{area}:{model_version}
  Value:   serialised simulation result JSON
  TTL:     120 seconds (2 minutes)
  Reason:  road graph doesn't change faster than this

forecast:{road_id}:{horizon_h}:{time_bucket}
  Value:   serialised forecast array (p10/p50/p90 per timestep)
  TTL:     300 seconds (5 minutes)
  Reason:  time_bucket = floor(now, 5min) — all requests in same window share result

node:{road_id}
  Value:   road metadata (name, speed limit, lanes)
  TTL:     3600 seconds (1 hour)
  Reason:  static OSM data — changes at most daily

nlp:policy:{query_hash}
  Value:   BERT classification result + SHAP tokens
  TTL:     600 seconds (10 minutes)
  Reason:  same query text always produces same classification
```

---

### API Design

These are the contracts that matter to engineering reviewers — not just "there's an endpoint" but what it accepts, what it returns, what errors it raises, and what the latency contract is.

#### Authentication

```
POST /auth/token
  Body:    { "username": string, "password": string }
  Returns: { "access_token": string, "token_type": "bearer", "expires_in": 86400 }
  Errors:  401 — invalid credentials
  Notes:   JWT signed with HS256. All /api/* endpoints except /api/roads require Bearer token.
```

#### Roads & Map

```
GET /api/roads?bbox={lat1},{lng1},{lat2},{lng2}
  Auth:    None (public read)
  Returns: GeoJSON FeatureCollection — road segments within bbox
  Query:   SELECT road_id, geom, name, speed_limit_kmh, occupancy_pct, aqi
           FROM road_nodes
           WHERE ST_Intersects(geom, ST_MakeEnvelope($1,$2,$3,$4, 4326))
  Latency: < 200ms
  Notes:   Debounced on frontend — fires at most once per 300ms on viewport change

GET /api/node/:road_id
  Auth:    None (public read)
  Returns: { road_id, name, speed_limit_kmh, lanes, speed_kmh, occupancy_pct,
             vehicle_count, no2_ugm3, pm25_ugm3, aqi, last_updated }
  Query:   Single row lookup on road_nodes PRIMARY KEY — always < 5ms
  Latency: < 50ms
```

#### WebSocket Feed

```
WS /ws/city-feed
  Auth:    None (public for hackathon — TODO: JWT on connect in production)
  Events emitted by server:
    road_update: {
      type: "road_update",
      road_id: string,
      speed_kmh: float,
      occupancy_pct: float,
      no2_ugm3: float,
      aqi: int,
      last_updated: ISO8601
    }
  Heartbeat: server sends ping every 30s, closes connection if no pong within 10s
  Reconnect: client uses exponential backoff — 1s, 2s, 4s, max 30s
```

#### Simulation

```
POST /api/simulate/run
  Auth:    Bearer token required
  Body:    { "area": "demo_district" }
  Returns: {
    run_id: UUID,
    intersections: [{ road_id, current_phase_s, recommended_phase_s, wait_time_delta_s }],
    mean_wait_delta_s: float,
    emission_delta_pct: float,
    cached: boolean,
    model_version: string
  }
  Latency: < 20ms (cache hit) · < 2s (cache miss, ONNX inference)
  Errors:  503 — model file not found
           400 — area not recognised
  Cache:   Redis key sim:{area}:{model_version}, TTL 120s

GET /api/simulations/history
  Auth:    Bearer token required
  Returns: [{ run_id, created_at, area, model_version, mean_wait_delta_s, emission_delta_pct }]
  Latency: < 100ms (PostgreSQL index scan)

GET /api/simulations/:run_id/frames
  Auth:    Bearer token required
  Returns: [{ frame_index, timestamp_offset_s, road_states: [{ road_id, occupancy_pct }] }]
  Latency: < 200ms (memory cache after first load)
```

#### Forecasting

```
GET /api/forecast?road_id={id}&horizon={1|6|24}
  Auth:    Bearer token required
  Returns: {
    road_id: string,
    horizon_hours: int,
    forecast: [{ timestamp: ISO8601, p50: float, p10: float, p90: float }],
    cached: boolean
  }
  Latency: < 30ms (cache hit) · < 1s (cache miss, TimescaleDB + ONNX)
  Errors:  404 — road has < 3 days of history
           400 — invalid horizon value
  Cache:   Redis key forecast:{road_id}:{horizon}:{bucket}, TTL 300s
  Notes:   96 timesteps for 24h horizon (15-min resolution)
           p10/p90 are 10th and 90th percentile bounds — not ±1σ
```

#### NLP Query

```
POST /api/nlp/query
  Auth:    Bearer token required
  Body:    { "query": string }
  Returns: {
    intent: "what_if_simulation" | "congestion_lookup" | "policy_advice" | "unknown",
    answer: string,
    road_id: string | null,
    road_name: string | null,
    data: object | null,
    shap_tokens: [{ token: string, value: float }] | null   — only on policy_advice
  }
  Latency: < 500ms (congestion_lookup) · < 1s (policy_advice) · < 4s (what_if_simulation)
  Errors:  never 500 — unknown intent always returns graceful fallback with answer string
  Notes:   road_id in response used by frontend to highlight road on map
           shap_tokens always present on policy_advice — never null for that intent

POST /api/policy/classify
  Auth:    Bearer token required
  Body:    { "query": string }
  Returns: {
    prediction: "speed_limit_reduction" | "lane_closure" | "signal_retiming" |
                "congestion_charge" | "public_transport_increase",
    confidence: float,
    shap_tokens: [{ token: string, value: float }],
    all_scores: { [intervention]: float }
  }
  Latency: < 500ms on CPU
  Notes:   SHAP tokens sorted descending by absolute value
           all_scores sum to 1.0
```

---

### ML Model Specifications

#### PPO Traffic Light Optimizer

```
Framework:      Stable-Baselines3 (PyTorch backend)
Algorithm:      Proximal Policy Optimisation (PPO)
Serving:        ONNX Runtime — no SB3 dependency at inference time

Environment:    Gymnasium custom env
State space:    Box(20,) — 16 occupancy values + 4 current phases
Action space:   MultiDiscrete([4, 4, 4, 4]) — phase duration per intersection
Reward:         −mean(queue_lengths) − 0.1 × co2_estimate
Timestep:       30 seconds simulation time
Training steps: 50,000
Training time:  1–3 hours on CPU (4 intersections)

Why PPO not DQN:
  DQN requires discrete scalar actions. Our action space is MultiDiscrete
  (a separate choice per intersection). PPO handles MultiDiscrete natively.
  PPO is also more stable on continuous-ish reward landscapes than vanilla DQN.

Why ONNX export:
  SB3 requires PyTorch + SB3 as runtime dependencies in production.
  ONNX Runtime is a single lightweight library. Inference latency drops
  from ~50ms (SB3) to <10ms (ONNX) on the same hardware.
```

#### LSTM Congestion Forecaster

```
Framework:      PyTorch
Architecture:   2-layer LSTM, hidden_size=128, dropout=0.2
Output heads:   3 linear layers — p50, p10, p90 (quantile regression)
Serving:        ONNX Runtime

Input:          7 features × 672 timesteps (7 days at 15-min resolution)
  Features:     speed_kmh, occupancy_pct, vehicle_count,
                sin/cos hour encoding, sin/cos day-of-week encoding
Output:         96 timesteps × 3 quantiles (24h at 15-min resolution)

Why LSTM not Transformer:
  Transformers have O(n²) attention complexity. For a 672-timestep input
  window, a Transformer is significantly more expensive to train and infer
  than an LSTM with no meaningful accuracy gain on univariate time-series.
  TFT (Temporal Fusion Transformer) would be the upgrade path for production
  where multi-variate cross-road interactions matter.

Why quantile regression not Gaussian uncertainty:
  Congestion distributions are not Gaussian — they are right-skewed with
  hard floors at 0. Quantile regression (p10/p50/p90) makes no distributional
  assumptions and correctly captures asymmetric uncertainty.

Training data:  30 days synthetic, realistic morning/evening peak patterns
Val MAPE:       ≤ 20% (relaxed from production 15% — synthetic data)
```

#### BERT Policy Classifier

```
Base model:     bert-base-uncased (110M parameters)
Fine-tuning:    Hugging Face Trainer, 3 epochs, batch_size=16
Task:           5-class sequence classification
Classes:        speed_limit_reduction, lane_closure, signal_retiming,
                congestion_charge, public_transport_increase

Training data:  500 synthetic labelled examples (100 per class)
Val accuracy:   ≥ 75%
Inference:      < 500ms on CPU

Explainability: SHAP (SHapley Additive exPlanations)
  Method:       shap.Explainer wrapping the fine-tuned BERT model
  Output:       Token-level contribution scores
  Use:          Every prediction includes which input tokens drove the
                classification — shown as coloured chips in the UI

Why fine-tuned BERT not few-shot LLM prompting:
  Few-shot prompting delegates classification to a black-box API. The team
  cannot inspect, retrain, or explain the decision boundary. Fine-tuned BERT
  gives full ownership of the model: we control the training data, the
  decision boundary, and the explainability layer. SHAP scores are only
  meaningful on a model you own — they cannot be meaningfully applied to
  a proprietary LLM's internal activations.

Why SHAP not LIME:
  LIME perturbs input tokens randomly and fits a local linear model, results are non-deterministic between runs. SHAP uses Shapley values
  from cooperative game theory, which are unique (axiomatically proven)
  and consistent across runs. For a system that produces policy recommendations,
  consistency in explanations is non-negotiable.
```

---

### System Architecture — Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│ SENSORS                                                             │
│  Traffic sensors · Pollution sensors · Energy grid meters          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ publish events on change
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ INGESTION — Redpanda (Kafka-compatible)                             │
│  city.traffic.raw · city.pollution.raw · city.energy.raw           │
│  city.traffic.clean · city.traffic.dlq                             │
└──────────┬─────────────────────────────────────────────────────────┘
           │ consumed by
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STREAM PROCESSOR — Faust                                            │
│  Validate (hard rules) → Enrich (road_id from PostGIS) → Route     │
│  Valid → clean topic → TimescaleDB + PostGIS upsert                │
│  Invalid → DLQ topic → Sentry alert                                │
└──────────┬──────────────────────────────────────────────────────────┘
           │ writes to
    ┌──────┴──────┐
    ▼             ▼
┌────────────┐  ┌──────────────────────────────────────────────────┐
│TimescaleDB │  │ PostGIS                                          │
│time-series │  │ current road state — upserted on every event    │
│history     │  │ spatial index for viewport queries              │
└────────────┘  └─────────────────┬────────────────────────────────┘
                                  │ triggers
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BACKEND — FastAPI                                                   │
│  REST endpoints · WebSocket server · JWT middleware                │
│  Simulation runner (ONNX PPO) · Forecast runner (ONNX LSTM)       │
│  NLP router · BERT classifier (ONNX/HuggingFace)                  │
│  Redis cache layer (TTL-based, no manual invalidation)            │
└──────────┬──────────────────────────────────────────────────────────┘
           │ serves
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FRONTEND — React + TypeScript + Mapbox GL                          │
│  Zustand store (road state, keyed by road_id)                      │
│  WebSocket client (auto-reconnect, exponential backoff)           │
│  Heatmap layer · Pollution overlay · NLP chat panel               │
│  Forecast chart (D3.js) · Simulation overlay · Playback scrubber  │
└─────────────────────────────────────────────────────────────────────┘

ML MODELS (served by Backend)
  PPO agent        → ONNX Runtime  → < 100ms inference
  LSTM forecaster  → ONNX Runtime  → < 200ms inference
  BERT classifier  → HuggingFace   → < 500ms inference + SHAP tokens
  All training runs logged to MLflow (localhost:5000)
```

---

### Technology Decision 

| Decision | Chosen | Rejected | Reason |
|----------|--------|----------|--------|
| Message broker | Redpanda | Apache Kafka | Kafka requires Zookeeper + broker config. Redpanda is one container, same API, starts in 30s. On 6-day timeline this is not preference, it is correct. |
| Time-series DB | TimescaleDB | PostgreSQL, InfluxDB | TimescaleDB is PostgreSQL with hypertables — same SQL, same driver, same ORM. InfluxDB requires a separate query language (Flux). TimescaleDB compresses 10× and keeps queries < 100ms indefinitely. |
| Road state store | PostGIS | Redis, DynamoDB | Road segments are geographic objects. Viewport queries require spatial indexes (`ST_Intersects`). Redis and DynamoDB cannot express spatial containment. |
| Frontend state | Zustand | Redux, React Context | WebSocket fires 50+ events/sec. Redux's action dispatch cycle adds unnecessary overhead. Zustand's partial `setState` merge is O(1). Context re-renders the full tree on every update. |
| Cache expiry | Redis TTL | Manual invalidation | Manual invalidation couples every write path to every cache key. TTL is decoupled — writers don't know about caches. Simulation results are deterministic on inputs, so TTL is the correct model. |
| Model serving | ONNX Runtime | Full framework (SB3, PyTorch) | SB3 requires PyTorch as a runtime dependency. ONNX Runtime is a single lightweight library. Inference latency: < 10ms (ONNX) vs ~50ms (SB3). Runtime independence means the serving layer doesn't know what trained the model. |
| Uncertainty quantification | Quantile regression (p10/p50/p90) | Gaussian (mean ± σ) | Congestion distributions are right-skewed with hard floors at 0. Gaussian assumptions are wrong. Quantile regression makes no distributional assumptions and correctly captures asymmetric uncertainty. |
| Explainability | SHAP | LIME | LIME is non-deterministic between runs (random perturbation). SHAP values are axiomatically unique — same input always produces the same explanation. For policy recommendations, consistency is non-negotiable. |
| Stream processing | Faust | Spark Streaming, Flink | Faust is Python-native, runs in the same Docker image as the producer. Spark requires a cluster. Flink requires JVM. For a 5-person hackathon team with a Python-first stack, Faust is the correct scope decision. |
