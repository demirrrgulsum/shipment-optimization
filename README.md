# Multi-Period Shipment Optimization (MILP)

Term project for **END 395 – Operations Research II** at TOBB ETÜ (**Spring 2026**), Group 5.

A Mixed-Integer Linear Programming model for a **Multi-Period Load Assignment and Vehicle Routing Problem** faced by an automotive manufacturer: over a 5-day horizon, decide which orders go on which truck, on which day, with which loading plan and along which route, so that the **total travel distance is minimized**.

Built with **Python, Pyomo and Gurobi**.

## Problem

- Orders for 4 vehicle types (Sports Car, Sedan, SUV, Pickup) must be shipped from one facility to 10 destinations.
- Each truck uses one of the feasible **loading plans**, which limit how many vehicles of each type it can carry.
- Trucks follow predefined single- or multi-destination **routes**.
- **Time windows:** an order may ship up to 1 day early or 2 days late relative to its due date.
- **Farthest-destination majority rule:** on a multi-stop route, at least 50% of the load must go to the farthest destination.

## Model

Decision variables (all binary):

| Variable | Meaning |
|---|---|
| `X[o,k,t]` | order *o* is assigned to truck *k* on day *t* |
| `Y[k,l,t]` | truck *k* uses loading plan *l* on day *t* |
| `Z[k,r,t]` | truck *k* follows route *r* on day *t* |
| `W[k,t]` | truck *k* is dispatched on day *t* |

**Objective:** minimize Σ distance(r) · Z[k,r,t]

Main constraints: one assignment per order, time windows, loading-plan capacity per vehicle type, one route per dispatched truck, destination–route compatibility, and the farthest-destination majority rule (linearized with Big-M).

**Strengthening techniques**
- Big-M tightened from the number of orders to the maximum truck capacity.
- Symmetry breaking for identical trucks: `W[k,t] ≤ W[k-1,t]`.
- Non-empty truck constraint, which removes "ghost" trucks created by symmetry breaking.
- Gurobi tuning: `MIPFocus=1`, `Presolve=2`, and `Heuristics=0.5` for the largest dataset.

The full formulation is in the [project report](docs/END395_Project_Report_Group05.pdf).

## Results

Each dataset was solved with cumulative time limits of 15, 30, 60 and 120 minutes (warm-started between runs).

| Dataset | Orders | Best distance (km) | Gap at 120 min |
|---|---|---|---|
| I | 40 | 6,800 | 27.1% |
| II | 120 | 13,500 | 47.0% |
| III | 240 | 25,940 | 63.0% |

The problem is NP-hard; good feasible plans are found early, while the lower bound improves slowly, which explains the remaining gaps. Detailed truck-by-truck dispatch plans are in [`results/`](results/).

## Repository structure

```
src/shipment_optimization.py   # Pyomo model + solve loop + Excel export
data/                          # Course-provided input datasets (I, II, III)
results/                       # Dispatch plans and summary for each dataset
docs/                          # Final project report
```

## How to run

Requires Python 3.10+ and a [Gurobi](https://www.gurobi.com/academia/) license (free for academic use).

```bash
pip install -r requirements.txt
python src/shipment_optimization.py
```

Run it from the repository root. Choose the dataset by setting `DATASET = "I"`, `"II"` or `"III"` at the top of the script. A full run takes about 2 hours per dataset.

## Team

Furkan Ateş · Gülsüm Demir · Eda Orhan · Berra Ülger
