# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:31:38 2026

@author: Berra
"""

import pyomo
from pyomo.environ import *
import pandas as pd
import time

start_time = time.time()

# Create the model
M = ConcreteModel()

# Select the dataset to solve: "I", "II" or "III"
DATASET = "III"

# Path to the Excel file (relative to the repository root)
path = f"data/END395_TermProject_Data{DATASET}.xlsx"

# Reading data from Excel
ef_masterData    = pd.read_excel(path, sheet_name="MasterData")
ef_vehicleTypes  = pd.read_excel(path, sheet_name="VehicleTypes")
ef_loadingPlans  = pd.read_excel(path, sheet_name="LoadingPlans")
ef_destinations  = pd.read_excel(path, sheet_name="Destinations")
ef_routes        = pd.read_excel(path, sheet_name="Routes")
ef_orders        = pd.read_excel(path, sheet_name="Orders")

M.planningHorizonDays = int(ef_masterData.set_index(ef_masterData.columns[0]).at["PlanningHorizonDays", ef_masterData.columns[1]])
E_MAX = int(ef_masterData.set_index(ef_masterData.columns[0]).at["PartII_EarlinessMaxDays", ef_masterData.columns[1]])  # 1 day
L_MAX = int(ef_masterData.set_index(ef_masterData.columns[0]).at["PartII_LatenessMaxDays",  ef_masterData.columns[1]])  # 2 days

df_temp = ef_loadingPlans.set_index("PlanID")[["Sports Car", "Sedan", "SUV", "Pickup"]]

# =========================================================
# ROUTE PRE-PROCESSING
# =========================================================
dest_dist_dict = ef_destinations.set_index("Destination")["DistanceFromFacility_km"].to_dict()
all_destinations_list = list(ef_destinations["Destination"])

route_dests_dict = {}
for _, row in ef_routes.iterrows():
    # Read the routes from Excel exactly as they are, stripping whitespace
    route_dests_dict[row["RouteID"]] = [d.strip() for d in str(row["Destinations"]).split(",")]

all_routes_list = list(route_dests_dict.keys())

# Calculate route parameters
dist_r_dict = {}   # Distance of each route (to the farthest destination)
dmax_r_dict = {}   # Farthest destination in each route
M_dr_dict   = {}   # M_dr[d, r] = 1 if destination d is in route r
route_multi = {}   # True if it's a multi-destination route

for r, dlist in route_dests_dict.items():
    max_dist   = 0
    farthest_d = ""
    for dest in dlist:
        if dest_dist_dict[dest] > max_dist:
            max_dist   = dest_dist_dict[dest]
            farthest_d = dest
            
    dist_r_dict[r] = max_dist
    dmax_r_dict[r] = farthest_d
    route_multi[r] = len(dlist) >= 2
    
    for d_all in all_destinations_list:
        M_dr_dict[d_all, r] = 1 if d_all in dlist else 0

# =========================================================
# SETS
# =========================================================
M.O = Set(initialize=ef_orders["OrderID"].unique())          # Set of customer orders
M.L = Set(initialize=ef_loadingPlans["PlanID"].unique())     # Set of feasible loading plans
M.T = RangeSet(1, M.planningHorizonDays)                     # Set of planning horizon (1..5)
M.V = Set(initialize=ef_vehicleTypes["VehicleType"])         # Set of vehicle types
M.D = Set(initialize=all_destinations_list)                  # Set of destinations
M.R = Set(initialize=all_routes_list)                        # Set of routes

# Upper bound for maximum number of trucks
min_capacity = df_temp.sum(axis=1).min()
max_truck    = int(len(ef_orders) / min_capacity) + 10
M.K = RangeSet(1, max_truck)                                 # Set of truck indices

# =========================================================
# PARAMETERS
# =========================================================
M.v_o    = ef_orders.set_index("OrderID")["VehicleType"].to_dict()   # Vehicle type of the order
M.tau_o  = ef_orders.set_index("OrderID")["DueDate"].to_dict()       # Due date of the order
M.d_o    = ef_orders.set_index("OrderID")["Destination"].to_dict()   # Destination of the order
M.a_lv   = df_temp.to_dict("index")                                  # Loading plan capacities
M.c_d    = dest_dist_dict                                            # Destination distances

M.Dist_r  = Param(M.R, initialize=dist_r_dict)                # Route distance
M.d_max_r = Param(M.R, initialize=dmax_r_dict, within=Any)    # Farthest destination of the route
M.M_dr    = Param(M.D, M.R, initialize=M_dr_dict)             # Destination-route compatibility matrix

# =========================================================
# DECISION VARIABLES
# =========================================================
# X[o, k, t] = 1: If order o is assigned to truck k on day t
M.X = Var(M.O, M.K, M.T, domain=Binary)

# Y[k, l, t] = 1: If truck k is dispatched with loading plan l on day t
M.Y = Var(M.K, M.L, M.T, domain=Binary)

# Z[k, r, t] = 1: If truck k is assigned to route r on day t
M.Z = Var(M.K, M.R, M.T, domain=Binary)

# W[k, t] = 1: If truck k is used (dispatched) on day t
M.W = Var(M.K, M.T, domain=Binary)

# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
# The objective in Part II is to minimize the total travel distance.
def obj_rule(model):
    return sum(model.Dist_r[r] * model.Z[k, r, t]
               for k in model.K for r in model.R for t in model.T)
M.Obj = Objective(rule=obj_rule, sense=minimize)

# =========================================================
# CONSTRAINTS
# =========================================================

# C1: Each order must be assigned to exactly one truck and one day.
def c1_rule(model, o):
    return sum(model.X[o, k, t] for k in model.K for t in model.T) == 1
M.C1 = Constraint(M.O, rule=c1_rule)

# C2: Time Windows (Specific to Part II)
def c2_rule(model, o, k, t):
    tau      = model.tau_o[o]
    earliest = max(1, tau - E_MAX)
    latest   = min(model.planningHorizonDays, tau + L_MAX)
    if t < earliest or t > latest:
        return model.X[o, k, t] == 0
    return Constraint.Skip
M.C2 = Constraint(M.O, M.K, M.T, rule=c2_rule)

# C3: Truck Usage and Loading Plan Relationship
def c3_rule(model, k, t):
    return sum(model.Y[k, l, t] for l in model.L) == model.W[k, t]
M.C3 = Constraint(M.K, M.T, rule=c3_rule)

# C3b: Link between W[k,t] and order assignments
Big_M_tight = int(df_temp.sum(axis=1).max()) 
def c3b_rule(model, k, t):
    return sum(model.X[o, k, t] for o in model.O) <= Big_M_tight * model.W[k, t]
M.C3b = Constraint(M.K, M.T, rule=c3b_rule) 

# C3c: A dispatched truck must carry at least one order 
def c3c_rule(model, k, t):
    return sum(model.X[o, k, t] for o in model.O) >= model.W[k, t]
M.C3c = Constraint(M.K, M.T, rule=c3c_rule)

# C4: Vehicle Capacity Constraint
def c4_rule(model, k, t, v):
    assigned_v = sum(model.X[o, k, t] for o in model.O if model.v_o[o] == v)
    capacity_v = sum(model.a_lv[l][v] * model.Y[k, l, t] for l in model.L)
    return assigned_v <= capacity_v
M.C4 = Constraint(M.K, M.T, M.V, rule=c4_rule)

# C5: Route Assignment Constraint
def c5_rule(model, k, t):
    return sum(model.Z[k, r, t] for r in model.R) == model.W[k, t]
M.C5 = Constraint(M.K, M.T, rule=c5_rule)

# C6: Destination Compatibility
def c6_rule(model, o, k, t):
    dest_o      = model.d_o[o]
    valid_z_sum = sum(model.Z[k, r, t] for r in model.R if model.M_dr[dest_o, r] == 1)
    return model.X[o, k, t] <= valid_z_sum
M.C6 = Constraint(M.O, M.K, M.T, rule=c6_rule)

# C7: Farthest-Destination Majority Requirement
Big_M_c7 = int(df_temp.sum(axis=1).max())  # Maximum number of vehicles a truck can hold
def c7_rule(model, k, t, r):
    if not route_multi[r]:
        return Constraint.Skip   # Only applies to multi-destination routes

    farthest_dest = model.d_max_r[r]
    n_far         = sum(model.X[o, k, t] for o in model.O if model.d_o[o] == farthest_dest)
    n_total       = sum(model.X[o, k, t] for o in model.O)

    return 2 * n_far >= n_total - Big_M_c7 * (1 - model.Z[k, r, t])
M.C7 = Constraint(M.K, M.T, M.R, rule=c7_rule)

# C8: Symmetry Breaking Constraint - Helps Gurobi
def c8_rule(model, k, t):
    if k == 1:
        return Constraint.Skip
    return model.W[k, t] <= model.W[k - 1, t]
M.C8 = Constraint(M.K, M.T, rule=c8_rule)

# =========================================================
# =========================================================
print("\nStarting automated multi-scenario solve (15, 30, 60, 120 mins)...")


scenarios = [
    {"name": "15_Mins",  "add_time": 900,  "total_mins": 15},
    {"name": "30_Mins",  "add_time": 900,  "total_mins": 30},   # 15 + 15 = 30
    {"name": "60_Mins",  "add_time": 1800, "total_mins": 60},   # 30 + 30 = 60
    {"name": "120_Mins", "add_time": 3600, "total_mins": 120}   # 60 + 60 = 120
]

all_results_summary = []
dispatch_dict = {}
cumulative_solve_time = 0

for sc in scenarios:
    s_name = sc["name"]
    s_add_time = sc["add_time"]
    s_total_mins = sc["total_mins"]

    print("\n" + "="*60)
    print(f"--- RUNNING SCENARIO: {s_name} (Target Total: {s_total_mins} mins) ---")
    print(f"--- Adding {s_add_time} seconds to the current progress... ---")
    print("="*60)

    solver = SolverFactory('gurobi')
    solver.options['MIPFocus']   = 1
    solver.options['Presolve']   = 2
    if DATASET == "III":
        solver.options['Heuristics'] = 0.5  # Used only for the largest dataset (see report, Section 4.4)
    solver.options['TimeLimit']  = s_add_time

    start_solve = time.time()
    
    # warmstart=True keeps the previous solution in memory to build upon it
    results = solver.solve(M, tee=True, warmstart=True)
    
    solve_duration = time.time() - start_solve
    cumulative_solve_time += solve_duration

    try:
        obj_val     = value(M.Obj)
        lower_bound = results.problem.lower_bound
        upper_bound = results.problem.upper_bound
        gap = 0 if upper_bound in [0, None] else abs(upper_bound - lower_bound) / abs(upper_bound) * 100
        status = "Feasible" if gap > 0.0001 else "Optimal"
    except:
        obj_val = None
        gap = None
        status = "Infeasible or No Solution"

    all_results_summary.append({
        "Scenario": s_name,
        "Status": status,
        "Objective_km": obj_val,
        "Optimality_Gap_%": gap,
        "Cumulative_Time_sec": cumulative_solve_time
    })

    # Dispatch Details
    scenario_dispatch = []
    if obj_val is not None:
        for k in M.K:
            for t in M.T:
                if value(M.W[k, t]) > 0.5:
                    chosen_r = next(r for r in M.R if value(M.Z[k, r, t]) > 0.5)
                    chosen_l = next(l for l in M.L if value(M.Y[k, l, t]) > 0.5)
                    distance = dist_r_dict[chosen_r]
                    assigned_orders = [o for o in M.O if value(M.X[o, k, t]) > 0.5]

                    scenario_dispatch.append({
                        "Day": int(t),
                        "Truck_ID": f"Truck_{int(k):02d}",
                        "Loading_Plan": chosen_l,
                        "Route_ID": chosen_r,
                        "Distance_km": distance,
                        "Total_Orders": len(assigned_orders),
                        "Assigned_Orders": ", ".join(assigned_orders)
                    })
                    
    dispatch_dict[s_name] = scenario_dispatch

# =========================================================
# PRINT EXCEL
# =========================================================
print("\n" + "="*60)
print("Exporting all scenarios to a single Excel file...")
output_filename = f"results/END395_PartII_Results_Dataset{len(DATASET)}.xlsx"

try:
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        
        # 1. Sheet: Summary 
        df_summary = pd.DataFrame(all_results_summary)
        df_summary.to_excel(writer, index=False, sheet_name="Summary")

        # Other Sheets: 15 Min, 30 Min, 60 Min and 120 Min Truck Assignments
        for s_name, data in dispatch_dict.items():
            if data: 
                df = pd.DataFrame(data)
                df = df.sort_values(by=["Day", "Truck_ID"])
                df.to_excel(writer, index=False, sheet_name=s_name)

    print(f"SUCCESS: All results exported perfectly to '{output_filename}'")
    print("Check your folder for the Excel file!")
except Exception as e:
    print(f"ERROR: Excel export failed. Is the file currently open? Details: {e}")

print("="*60)
print("RUN COMPLETE. YOU CAN NOW USE THE EXCEL FILE FOR YOUR REPORT.")