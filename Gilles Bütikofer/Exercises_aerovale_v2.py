#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:41:09 2026

@author: gillesbutikofer
"""


"""
AEROVALE strategic planning model for ModAvi Block 1.

Purpose
-------
This script models future airside facility requirements for AEROVALE Regional Airport.
It compares a conventional planning strategy with a flexible planning strategy under
uncertain future demand.

Workflow
--------
1. Load historical passenger and flight movement data.
2. Remove the COVID years 2020-2022 as trend-breaking outliers.
3. Estimate historical growth parameters for passengers and flight movements.
4. Generate stochastic passenger and flight-movement scenarios using GBM.
5. Introduce new user groups:
   - VTOL/UAM traffic from 2030
   - UAV traffic from 2035
6. Convert total traffic into market shares for traditional aviation, UAVs, and VTOLs.
7. Define revenues, operating costs, capital costs, and discounting assumptions.
8. Compute the NPV for each demand scenario.
9. Compare conventional planning with flexible planning using ENPV.
10. Use grid search to find the best conventional capacity parameters and the best
    flexible decision-rule parameters.
11. Plot capacity development and target curves / ECDFs for the final comparison.

Main outputs
------------
- Historical and stochastic demand plots
- Market share and movement forecasts by user group
- Yearly revenue vs. cost plot
- Best conventional and flexible ENPV values
- Capacity development plots
- Target curves showing the distribution of scenario NPVs

Important assumptions
---------------------
The model is simplified for educational purposes. Cost and revenue parameters are
rough estimates, and the market-entry assumptions for UAV and VTOL traffic are
scenario assumptions rather than forecasts.
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================

FAST_MODE = False

# =============================================================================

# Import data
AEROVALE_Demand = pd.read_csv("aerovale_demand.csv", sep=";")

year = AEROVALE_Demand["Year"].to_numpy()
flight_movements = AEROVALE_Demand["Flight_Movements"].to_numpy()
passengers = AEROVALE_Demand["Passengers"].to_numpy()

fig, ax1 = plt.subplots()

# Passengers (left axis)
ax1.scatter(year, passengers, color='blue')
ax1.plot(year, passengers, color='blue', label='Passengers')
ax1.set_xlabel("Year")
ax1.set_ylabel("Passengers", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Flight movements (right axis)
ax2 = ax1.twinx()
ax2.scatter(year, flight_movements, color='red')
ax2.plot(year, flight_movements, color='red', label='Flight Movements')
ax2.set_ylabel("Flight Movements", color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Aerovale Demand")
plt.savefig("Aerovale Demand.png")
plt.show()
plt.close()

# REMOVE OUTLIERS - TREND BREAKERS
# remove years 2020, 2021, 2022
AEROVALE_Demand = AEROVALE_Demand[~AEROVALE_Demand["Year"].isin([2020, 2021, 2022])]

year = AEROVALE_Demand["Year"].to_numpy()
flight_movements = AEROVALE_Demand["Flight_Movements"].to_numpy()
passengers = AEROVALE_Demand["Passengers"].to_numpy()

fig, ax1 = plt.subplots()

# Passengers (left axis)
ax1.scatter(year, passengers, color='blue')
ax1.plot(year, passengers, color='blue', label='Passengers')
ax1.set_xlabel("Year")
ax1.set_ylabel("Passengers", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Flight movements (right axis)
ax2 = ax1.twinx()
ax2.scatter(year, flight_movements, color='red')
ax2.plot(year, flight_movements, color='red', label='Flight Movements')
ax2.set_ylabel("Flight Movements", color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Aerovale Demand, without outliers")
plt.savefig("Aerovale Demand, without outliers.png")
plt.show()
plt.close()

# PASSENGERS
growth_PAX_ln = np.log(passengers[1:] / passengers[:-1])
mu_hat_PAX = np.mean(growth_PAX_ln)
sigma_hat_PAX = np.std(growth_PAX_ln)
print("mu_hat_PAX =", round(mu_hat_PAX * 100,3), "%")
print("sigma_hat_PAX =", round(sigma_hat_PAX * 100,3), "%")

# FLIGHTS
growth_FLIGTHS_ln = np.log(flight_movements[1:] / flight_movements[:-1])
mu_hat_FLIGTHS = np.mean(growth_FLIGTHS_ln)
sigma_hat_FLIGTHS = np.std(growth_FLIGTHS_ln)
print("mu_hat_FLIGTHS =", round(mu_hat_FLIGTHS * 100,3), "%")
print("sigma_hat_FLIGTHS =", round(sigma_hat_FLIGTHS * 100,3), "%")



# ----- GBM PASSENGERS
# GBM parameters
DELTA_t = year[1] - year[0]
passengers[-1]

# =============================================================================
# # SIMULATION OF 1000 CASES
# epsilon = np.random.normal(0, 1, 1000) see epsilon_P 
# =============================================================================

# ## commented because overwrittent later 
# PAX_GBM = [passengers[-1]]
# for i in range(25):
#     PAX_GBM.append(PAX_GBM[i] * np.exp(mu_hat_PAX*DELTA_t+sigma_hat_PAX*epsilon*np.sqrt(DELTA_t)))
    
# print(PAX_GBM)

# store in nd array



T = 25


if FAST_MODE:
    N = 200
else:
    N = 1000

t = np.arange(2024, 2050, 1)


# ------ GBM forecast for PASSENGERS 

epsilon_P = np.random.normal(0, 1, (T, N))

PAX_0 = passengers[-1]
PAX_GBM = np.zeros((T+1, N))
PAX_GBM[0] = PAX_0

for t_step in range(T):
    PAX_GBM[t_step+1] = PAX_GBM[t_step] * np.exp(mu_hat_PAX + sigma_hat_PAX * epsilon_P[t_step])


# plt.plot(t,PAX_GBM)
# plt.show()
# plt.close()

mean_PAX_GBM = np.mean(PAX_GBM, axis=1)
std_PAX_GBM = np.std(PAX_GBM, axis=1)

plt.figure(figsize=(10,7))
plt.plot(year, passengers, marker ="o", label="Passengers")
plt.plot(t, PAX_GBM, color="lightblue", alpha=0.2, label="Passengers forecast")
plt.plot(t, mean_PAX_GBM)
plt.plot(t, mean_PAX_GBM-std_PAX_GBM)
plt.plot(t, mean_PAX_GBM+std_PAX_GBM)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2])
plt.title("Passenger forecast 2024 - 2049")
plt.savefig("Passenger forecast 2024 - 2049.png")
plt.show()
plt.close()

# ------ GBM forecast for FLIGHT MOVEMENTS 

epsilon_F = np.random.normal(0, 1, (T, N))

FLIGTHS_0 = flight_movements[-1]
FLIGTHS_GBM = np.zeros((T+1, N))
FLIGTHS_GBM[0] = FLIGTHS_0

for t_step in range(T):
    FLIGTHS_GBM[t_step+1] = FLIGTHS_GBM[t_step] * np.exp(mu_hat_FLIGTHS + sigma_hat_FLIGTHS * epsilon_F[t_step])


# plt.plot(t,FLIGTHS_GBM)
# plt.show()
# plt.close()

mean_FLIGTHS_GBM = np.mean(FLIGTHS_GBM, axis=1)
std_FLIGTHS_GBM = np.std(FLIGTHS_GBM, axis=1)

plt.figure(figsize=(10,7))
plt.plot(year, flight_movements, marker ="o", label="Flight movements")
plt.plot(t, FLIGTHS_GBM, color="lightblue", alpha=0.2, label="Flight movements forecast")
plt.plot(t, mean_FLIGTHS_GBM)
plt.plot(t, mean_FLIGTHS_GBM-std_FLIGTHS_GBM)
plt.plot(t, mean_FLIGTHS_GBM+std_FLIGTHS_GBM)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2])
plt.title("Flight movements forecast 2024 - 2049")
plt.savefig("Flight movements forecast 2024 - 2049.png")
plt.show()
plt.close()


### WE CAN SEE THAT THE FORECAST FOR THE NUMBER OF PASSENGER GROWS EXPONENTIALLY 
### WHERE AS THE NUMBER OF FILGHT MOVEMENTS IS MORE OR LESS LINEAR



# ------ POINT 3 : MARKET SHARE SCENARIOS UAV & VTOL 

# ------ GBM MARKET SHARE SCENARIOS UAV & VTOL
# IDEA:
# Traditional aviation remains the base user group
# VTOL enters the market
# UAV enters the market
# we forecast the VTOL traffic  
# we forecast the UAV traffic
    
# ASSUMPTIONS 
# UAV:
# - introduction in 2035
# - 15 movements per day
# - growth rate of 10% per year

#
# VTOL / UAM:
# - introduction in 2030
# - 10 movements per day
# - 2 passengers per aircraft
# - growth rate of 10% per year




# Total market forecast = your already simulated flight forecast
total_movements = FLIGTHS_GBM.copy()   # shape (T+1, N)

# -----------------------------
# VTOL forecast
# -----------------------------
VTOL_intro_year = 2030
VTOL_intro_idx = VTOL_intro_year - t[0]

mu_hat_VTOL = 0.10
sigma_hat_VTOL = sigma_hat_FLIGTHS

epsilon_VTOL = np.random.normal(0, 1, (T, N))

VTOL_0 = 10 * 365
VTOL_raw = np.zeros((T+1, N))
VTOL_raw[VTOL_intro_idx] = VTOL_0

for t_step in range(VTOL_intro_idx, T):
    VTOL_raw[t_step+1] = VTOL_raw[t_step] * np.exp(mu_hat_VTOL + sigma_hat_VTOL * epsilon_VTOL[t_step])

# -----------------------------
# UAV forecast
# -----------------------------
UAV_intro_year = 2035
UAV_intro_idx = UAV_intro_year - t[0]

mu_hat_UAV = 0.10
sigma_hat_UAV = sigma_hat_FLIGTHS

epsilon_UAV = np.random.normal(0, 1, (T, N))

UAV_0 = 25 * 365
UAV_raw = np.zeros((T+1, N))
UAV_raw[UAV_intro_idx] = UAV_0

for t_step in range(UAV_intro_idx, T):
    UAV_raw[t_step+1] = UAV_raw[t_step] * np.exp(mu_hat_UAV + sigma_hat_UAV * epsilon_UAV[t_step])

# -----------------------------
# Convert raw demands into market shares
# -----------------------------
# Traditional aviation is the residual demand, but never negative
TRAD_raw = np.maximum(total_movements - UAV_raw - VTOL_raw, 0)

# Important:
# If UAV_raw + VTOL_raw exceeds total_movements, we normalize all 3 groups (market saturation)
# so that the shares sum exactly to 1.
raw_sum = TRAD_raw + UAV_raw + VTOL_raw

# avoid division by zero just in case
raw_sum = np.where(raw_sum == 0, 1, raw_sum)

market_share_TRAD = TRAD_raw / raw_sum
market_share_UAV  = UAV_raw  / raw_sum
market_share_VTOL = VTOL_raw / raw_sum

# Reconstruct movements so stacked bars always sum exactly to total_movements
TRAD_GBM = market_share_TRAD * total_movements
UAV_GBM  = market_share_UAV  * total_movements
VTOL_GBM = market_share_VTOL * total_movements

# -----------------------------
# Mean market shares
# -----------------------------
mean_market_share_TRAD = np.mean(market_share_TRAD, axis=1)
mean_market_share_UAV  = np.mean(market_share_UAV, axis=1)
mean_market_share_VTOL = np.mean(market_share_VTOL, axis=1)


# -----------------------------
# Stacked market-share bar chart
# -----------------------------
plt.figure(figsize=(10,7))

plt.bar(t, mean_market_share_TRAD, label="Traditional aviation")
plt.bar(t, mean_market_share_UAV,
        bottom=mean_market_share_TRAD, label="UAV")
plt.bar(t, mean_market_share_VTOL,
        bottom=mean_market_share_TRAD + mean_market_share_UAV, label="VTOL")

plt.xlabel("Year")
plt.ylabel("Market share")
plt.title("Market share distribution (stacked) 2024 - 2050")
plt.legend()
plt.savefig("Market share distribution (stacked) 2024 - 2050.png")
plt.show()
plt.close()

# -----------------------------
# Mean movements per user group
# -----------------------------
mean_TRAD_movements = np.mean(TRAD_GBM, axis=1)
mean_UAV_movements  = np.mean(UAV_GBM, axis=1)
mean_VTOL_movements = np.mean(VTOL_GBM, axis=1)

mean_total_movements = np.mean(total_movements, axis=1)


# -----------------------------
# Stacked absolute movements chart
# -----------------------------
plt.figure(figsize=(10,7))

plt.bar(t, mean_TRAD_movements, label="Traditional aviation")
plt.bar(t, mean_UAV_movements,
        bottom=mean_TRAD_movements, label="UAV")
plt.bar(t, mean_VTOL_movements,
        bottom=mean_TRAD_movements + mean_UAV_movements, label="VTOL")

plt.xlabel("Year")
plt.ylabel("Number of movements")
plt.title("Total flight movements by user group (stacked) 2024 - 2050")
plt.legend()
plt.savefig("Total flight movements by user group (stacked) 2024 - 2050.png")
plt.show()
plt.close()

# -----------------------------
# Historical + forecast by user group
# -----------------------------
plt.figure(figsize=(10,7))

# historical
plt.plot(year, flight_movements, marker="o", label="Historical flights (TRAD)")

# forecast means
plt.plot(t, mean_TRAD_movements, label="TRAD forecast")
plt.plot(t, mean_UAV_movements, label="UAV forecast")
plt.plot(t, mean_VTOL_movements, label="VTOL forecast")
plt.plot(t, mean_total_movements, linestyle="--", label="Total forecast")

plt.xlabel("Year")
plt.ylabel("Number of movements")
plt.title("Flight movements by user group (historical + forecast)")
plt.legend()
plt.savefig("Flight movements by user group (historical + forecast).png")
plt.show()
plt.close()





# =================== Exercise 2 ===================
# ============== Exercise 2 - Point 6 ==============

# Planning horizon
years = np.arange(2024, 2050, 1)   # 2024 ... 2049
n_years = len(years)

# Demand vectors from Exercise 1 forecast
demand_TRAD = mean_TRAD_movements.copy()
demand_UAV  = mean_UAV_movements.copy()
demand_VTOL = mean_VTOL_movements.copy()

# -----------------------------
# Market launch timing
# -----------------------------
# VTOL_intro_year UAV_intro_year are already defined earlier

# Initial investment is done one year before market launch

VTOL_pre_idx = VTOL_intro_idx - 1   # 2029
UAV_pre_idx  = UAV_intro_idx - 1    # 2034

# -----------------------------
# Initial capacities
# -----------------------------
K0_TRAD = 35000   # initial TRAD capacity at t = 0
K0_UAV  = 10000   # first UAV module, installed in 2034
K0_VTOL = 5000    # first VTOL module, installed in 2029

# -----------------------------
# Decision rules for expansion
# -----------------------------
THETA1 = 1.05   # trigger threshold
THETA2 = 1.20   # expansion factor

# -----------------------------
# Revenues
# -----------------------------
landing_fee_TRAD = 200
landing_fee_UAV  = 75
landing_fee_VTOL = 100
charging_fee     = 80   # per VTOL movement

# -----------------------------
# Operational costs
# -----------------------------
charging_cost = 25      # electricity cost per VTOL movement

runway_maintenance = 100000
maint_TRAD = 2.0        # CHF per unit of installed TRAD capacity
maint_UAV  = 1.5        # CHF per unit of installed UAV capacity
maint_VTOL = 0.75       # CHF per unit of installed VTOL capacity

# -----------------------------
# Capital costs
# installation cost parameters are defined per 1000 additional yearly movements
# -----------------------------
alpha = 0.7
installation_cost_TRAD = 50000
installation_cost_UAV  = 40000
installation_cost_VTOL = 25000

# -----------------------------
# Discount rate
# -----------------------------
discount_rate = 0.04

# -----------------------------
# Helper function
# delta_K is in yearly movements
# c_inst is the reference cost for 1000 additional yearly movements
# -----------------------------
def Cost_installation(delta_K, alpha, c_inst):
    delta_K = max(delta_K, 0)
    return (delta_K / 1000)**alpha * c_inst

# -----------------------------
# Storage vectors
# -----------------------------
capacity_TRAD = np.zeros(n_years)
capacity_UAV  = np.zeros(n_years)
capacity_VTOL = np.zeros(n_years)

activity_TRAD = np.zeros(n_years)
activity_UAV  = np.zeros(n_years)
activity_VTOL = np.zeros(n_years)

delta_K_TRAD_vec = np.zeros(n_years)
delta_K_UAV_vec  = np.zeros(n_years)
delta_K_VTOL_vec = np.zeros(n_years)

revenues   = np.zeros(n_years)
opex       = np.zeros(n_years)
capex      = np.zeros(n_years)
total_cost = np.zeros(n_years)
cash_flow  = np.zeros(n_years)

# -----------------------------
# Year-by-year simulation of revenues and costs
# -----------------------------
for i in range(n_years):

    # carry forward last year capacity
    if i > 0:
        capacity_TRAD[i] = capacity_TRAD[i-1]
        capacity_UAV[i]  = capacity_UAV[i-1]
        capacity_VTOL[i] = capacity_VTOL[i-1]

    # ----- initial investments
    if i == 0:
        capacity_TRAD[i] += K0_TRAD

    if i == VTOL_pre_idx:
        capacity_VTOL[i] += K0_VTOL

    if i == UAV_pre_idx:
        capacity_UAV[i] += K0_UAV

    # ----- expansion decisions rules
    if demand_TRAD[i] > capacity_TRAD[i] * THETA1:
        capacity_TRAD[i] *= THETA2

    if i >= UAV_intro_idx and demand_UAV[i] > capacity_UAV[i] * THETA1:
        capacity_UAV[i] *= THETA2

    if i >= VTOL_intro_idx and demand_VTOL[i] > capacity_VTOL[i] * THETA1:
        capacity_VTOL[i] *= THETA2

    # previous capacities (needed for delta_K)
    prev_TRAD = capacity_TRAD[i-1] if i > 0 else K0_TRAD
    prev_UAV  = capacity_UAV[i-1]  if i > 0 else K0_UAV
    prev_VTOL = capacity_VTOL[i-1] if i > 0 else K0_VTOL

    # capacity additions in year i
    delta_K_TRAD = capacity_TRAD[i] - prev_TRAD
    delta_K_UAV  = capacity_UAV[i]  - prev_UAV
    delta_K_VTOL = capacity_VTOL[i] - prev_VTOL

    delta_K_TRAD_vec[i] = delta_K_TRAD
    delta_K_UAV_vec[i]  = delta_K_UAV
    delta_K_VTOL_vec[i] = delta_K_VTOL

    # ----- actual served activity = min(demand, capacity)
    activity_TRAD[i] = min(demand_TRAD[i], capacity_TRAD[i])
    activity_UAV[i]  = min(demand_UAV[i], capacity_UAV[i])
    activity_VTOL[i] = min(demand_VTOL[i], capacity_VTOL[i])

    # ----- revenues
    revenues[i] = (
        activity_TRAD[i] * landing_fee_TRAD +
        activity_UAV[i]  * landing_fee_UAV +
        activity_VTOL[i] * landing_fee_VTOL +
        activity_VTOL[i] * charging_fee
    )

    # ----- operational expenditures
    electricity_charge_cost = activity_VTOL[i] * charging_cost

    opex[i] = (
        runway_maintenance +
        capacity_TRAD[i] * maint_TRAD +
        capacity_UAV[i]  * maint_UAV +
        capacity_VTOL[i] * maint_VTOL +
        electricity_charge_cost
    )

    # ----- capital expenditures
    capex[i] = (
        Cost_installation(delta_K_TRAD, alpha, installation_cost_TRAD) +
        Cost_installation(delta_K_UAV,  alpha, installation_cost_UAV) +
        Cost_installation(delta_K_VTOL, alpha, installation_cost_VTOL)
    )

    # ----- total cost and cash flow
    total_cost[i] = opex[i] + capex[i]
    cash_flow[i]  = revenues[i] - total_cost[i]

# -----------------------------
# Discount factors (for point 7 / point 8 later)
# -----------------------------
discount_factor = 1 / (1 + discount_rate)**np.arange(n_years)

# summary table - checking
exercise2_point6 = pd.DataFrame({
    "Year": years,
    "Demand_TRAD": demand_TRAD,
    "Demand_UAV": demand_UAV,
    "Demand_VTOL": demand_VTOL,
    "Capacity_TRAD": capacity_TRAD,
    "Capacity_UAV": capacity_UAV,
    "Capacity_VTOL": capacity_VTOL,
    "DeltaK_TRAD": delta_K_TRAD_vec,
    "DeltaK_UAV": delta_K_UAV_vec,
    "DeltaK_VTOL": delta_K_VTOL_vec,
    "Activity_TRAD": activity_TRAD,
    "Activity_UAV": activity_UAV,
    "Activity_VTOL": activity_VTOL,
    "Revenues": revenues,
    "OPEX": opex,
    "CAPEX": capex,
    "Total_Cost": total_cost,
    "Cash_Flow": cash_flow,
    "Discount_Factor": discount_factor
})

# ===== Check if decision rules / financial logic behave properly =====

x = np.arange(n_years)
bar_width = 0.4

plt.figure(figsize=(14, 7))

plt.bar(x - bar_width/2, revenues, bar_width, label="Revenues")
plt.bar(x + bar_width/2, total_cost, bar_width, label="Total costs")

plt.xticks(x, years, rotation=45)
plt.xlabel("Year")
plt.ylabel("CHF")
plt.title("Yearly revenues vs total costs")
plt.legend()
plt.tight_layout()
plt.savefig("Yearly revenues vs total costs.png")
plt.show()
plt.close()


# ===== Point 7 - Conventional planning =====

def compute_npv_conventional_with_capacities(
    demand_TRAD, demand_UAV, demand_VTOL,
    K0_TRAD, K0_UAV, K0_VTOL,
    expand_year_TRAD=None, deltaK_TRAD=0,
    expand_year_UAV=None,  deltaK_UAV=0,
    expand_year_VTOL=None, deltaK_VTOL=0
):
    
    capacity_TRAD = np.zeros(n_years)
    capacity_UAV  = np.zeros(n_years)
    capacity_VTOL = np.zeros(n_years)

    revenues   = np.zeros(n_years)
    opex       = np.zeros(n_years)
    capex      = np.zeros(n_years)
    total_cost = np.zeros(n_years)
    cash_flow  = np.zeros(n_years)

    expand_idx_TRAD = np.where(years == expand_year_TRAD)[0][0] if expand_year_TRAD is not None else None
    expand_idx_UAV  = np.where(years == expand_year_UAV)[0][0]  if expand_year_UAV is not None else None
    expand_idx_VTOL = np.where(years == expand_year_VTOL)[0][0] if expand_year_VTOL is not None else None

    for i in range(n_years):

        if i > 0:
            capacity_TRAD[i] = capacity_TRAD[i-1]
            capacity_UAV[i]  = capacity_UAV[i-1]
            capacity_VTOL[i] = capacity_VTOL[i-1]

        if i == 0:
            capacity_TRAD[i] += K0_TRAD

        if i == UAV_pre_idx:
            capacity_UAV[i] += K0_UAV

        if i == VTOL_pre_idx:
            capacity_VTOL[i] += K0_VTOL

        if expand_idx_TRAD is not None and i == expand_idx_TRAD:
            capacity_TRAD[i] += deltaK_TRAD

        if expand_idx_UAV is not None and i == expand_idx_UAV:
            capacity_UAV[i] += deltaK_UAV

        if expand_idx_VTOL is not None and i == expand_idx_VTOL:
            capacity_VTOL[i] += deltaK_VTOL

        prev_TRAD = capacity_TRAD[i-1] if i > 0 else 0
        prev_UAV  = capacity_UAV[i-1]  if i > 0 else 0
        prev_VTOL = capacity_VTOL[i-1] if i > 0 else 0

        delta_K_TRAD = capacity_TRAD[i] - prev_TRAD
        delta_K_UAV  = capacity_UAV[i]  - prev_UAV
        delta_K_VTOL = capacity_VTOL[i] - prev_VTOL

        activity_TRAD = min(demand_TRAD[i], capacity_TRAD[i])
        activity_UAV  = min(demand_UAV[i], capacity_UAV[i])
        activity_VTOL = min(demand_VTOL[i], capacity_VTOL[i])

        revenues[i] = (
            activity_TRAD * landing_fee_TRAD +
            activity_UAV  * landing_fee_UAV +
            activity_VTOL * landing_fee_VTOL +
            activity_VTOL * charging_fee
        )

        electricity_charge_cost = activity_VTOL * charging_cost

        opex[i] = (
            runway_maintenance +
            capacity_TRAD[i] * maint_TRAD +
            capacity_UAV[i]  * maint_UAV +
            capacity_VTOL[i] * maint_VTOL +
            electricity_charge_cost
        )

        capex[i] = (
            Cost_installation(delta_K_TRAD, alpha, installation_cost_TRAD) +
            Cost_installation(delta_K_UAV,  alpha, installation_cost_UAV) +
            Cost_installation(delta_K_VTOL, alpha, installation_cost_VTOL)
        )

        total_cost[i] = opex[i] + capex[i]
        cash_flow[i]  = revenues[i] - total_cost[i]

    npv = np.sum(cash_flow / (1 + discount_rate)**np.arange(n_years))
    
    return npv, capacity_TRAD, capacity_UAV, capacity_VTOL


# ===== Point 7 - Flexible planning =====

def compute_npv_flexible_with_capacities(
    demand_TRAD, demand_UAV, demand_VTOL,
    K0_TRAD, K0_UAV, K0_VTOL,
    THETA1, THETA2
):
    
    capacity_TRAD = np.zeros(n_years)
    capacity_UAV  = np.zeros(n_years)
    capacity_VTOL = np.zeros(n_years)

    revenues   = np.zeros(n_years)
    opex       = np.zeros(n_years)
    capex      = np.zeros(n_years)
    total_cost = np.zeros(n_years)
    cash_flow  = np.zeros(n_years)

    for i in range(n_years):

        if i > 0:
            capacity_TRAD[i] = capacity_TRAD[i-1]
            capacity_UAV[i]  = capacity_UAV[i-1]
            capacity_VTOL[i] = capacity_VTOL[i-1]

        if i == 0:
            capacity_TRAD[i] += K0_TRAD

        if i == UAV_pre_idx:
            capacity_UAV[i] += K0_UAV

        if i == VTOL_pre_idx:
            capacity_VTOL[i] += K0_VTOL

        if demand_TRAD[i] > capacity_TRAD[i] * THETA1:
            capacity_TRAD[i] *= THETA2 

        if i >= UAV_intro_idx and demand_UAV[i] > capacity_UAV[i] * THETA1:
            capacity_UAV[i] *= THETA2

        if i >= VTOL_intro_idx and demand_VTOL[i] > capacity_VTOL[i] * THETA1:
            capacity_VTOL[i] *= THETA2

        prev_TRAD = capacity_TRAD[i-1] if i > 0 else 0
        prev_UAV  = capacity_UAV[i-1]  if i > 0 else 0
        prev_VTOL = capacity_VTOL[i-1] if i > 0 else 0

        delta_K_TRAD = capacity_TRAD[i] - prev_TRAD
        delta_K_UAV  = capacity_UAV[i]  - prev_UAV
        delta_K_VTOL = capacity_VTOL[i] - prev_VTOL

        activity_TRAD = min(demand_TRAD[i], capacity_TRAD[i])
        activity_UAV  = min(demand_UAV[i], capacity_UAV[i])
        activity_VTOL = min(demand_VTOL[i], capacity_VTOL[i])

        revenues[i] = (
            activity_TRAD * landing_fee_TRAD +
            activity_UAV  * landing_fee_UAV +
            activity_VTOL * landing_fee_VTOL +
            activity_VTOL * charging_fee
        )

        electricity_charge_cost = activity_VTOL * charging_cost

        opex[i] = (
            runway_maintenance +
            capacity_TRAD[i] * maint_TRAD +
            capacity_UAV[i]  * maint_UAV +
            capacity_VTOL[i] * maint_VTOL +
            electricity_charge_cost
        )

        capex[i] = (
            Cost_installation(delta_K_TRAD, alpha, installation_cost_TRAD) +
            Cost_installation(delta_K_UAV,  alpha, installation_cost_UAV) +
            Cost_installation(delta_K_VTOL, alpha, installation_cost_VTOL)
        )

        total_cost[i] = opex[i] + capex[i]
        cash_flow[i]  = revenues[i] - total_cost[i]

    npv = np.sum(cash_flow / (1 + discount_rate)**np.arange(n_years))
    
    return npv, capacity_TRAD, capacity_UAV, capacity_VTOL


# ===== ENPV - conventional planning =====

npv_conventional = np.zeros(N)

for j in range(N):
    npv_conventional[j], _, _, _ = compute_npv_conventional_with_capacities(
        demand_TRAD=TRAD_GBM[:, j],
        demand_UAV=UAV_GBM[:, j],
        demand_VTOL=VTOL_GBM[:, j],
        K0_TRAD=35000,
        K0_UAV=10000,
        K0_VTOL=5000,
        expand_year_UAV=2034,  deltaK_UAV=4000,
        expand_year_VTOL=2029, deltaK_VTOL=3000
    )

ENPV_conventional = np.mean(npv_conventional) # mean of the 1000 scenarios 

print("ENPV conventional =", round(ENPV_conventional, 2), "CHF")
print("ENPV conventional =", round(ENPV_conventional/1e6, 2), "M CHF")


# ===== Grid search - conventional planning =====
# Assumption: TRAD has no expansion

best_ENPV_conventional = -np.inf
best_params_conventional = None

results_conventional = []

K0_UAV_grid = [15000, 25000, 35000]
K0_VTOL_grid = [10000, 20000, 30000,]

expand_year_UAV_grid = [2034] # removed to limit amount of loops
expand_year_VTOL_grid = [2029] # removed to limit amount of loops

deltaK_UAV_grid = [5000, 5500, 6000, 6500, 7000]
deltaK_VTOL_grid = [2000, 2500, 3000, 3500, 4000]

print("Looping in Grid search - conventional planning ...")
for K0_UAV_test in K0_UAV_grid:
    print()
    print("A",end="")
    for K0_VTOL_test in K0_VTOL_grid:
        print("B",end="")
        for expand_year_UAV_test in expand_year_UAV_grid:
            for expand_year_VTOL_test in expand_year_VTOL_grid:
                for deltaK_UAV_test in deltaK_UAV_grid:
                    for deltaK_VTOL_test in deltaK_VTOL_grid:

                        npv_conventional = np.zeros(N)

                        for j in range(N):


                            npv_conventional[j], capacity_TRAD, capacity_UAV, capacity_VTOL = compute_npv_conventional_with_capacities(
                                demand_TRAD=TRAD_GBM[:, j],
                                demand_UAV=UAV_GBM[:, j],
                                demand_VTOL=VTOL_GBM[:, j],
                                K0_TRAD=35000,
                                K0_UAV=K0_UAV_test,
                                K0_VTOL=K0_VTOL_test,
                                expand_year_UAV=expand_year_UAV_test,
                                deltaK_UAV=deltaK_UAV_test,
                                expand_year_VTOL=expand_year_VTOL_test,
                                deltaK_VTOL=deltaK_VTOL_test
                            )

                        ENPV_conventional = np.mean(npv_conventional)

                        results_conventional.append([
                            K0_UAV_test,
                            K0_VTOL_test,
                            expand_year_UAV_test,
                            expand_year_VTOL_test,
                            deltaK_UAV_test,
                            deltaK_VTOL_test,
                            ENPV_conventional
                        ])

                        if ENPV_conventional > best_ENPV_conventional:
                            best_ENPV_conventional = ENPV_conventional
                            best_params_conventional = {
                                "K0_UAV": K0_UAV_test,
                                "K0_VTOL": K0_VTOL_test,
                                "expand_year_UAV": expand_year_UAV_test,
                                "expand_year_VTOL": expand_year_VTOL_test,
                                "deltaK_UAV": deltaK_UAV_test,
                                "deltaK_VTOL": deltaK_VTOL_test
                            }
print()
print("Best ENPV conventional =", round(best_ENPV_conventional, 2), "CHF")
print("Best ENPV conventional =", round(best_ENPV_conventional / 1e6, 2), "M CHF")
print("Best conventional parameters =", best_params_conventional)

# ===== ENPV - flexible planning =====

npv_flexible = np.zeros(N)

for j in range(N):
    npv_flexible[j], _, _, _ = compute_npv_flexible_with_capacities(
        demand_TRAD=TRAD_GBM[:, j],
        demand_UAV=UAV_GBM[:, j],
        demand_VTOL=VTOL_GBM[:, j],
        K0_TRAD=35000,
        K0_UAV=10000,
        K0_VTOL=5000,
        THETA1=1.05,
        THETA2=1.20
    )

ENPV_flexible = np.mean(npv_flexible) # mean of the 1000 scenarios 

print("ENPV flexible =", round(ENPV_flexible, 2), "CHF")
print("ENPV flexible =", round(ENPV_flexible/1e6, 2), "M CHF")


# ===== Grid search - flexible planning =====

best_ENPV_flexible = -np.inf
best_params_flexible = None

results_flexible = []

# K0_UAV_grid = [15000, 20000, 25000, 30000, 35000, 40000]
# K0_VTOL_grid = [7500, 10000, 15000, 20000, 25000, 30000, 35000, 40000]

K0_UAV_grid = [15000, 25000, 35000, 45000]
K0_VTOL_grid = [10000, 20000, 30000, 40000]

THETA1_grid = [0.9, 0.95, 1.00, 1.05, 1.10]
THETA2_grid = [1.15, 1.20, 1.25]

print("Looping in Grid search - flexible planning ...")

for K0_UAV_test in K0_UAV_grid:
    print()
    print("A",end="")
    for K0_VTOL_test in K0_VTOL_grid:
        print("B",end="")
        for THETA1_test in THETA1_grid:
            print("C",end="")
            for THETA2_test in THETA2_grid:

                npv_flexible = np.zeros(N)

                for j in range(N):
                    npv_flexible[j], capacity_TRAD, capacity_UAV, capacity_VTOL = compute_npv_flexible_with_capacities(
                        demand_TRAD=TRAD_GBM[:, j],
                        demand_UAV=UAV_GBM[:, j],
                        demand_VTOL=VTOL_GBM[:, j],
                        K0_TRAD=35000,
                        K0_UAV=K0_UAV_test,
                        K0_VTOL=K0_VTOL_test,
                        THETA1=THETA1_test,
                        THETA2=THETA2_test
                    )

                ENPV_flexible = np.mean(npv_flexible)

                results_flexible.append([
                    K0_UAV_test,
                    K0_VTOL_test,
                    THETA1_test,
                    THETA2_test,
                    ENPV_flexible
                ])

                if ENPV_flexible > best_ENPV_flexible:
                    best_ENPV_flexible = ENPV_flexible
                    best_params_flexible = {
                        "K0_UAV": K0_UAV_test,
                        "K0_VTOL": K0_VTOL_test,
                        "THETA1": THETA1_test,
                        "THETA2": THETA2_test
                    }
print()
print("Best ENPV flexible =", round(best_ENPV_flexible, 2), "CHF")
print("Best ENPV flexible =", round(best_ENPV_flexible / 1e6, 2), "M CHF")
print("Best flexible parameters =", best_params_flexible)


print("Best ENPV conventional =", round(best_ENPV_conventional / 1e6, 2), "M CHF")
print("Best ENPV flexible     =", round(best_ENPV_flexible / 1e6, 2), "M CHF")
ratio_ENPV = (best_ENPV_flexible - best_ENPV_conventional)/best_ENPV_conventional
print("Difference flexible vs conventional =",
      round((best_ENPV_flexible - best_ENPV_conventional) / 1e6, 2), "M CHF",
      "(", round(ratio_ENPV*100,2), "% )")


# -------- get best case timeline --------

# best conventional capacity path
_, cap_TRAD_conv, cap_UAV_conv, cap_VTOL_conv = compute_npv_conventional_with_capacities(
    demand_TRAD=mean_TRAD_movements,
    demand_UAV=mean_UAV_movements,
    demand_VTOL=mean_VTOL_movements,
    K0_TRAD=35000,
    K0_UAV=best_params_conventional["K0_UAV"],
    K0_VTOL=best_params_conventional["K0_VTOL"],
    expand_year_UAV=best_params_conventional["expand_year_UAV"],
    deltaK_UAV=best_params_conventional["deltaK_UAV"],
    expand_year_VTOL=best_params_conventional["expand_year_VTOL"],
    deltaK_VTOL=best_params_conventional["deltaK_VTOL"]
)

# best flexible capacity path
_, cap_TRAD_flex, cap_UAV_flex, cap_VTOL_flex = compute_npv_flexible_with_capacities(
    demand_TRAD=mean_TRAD_movements,
    demand_UAV=mean_UAV_movements,
    demand_VTOL=mean_VTOL_movements,
    K0_TRAD=35000,
    K0_UAV=best_params_flexible["K0_UAV"],
    K0_VTOL=best_params_flexible["K0_VTOL"],
    THETA1=best_params_flexible["THETA1"],
    THETA2=best_params_flexible["THETA2"]
)

plt.figure(figsize=(10, 7))
plt.plot(years, cap_TRAD_conv, marker="o", label="TRAD capacity")
plt.plot(years, cap_UAV_conv, marker="o", label="UAV capacity")
plt.plot(years, cap_VTOL_conv, marker="o", label="VTOL capacity")
plt.xlabel("Year")
plt.ylabel("Capacity")
plt.title("Capacity development over time - Best conventional planning")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Capacity development over time - Best conventional planning.png")
plt.show()
plt.close()

plt.figure(figsize=(10, 7))
plt.plot(years, cap_TRAD_flex, marker="o", label="TRAD capacity")
plt.plot(years, cap_UAV_flex, marker="o", label="UAV capacity")
plt.plot(years, cap_VTOL_flex, marker="o", label="VTOL capacity")
plt.xlabel("Year")
plt.ylabel("Capacity")
plt.title("Capacity development over time - Best flexible planning")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Capacity development over time - Best flexible planning.png")
plt.show()
plt.close()


# ===== Exercise 2 - Point 8 =====

# ============================================================
# Recompute NPV distributions using the BEST parameters
# This is needed for the target curves / ECDF plot.
# Otherwise npv_conventional and npv_flexible may still contain
# the last grid-search case, not the best case.
# ============================================================

npv_conventional_best = np.zeros(N)

for j in range(N):
    npv_conventional_best[j], _, _, _ = compute_npv_conventional_with_capacities(
        demand_TRAD=TRAD_GBM[:, j],
        demand_UAV=UAV_GBM[:, j],
        demand_VTOL=VTOL_GBM[:, j],
        K0_TRAD=35000,
        K0_UAV=best_params_conventional["K0_UAV"],
        K0_VTOL=best_params_conventional["K0_VTOL"],
        expand_year_UAV=best_params_conventional["expand_year_UAV"],
        deltaK_UAV=best_params_conventional["deltaK_UAV"],
        expand_year_VTOL=best_params_conventional["expand_year_VTOL"],
        deltaK_VTOL=best_params_conventional["deltaK_VTOL"]
    )

npv_flexible_best = np.zeros(N)

for j in range(N):
    npv_flexible_best[j], _, _, _ = compute_npv_flexible_with_capacities(
        demand_TRAD=TRAD_GBM[:, j],
        demand_UAV=UAV_GBM[:, j],
        demand_VTOL=VTOL_GBM[:, j],
        K0_TRAD=35000,
        K0_UAV=best_params_flexible["K0_UAV"],
        K0_VTOL=best_params_flexible["K0_VTOL"],
        THETA1=best_params_flexible["THETA1"],
        THETA2=best_params_flexible["THETA2"]
    )

# Optional sanity check
print("Recomputed best ENPV conventional =", round(np.mean(npv_conventional_best) / 1e6, 2), "M CHF")
print("Recomputed best ENPV flexible     =", round(np.mean(npv_flexible_best) / 1e6, 2), "M CHF")


# ----- target curves = ECDFs of NPVs
npv_conventional_sorted = np.sort(npv_conventional_best)
npv_flexible_sorted = np.sort(npv_flexible_best)

ecdf_conventional = np.arange(1, len(npv_conventional_sorted) + 1) / len(npv_conventional_sorted)
ecdf_flexible = np.arange(1, len(npv_flexible_sorted) + 1) / len(npv_flexible_sorted)

plt.figure(figsize=(10, 7))
plt.plot(npv_conventional_sorted / 1e6, ecdf_conventional, label="Conventional planning", color="red")
plt.plot(npv_flexible_sorted / 1e6, ecdf_flexible, label="Flexible planning")

plt.axvline(np.mean(npv_conventional_best) / 1e6, linestyle="--", label="ENPV conventional", color="red")
plt.axvline(np.mean(npv_flexible_best) / 1e6, linestyle="--", label="ENPV flexible")

plt.xlabel("NPV [M CHF]")
plt.ylabel("Cumulative probability")
plt.title("Target curves (ECDF) of conventional and flexible planning")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Target curves ECDF of conventional and flexible planning.png")
plt.show()
plt.close()



