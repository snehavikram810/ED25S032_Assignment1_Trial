import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ortools.sat.python import cp_model


# -----------------------------------------------------------
# Chennai dark store location model with maximum service distance
# -----------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@st.cache_data
def generate_chennai_locations(store_count: int, customer_count: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Generate candidate dark store and customer locations around Chennai."""
    rng = np.random.default_rng(42)

    # Approximate Chennai area bounds
    lat_min, lat_max = 12.78, 13.22
    lon_min, lon_max = 80.10, 80.35

    stores = [
        (
            float(rng.uniform(lat_min, lat_max)),
            float(rng.uniform(lon_min, lon_max)),
        )
        for _ in range(store_count)
    ]
    customers = [
        (
            float(rng.uniform(lat_min, lat_max)),
            float(rng.uniform(lon_min, lon_max)),
        )
        for _ in range(customer_count)
    ]
    return stores, customers


def build_distance_matrix(
    stores: List[Tuple[float, float]], customers: List[Tuple[float, float]]
) -> pd.DataFrame:
    rows = []
    for i, customer in enumerate(customers):
        dists = []
        for j, store in enumerate(stores):
            dist = haversine_km(customer[0], customer[1], store[0], store[1])
            dists.append(dist)
        rows.append(dists)
    return pd.DataFrame(rows, index=[f"C{i+1}" for i in range(len(customers))], columns=[f"S{j+1}" for j in range(len(stores))])


def solve_dark_store_model(
    stores: List[Tuple[float, float]],
    customers: List[Tuple[float, float]],
    max_distance_km: float,
    demand: List[int],
    fixed_cost: List[float],
    capacity: List[int],
    max_open_stores: int | None = None,
) -> Tuple[Dict, Dict, Dict, float, float, float, List[str]]:
    """Solve a CP-SAT facility model. Each customer must be assigned to exactly one open store within max distance."""
    n_customers = len(customers)
    n_stores = len(stores)

    # Build distance matrix
    dist = np.zeros((n_customers, n_stores), dtype=float)
    for i in range(n_customers):
        for j in range(n_stores):
            dist[i, j] = haversine_km(
                customers[i][0], customers[i][1], stores[j][0], stores[j][1]
            )

    # Infeasibility guard
    for i in range(n_customers):
        eligible = [j for j in range(n_stores) if dist[i, j] <= max_distance_km]
        if not eligible:
            raise ValueError(
                f"Customer {i+1} cannot be served within the max distance of {max_distance_km} km. "
                "Please increase the distance limit or change store locations."
            )

    model = cp_model.CpModel()

    x = {}
    for i in range(n_customers):
        for j in range(n_stores):
            if dist[i, j] <= max_distance_km:
                x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

    y = {j: model.NewBoolVar(f"y_{j}") for j in range(n_stores)}

    # Each customer is assigned once to a feasible store
    for i in range(n_customers):
        feasible = [x[(i, j)] for j in range(n_stores) if (i, j) in x]
        model.Add(sum(feasible) == 1)

    # Capacity constraint at each open store
    for j in range(n_stores):
        demand_terms = [x[(i, j)] * demand[i] for i in range(n_customers) if (i, j) in x]
        if demand_terms:
            model.Add(sum(demand_terms) <= capacity[j] * y[j])

    # Open/closed store logic: if a store is closed, no customer assigned
    for i in range(n_customers):
        for j in range(n_stores):
            if (i, j) in x:
                model.Add(x[(i, j)] <= y[j])

    if max_open_stores is not None:
        model.Add(sum(y.values()) <= max_open_stores)

    # Objective: minimize transport + fixed costs
    objective_terms = []
    for i in range(n_customers):
        for j in range(n_stores):
            if (i, j) in x:
                # Use a scaled cost to keep integer objective.
                cost = int(round(demand[i] * dist[i, j] * 100))
                objective_terms.append(cost * x[(i, j)])

    for j in range(n_stores):
        objective_terms.append(int(round(fixed_cost[j] * 100)) * y[j])

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution found for the selected parameters.")

    assigned = {}
    for i in range(n_customers):
        for j in range(n_stores):
            if (i, j) in x and solver.Value(x[(i, j)]) == 1:
                assigned[i] = j

    open_stores = [j for j in range(n_stores) if solver.Value(y[j])]  # type: ignore[arg-type]

    total_transport_cost = 0.0
    total_fixed_cost = 0.0
    total_demand_served = 0

    for i in range(n_customers):
        j = assigned[i]
        total_transport_cost += demand[i] * dist[i, j]
        total_demand_served += demand[i]

    for j in open_stores:
        total_fixed_cost += fixed_cost[j]

    profit = (total_demand_served * 18.0) - total_transport_cost - total_fixed_cost

    result = {
        "status": solver.StatusName(status),
        "total_demand_served": total_demand_served,
        "total_transport_cost": total_transport_cost,
        "total_fixed_cost": total_fixed_cost,
        "profit": profit,
        "open_stores": open_stores,
    }

    assignment_detail = {
        i: j for i, j in assigned.items()
    }
    return result, assignment_detail, dist, total_transport_cost, total_fixed_cost, profit, open_stores


def build_lookup_df(
    stores: List[Tuple[float, float]],
    customers: List[Tuple[float, float]],
    assignment_detail: Dict[int, int],
    dist: np.ndarray,
    demand: List[int],
) -> pd.DataFrame:
    rows = []
    for i, customer in enumerate(customers):
        j = assignment_detail.get(i)
        if j is None:
            continue
        rows.append(
            {
                "customer_id": f"C{i+1}",
                "customer_lat": customer[0],
                "customer_lon": customer[1],
                "assigned_store": f"S{j+1}",
                "store_lat": stores[j][0],
                "store_lon": stores[j][1],
                "distance_km": round(float(dist[i, j]), 2),
                "demand": demand[i],
            }
        )
    return pd.DataFrame(rows)


def greedy_large_scale_profit_curve(
    store_count: int = 50,
    customer_count: int = 10000,
    max_distance_km: float = 10.0,
    max_opening_options: List[int] = [2, 5, 10, 20, 25],
) -> pd.DataFrame:
    """A fast approximate large-scale profit curve for the dashboard. This is a reasonable assumption for 10k customers and 50 stores."""
    rng = np.random.default_rng(2024)
    lat_min, lat_max = 12.78, 13.22
    lon_min, lon_max = 80.10, 80.35

    stores = [
        (float(rng.uniform(lat_min, lat_max)), float(rng.uniform(lon_min, lon_max)))
        for _ in range(store_count)
    ]
    customers = [
        (float(rng.uniform(lat_min, lat_max)), float(rng.uniform(lon_min, lon_max)))
        for _ in range(customer_count)
    ]

    demand = rng.integers(1, 25, size=customer_count).tolist()
    fixed_cost = [1200 + rng.uniform(200, 800) for _ in range(store_count)]

    distances = np.zeros((customer_count, store_count), dtype=float)
    for i in range(customer_count):
        for j in range(store_count):
            distances[i, j] = haversine_km(customers[i][0], customers[i][1], stores[j][0], stores[j][1])

    feasible_mask = distances <= max_distance_km
    profit_curve = []

    for max_open in max_opening_options:
        # Greedy selection by incremental gain
        selected = set()
        remaining = set(range(store_count))
        total_profit = 0.0

        while len(selected) < max_open and remaining:
            best_store = None
            best_gain = -1e18

            for j in sorted(remaining):
                gain = 0.0
                for i in range(customer_count):
                    if feasible_mask[i, j]:
                        # Demand can be served by this store if selected; assume a simplified profit contribution.
                        gain += demand[i] * 18.0
                gain -= fixed_cost[j]

                # reward if this store covers many previously uncovered customers
                if best_store is None or gain > best_gain:
                    best_store = j
                    best_gain = gain

            if best_store is None:
                break

            selected.add(best_store)
            remaining.remove(best_store)
            total_profit += best_gain

        # approximate final profit when the set is selected
        total_revenue = 0.0
        for i in range(customer_count):
            best_store = None
            best_distance = float("inf")
            for j in selected:
                if distances[i, j] < best_distance:
                    best_distance = distances[i, j]
                    best_store = j
            if best_store is not None:
                total_revenue += demand[i] * 18.0

        total_fixed = sum(fixed_cost[j] for j in selected)
        total_transport = 0.0
        for i in range(customer_count):
            best_store = None
            best_distance = float("inf")
            for j in selected:
                if distances[i, j] < best_distance:
                    best_distance = distances[i, j]
                    best_store = j
            if best_store is not None:
                total_transport += demand[i] * best_distance * 0.8

        profit = total_revenue - total_transport - total_fixed
        profit_curve.append(
            {
                "max_open_stores": max_open,
                "profit": round(float(profit), 2),
                "selected_stores": len(selected),
            }
        )

    return pd.DataFrame(profit_curve)


def render_model_results(
    stores: List[Tuple[float, float]],
    customers: List[Tuple[float, float]],
    result: Dict,
    assignment_detail: Dict[int, int],
    dist: np.ndarray,
    demand: List[int],
    max_distance_km: float,
):
    st.subheader("Optimization results")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", result["status"])
    col2.metric("Open stores", len(result["open_stores"]))
    col3.metric("Demand served", f"{result['total_demand_served']}")
    col4.metric("Profit", f"${result['profit']:.2f}")

    kpi_df = pd.DataFrame(
        [
            {"KPI": "Total transport cost", "Value": round(result["total_transport_cost"], 2)},
            {"KPI": "Total fixed cost", "Value": round(result["total_fixed_cost"], 2)},
            {"KPI": "Total profit", "Value": round(result["profit"], 2)},
            {"KPI": "Open stores", "Value": len(result["open_stores"])},
            {"KPI": "Max distance constraint", "Value": f"{max_distance_km} km"},
        ]
    )
    st.table(kpi_df)

    # Map
    map_df = build_lookup_df(stores, customers, assignment_detail, dist, demand)

    if map_df.empty:
        st.warning("No customer assignments were produced. Adjust the constraints or the generated dataset.")
        return

    store_rows = []
    for idx, store in enumerate(stores):
        store_rows.append({
            "store_id": f"S{idx + 1}",
            "lat": store[0],
            "lon": store[1],
            "open": idx in result["open_stores"],
        })
    store_df = pd.DataFrame(store_rows)

    fig = go.Figure()

    # Customers
    fig.add_trace(
        go.Scattermapbox(
            lat=map_df["customer_lat"],
            lon=map_df["customer_lon"],
            mode="markers",
            marker={"size": 8, "color": "#1f77b4", "opacity": 0.7},
            text=map_df["customer_id"],
            hovertemplate="Customer: %{text}<extra></extra>",
            name="Customers",
        )
    )

    # Open stores
    open_store_df = store_df[store_df["open"] == True]
    if not open_store_df.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=open_store_df["lat"],
                lon=open_store_df["lon"],
                mode="markers+text",
                marker={"size": 20, "color": "#ff7f0e", "symbol": "star"},
                text=open_store_df["store_id"],
                textposition="bottom center",
                hovertemplate="Open store: %{text}<extra></extra>",
                name="Open stores",
            )
        )

    # Connections from customers to assigned stores
    for _, row in map_df.iterrows():
        fig.add_trace(
            go.Scattermapbox(
                lat=[row["customer_lat"], row["store_lat"]],
                lon=[row["customer_lon"], row["store_lon"]],
                mode="lines",
                line={"width": 1, "color": "rgba(255, 127, 14, 0.6)"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        mapbox={
            "style": "carto-positron",
            "center": {"lat": 13.0, "lon": 80.2},
            "zoom": 9,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        width=900,
        height=650,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Assignment detail table
    st.subheader("Assigned customers and stores")
    st.dataframe(map_df[["customer_id", "assigned_store", "distance_km", "demand"]].sort_values("customer_id"), use_container_width=True)


def main():
    st.set_page_config(page_title="Dark Store Optimisation", layout="wide")
    st.title("Dark Store Location Optimisation using OR-Tools")

    with st.sidebar:
        st.header("Inputs")
        store_count = st.number_input("Number of dark store candidates", min_value=2, max_value=100, value=8, step=1)
        customer_count = st.number_input("Number of customers", min_value=5, max_value=2000, value=40, step=1)
        max_distance_km = st.slider("Maximum distance between any dark store and customer (km)", min_value=2.0, max_value=30.0, value=10.0, step=0.5)
        max_open_stores = st.number_input("Maximum number of stores allowed to open (optional)", min_value=1, max_value=store_count, value=store_count, step=1)
        solve_button = st.button("Run optimisation")

    if not solve_button:
        st.info("Set the inputs and click 'Run optimisation' to solve the model.")
        return

    # Example assumptions
    stores, customers = generate_chennai_locations(store_count, customer_count)
    demand = np.random.randint(1, 25, size=customer_count).tolist()
    fixed_cost = [1200 + (10 * idx) for idx in range(store_count)]
    capacity = [max(25, int(sum(demand) / store_count * 1.2)) for _ in range(store_count)]

    try:
        result, assignment_detail, dist, _, _, _, open_stores = solve_dark_store_model(
            stores=stores,
            customers=customers,
            max_distance_km=max_distance_km,
            demand=demand,
            fixed_cost=fixed_cost,
            capacity=capacity,
            max_open_stores=max_open_stores,
        )
        render_model_results(stores, customers, result, assignment_detail, dist, demand, max_distance_km)
    except Exception as e:
        st.error(f"Model failed: {str(e)}")
        st.write("Try increasing the maximum distance, lowering the store count, or using fewer customers.")

    # Large-scale profit curve for 10,000 customers and 50 store candidates
    st.markdown("---")
    st.subheader("Large-scale profit curve (10,000 customers, 50 store candidates)")
    profit_curve = greedy_large_scale_profit_curve(
        store_count=50,
        customer_count=10000,
        max_distance_km=max_distance_km,
        max_opening_options=[2, 5, 10, 20, 25],
    )

    fig_profit = go.Figure(
        data=[
            go.Scatter(
                x=profit_curve["max_open_stores"],
                y=profit_curve["profit"],
                mode="lines+markers",
                line={"width": 3, "color": "#ff7f0e"},
                marker={"size": 10, "color": "#ff7f0e"},
                name="Profit",
            )
        ]
    )
    fig_profit.update_layout(
        title="Profit by maximum number of open stores",
        xaxis_title="Max stores allowed to open",
        yaxis_title="Profit ($)",
        template="plotly_white",
    )
    st.plotly_chart(fig_profit, use_container_width=True)

    st.dataframe(profit_curve, use_container_width=True)


if __name__ == "__main__":
    main()
