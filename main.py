from typing import List, Literal, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ortools.linear_solver import pywraplp
from datetime import datetime
import math

app = FastAPI(title="Battery MILP Solver API")


# =========================
# Pydantic models
# =========================
class Battery(BaseModel):
    power_kw: float
    capacity_kwh: float


class Site(BaseModel):
    max_import_kw: float
    max_export_kw: float


class Series(BaseModel):
    # Stuur alleen de nog te optimaliseren kwartieren mee.
    # In jouw proces zal dit typisch beginnen op 13:00.
    timestamps: List[str]
    load_kwh: List[float]
    pv_kwh: List[float]
    da_price_eur_per_mwh: List[float]
    id2_price_eur_per_mwh: List[float]
    id3_price_eur_per_mwh: List[float]


class FutureConstraint(BaseModel):
    timestamp: str
    target_soc_kwh: Optional[float] = None
    min_soc_kwh: Optional[float] = None
    max_soc_kwh: Optional[float] = None


class Options(BaseModel):
    final_soc_target_kwh: Optional[float] = None

    enforce_no_simultaneous_charge_discharge: bool = True
    enforce_no_simultaneous_import_export: bool = True

    energy_tax_eur_per_kwh: float = 0.072
    supplier_margin_eur_per_kwh: float = 0.02
    round_trip_efficiency: float = 0.90
    degradation_cost_eur_per_kwh_throughput: float = 0.01

    # True  = GEEN energiebelasting
    # False = WEL energiebelasting
    mloea: bool = True

    dod: float = 0.90
    solver_backend: Literal["SCIP", "CBC"] = "SCIP"

    # kW-max per maand
    include_kw_max: bool = True
    kw_max_free_threshold_kw: float = 50.0
    kw_max_mid_threshold_kw: float = 2000.0
    kw_max_mid_tariff_eur_per_kw: float = 3.57
    kw_max_high_tariff_eur_per_kw: float = 6.24


class SolveRequest(BaseModel):
    mode: Literal["dayahead", "id2", "id3"]

    time_step_minutes: int
    horizon_steps: int

    # In jouw proces draait de heroptimalisatie normaal om 13:00.
    # Voorbeeld: "2026-03-01T13:00:00"
    optimization_timestamp: str

    # Werkelijke SOC op optimization_timestamp
    actual_soc_kwh: float

    battery: Battery
    site: Site
    series: Series

    # Optionele constraints op specifieke toekomstige momenten
    future_constraints: List[FutureConstraint] = Field(default_factory=list)

    options: Options = Options()


# =========================
# Helpers
# =========================
def pad(arr: List[float], n: int) -> List[float]:
    if len(arr) == 0:
        return [0.0] * n
    if len(arr) < n:
        return arr + [arr[-1]] * (n - len(arr))
    return arr[:n]


def pad_timestamps(arr: List[str], n: int) -> List[str]:
    if len(arr) == 0:
        raise HTTPException(status_code=400, detail="series.timestamps mag niet leeg zijn.")
    if len(arr) < n:
        raise HTTPException(status_code=400, detail="series.timestamps moet exact horizon_steps lang zijn.")
    return arr[:n]


def parse_timestamp(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ongeldig timestamp-formaat: {ts}") from e


def validate_and_pad(req: SolveRequest) -> None:
    n = req.horizon_steps

    req.series.timestamps = pad_timestamps(req.series.timestamps, n)
    req.series.load_kwh = pad(req.series.load_kwh, n)
    req.series.pv_kwh = pad(req.series.pv_kwh, n)
    req.series.da_price_eur_per_mwh = pad(req.series.da_price_eur_per_mwh, n)
    req.series.id2_price_eur_per_mwh = pad(req.series.id2_price_eur_per_mwh, n)
    req.series.id3_price_eur_per_mwh = pad(req.series.id3_price_eur_per_mwh, n)

    if req.time_step_minutes <= 0:
        raise HTTPException(status_code=400, detail="time_step_minutes moet groter zijn dan 0.")

    if req.horizon_steps <= 0:
        raise HTTPException(status_code=400, detail="horizon_steps moet groter zijn dan 0.")

    if req.battery.power_kw <= 0:
        raise HTTPException(status_code=400, detail="battery.power_kw moet groter zijn dan 0.")

    if req.battery.capacity_kwh <= 0:
        raise HTTPException(status_code=400, detail="battery.capacity_kwh moet groter zijn dan 0.")

    if not (0 < req.options.round_trip_efficiency <= 1):
        raise HTTPException(status_code=400, detail="round_trip_efficiency moet tussen 0 en 1 liggen.")

    if not (0 < req.options.dod <= 1):
        raise HTTPException(status_code=400, detail="dod moet tussen 0 en 1 liggen.")

    if req.site.max_import_kw < 0 or req.site.max_export_kw < 0:
        raise HTTPException(status_code=400, detail="max_import_kw en max_export_kw moeten >= 0 zijn.")

    parsed = [parse_timestamp(ts) for ts in req.series.timestamps]
    for i in range(1, len(parsed)):
        if parsed[i] <= parsed[i - 1]:
            raise HTTPException(status_code=400, detail="series.timestamps moeten strikt oplopend zijn.")

    opt_dt = parse_timestamp(req.optimization_timestamp)

    # Deze solver verwacht dat de doorgestuurde horizon begint op of na het re-optimalisatiemoment.
    if parsed[0] < opt_dt:
        raise HTTPException(
            status_code=400,
            detail=(
                "De eerste timestamp in series.timestamps ligt vóór optimization_timestamp. "
                "Stuur alleen de nog te optimaliseren kwartieren mee, typisch vanaf 13:00."
            )
        )


def get_prices(req: SolveRequest) -> List[float]:
    if req.mode == "dayahead":
        return req.series.da_price_eur_per_mwh
    if req.mode == "id2":
        return req.series.id2_price_eur_per_mwh
    return req.series.id3_price_eur_per_mwh


def derive_soc_bounds(capacity_kwh: float, dod: float) -> tuple[float, float]:
    """
    DoD 90% => 5% buffer onder en 5% buffer boven.
    Voorbeeld:
    100 kWh, DoD 0.90 => min 5 kWh, max 95 kWh
    """
    buffer_fraction = (1.0 - dod) / 2.0
    soc_min_kwh = capacity_kwh * buffer_fraction
    soc_max_kwh = capacity_kwh * (1.0 - buffer_fraction)
    return soc_min_kwh, soc_max_kwh


def choose_solver(backend: str) -> pywraplp.Solver:
    solver = pywraplp.Solver.CreateSolver(backend)
    if solver:
        return solver

    fallback = "CBC" if backend == "SCIP" else "SCIP"
    solver = pywraplp.Solver.CreateSolver(fallback)
    if solver:
        return solver

    raise HTTPException(status_code=500, detail="Geen MILP solver beschikbaar (SCIP/CBC).")


def build_month_map(timestamps: List[str]) -> Dict[str, List[int]]:
    month_map: Dict[str, List[int]] = {}
    for idx, ts in enumerate(timestamps):
        dt = parse_timestamp(ts)
        key = f"{dt.year:04d}-{dt.month:02d}"
        month_map.setdefault(key, []).append(idx)
    return month_map


def timestamp_to_index(timestamps: List[str], target_timestamp: str) -> int:
    target_dt = parse_timestamp(target_timestamp)
    parsed = [parse_timestamp(ts) for ts in timestamps]

    for i, dt in enumerate(parsed):
        if dt == target_dt:
            return i

    raise HTTPException(
        status_code=400,
        detail=f"Timestamp {target_timestamp} niet gevonden in de doorgestuurde horizon."
    )


# =========================
# Core solver
# =========================
def solve_battery_dispatch(req: SolveRequest):
    validate_and_pad(req)

    n = req.horizon_steps
    dt_hours = req.time_step_minutes / 60.0
    prices_eur_per_mwh = get_prices(req)
    month_map = build_month_map(req.series.timestamps)

    max_step_energy = req.battery.power_kw * dt_hours
    max_import_step = req.site.max_import_kw * dt_hours
    max_export_step = req.site.max_export_kw * dt_hours

    soc_min_kwh, soc_max_kwh = derive_soc_bounds(
        req.battery.capacity_kwh,
        req.options.dod
    )

    if not (soc_min_kwh <= req.actual_soc_kwh <= soc_max_kwh):
        raise HTTPException(
            status_code=400,
            detail=(
                f"actual_soc_kwh={req.actual_soc_kwh} ligt buiten de DoD-grenzen "
                f"({soc_min_kwh:.3f} tot {soc_max_kwh:.3f})."
            )
        )

    if req.options.final_soc_target_kwh is not None:
        if not (soc_min_kwh <= req.options.final_soc_target_kwh <= soc_max_kwh):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"final_soc_target_kwh={req.options.final_soc_target_kwh} ligt buiten de DoD-grenzen "
                    f"({soc_min_kwh:.3f} tot {soc_max_kwh:.3f})."
                )
            )

    charge_eff = math.sqrt(req.options.round_trip_efficiency)
    discharge_eff = math.sqrt(req.options.round_trip_efficiency)

    # MLOEA:
    # True  = GEEN energiebelasting
    # False = WEL energiebelasting
    energy_tax = 0.0 if req.options.mloea else req.options.energy_tax_eur_per_kwh

    margin = req.options.supplier_margin_eur_per_kwh
    degradation = req.options.degradation_cost_eur_per_kwh_throughput

    load = req.series.load_kwh
    pv = req.series.pv_kwh

    solver = choose_solver(req.options.solver_backend)
    used_backend = solver.SolverVersion()

    # Decision vars
    charge = [solver.NumVar(0.0, max_step_energy, f"charge_{t}") for t in range(n)]
    discharge = [solver.NumVar(0.0, max_step_energy, f"discharge_{t}") for t in range(n)]
    grid_import = [solver.NumVar(0.0, max_import_step, f"grid_import_{t}") for t in range(n)]
    grid_export = [solver.NumVar(0.0, max_export_step, f"grid_export_{t}") for t in range(n)]
    soc = [solver.NumVar(soc_min_kwh, soc_max_kwh, f"soc_{t}") for t in range(n)]

    is_charging = [solver.IntVar(0, 1, f"is_charging_{t}") for t in range(n)]
    is_importing = [solver.IntVar(0, 1, f"is_importing_{t}") for t in range(n)]

    # kW-max per maand
    monthly_peak_kw_vars = {}
    peak_mid_kw_vars = {}
    peak_high_kw_vars = {}

    if req.options.include_kw_max:
        for month_key in month_map.keys():
            monthly_peak_kw_vars[month_key] = solver.NumVar(0.0, solver.infinity(), f"monthly_peak_kw_{month_key}")
            peak_mid_kw_vars[month_key] = solver.NumVar(0.0, solver.infinity(), f"peak_mid_kw_{month_key}")
            peak_high_kw_vars[month_key] = solver.NumVar(0.0, solver.infinity(), f"peak_high_kw_{month_key}")

    for t in range(n):
        if req.options.enforce_no_simultaneous_charge_discharge:
            solver.Add(charge[t] <= is_charging[t] * max_step_energy)
            solver.Add(discharge[t] <= (1 - is_charging[t]) * max_step_energy)

        if req.options.enforce_no_simultaneous_import_export:
            solver.Add(grid_import[t] <= is_importing[t] * max_import_step)
            solver.Add(grid_export[t] <= (1 - is_importing[t]) * max_export_step)

        # Energiebalans
        solver.Add(
            grid_import[t] + discharge[t] + pv[t] == load[t] + charge[t] + grid_export[t]
        )

        # Rolling horizon start vanuit actuele SOC op optimization_timestamp
        if t == 0:
            solver.Add(
                soc[t] == req.actual_soc_kwh + charge[t] * charge_eff - discharge[t] / discharge_eff
            )
        else:
            solver.Add(
                soc[t] == soc[t - 1] + charge[t] * charge_eff - discharge[t] / discharge_eff
            )

        # Maandpiek op basis van import in kW
        if req.options.include_kw_max:
            dt = parse_timestamp(req.series.timestamps[t])
            month_key = f"{dt.year:04d}-{dt.month:02d}"
            solver.Add(monthly_peak_kw_vars[month_key] >= grid_import[t] / dt_hours)

    # Optioneel eindtarget
    if req.options.final_soc_target_kwh is not None:
        solver.Add(soc[n - 1] == req.options.final_soc_target_kwh)

    # Tijdsafhankelijke future constraints
    for fc in req.future_constraints:
        idx = timestamp_to_index(req.series.timestamps, fc.timestamp)

        if fc.target_soc_kwh is not None:
            solver.Add(soc[idx] == fc.target_soc_kwh)

        if fc.min_soc_kwh is not None:
            solver.Add(soc[idx] >= fc.min_soc_kwh)

        if fc.max_soc_kwh is not None:
            solver.Add(soc[idx] <= fc.max_soc_kwh)

    kw_max_cost_terms = []
    if req.options.include_kw_max:
        free_threshold = req.options.kw_max_free_threshold_kw
        mid_threshold = req.options.kw_max_mid_threshold_kw
        mid_band_width = mid_threshold - free_threshold

        for month_key in month_map.keys():
            monthly_peak_kw = monthly_peak_kw_vars[month_key]
            peak_mid_kw = peak_mid_kw_vars[month_key]
            peak_high_kw = peak_high_kw_vars[month_key]

            # Deel tussen 50 kW en 2000 kW
            solver.Add(peak_mid_kw >= monthly_peak_kw - free_threshold)
            solver.Add(peak_mid_kw >= 0)
            solver.Add(peak_mid_kw <= mid_band_width)

            # Deel boven 2000 kW
            solver.Add(peak_high_kw >= monthly_peak_kw - mid_threshold)
            solver.Add(peak_high_kw >= 0)

            kw_max_cost_terms.append(
                peak_mid_kw * req.options.kw_max_mid_tariff_eur_per_kw
                + peak_high_kw * req.options.kw_max_high_tariff_eur_per_kw
            )

    # Energiekosten
    energy_cost_expr = solver.Sum(
        (
            grid_import[t] * (prices_eur_per_mwh[t] / 1000.0)
            + grid_import[t] * energy_tax
            + grid_import[t] * margin
            - grid_export[t] * (prices_eur_per_mwh[t] / 1000.0)
            + grid_export[t] * margin
            + (charge[t] + discharge[t]) * degradation
        )
        for t in range(n)
    )

    # kW-max kosten
    kw_max_cost_expr = solver.Sum(kw_max_cost_terms) if kw_max_cost_terms else 0.0

    solver.Minimize(energy_cost_expr + kw_max_cost_expr)

    status = solver.Solve()

    status_map = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "abnormal",
        pywraplp.Solver.NOT_SOLVED: "not_solved",
    }
    result_status = status_map.get(status, "unknown")

    if result_status not in ["optimal", "feasible"]:
        return {
            "status": result_status,
            "solver_backend": used_backend,
            "message": "Geen bruikbare oplossing gevonden."
        }

    schedule = []
    total_import_kwh = 0.0
    total_export_kwh = 0.0
    total_charge_kwh = 0.0
    total_discharge_kwh = 0.0

    total_market_import_cost_eur = 0.0
    total_energy_tax_eur = 0.0
    total_import_margin_eur = 0.0
    total_export_market_revenue_eur = 0.0
    total_export_margin_cost_eur = 0.0
    total_degradation_cost_eur = 0.0

    for t in range(n):
        import_val = grid_import[t].solution_value()
        export_val = grid_export[t].solution_value()
        charge_val = charge[t].solution_value()
        discharge_val = discharge[t].solution_value()
        soc_val = soc[t].solution_value()
        price_kwh = prices_eur_per_mwh[t] / 1000.0

        market_import_cost = import_val * price_kwh
        energy_tax_cost = import_val * energy_tax
        import_margin_cost = import_val * margin
        export_market_revenue = export_val * price_kwh
        export_margin_cost = export_val * margin
        degradation_cost = (charge_val + discharge_val) * degradation

        total_import_kwh += import_val
        total_export_kwh += export_val
        total_charge_kwh += charge_val
        total_discharge_kwh += discharge_val

        total_market_import_cost_eur += market_import_cost
        total_energy_tax_eur += energy_tax_cost
        total_import_margin_eur += import_margin_cost
        total_export_market_revenue_eur += export_market_revenue
        total_export_margin_cost_eur += export_margin_cost
        total_degradation_cost_eur += degradation_cost

        dt = parse_timestamp(req.series.timestamps[t])
        month_key = f"{dt.year:04d}-{dt.month:02d}"

        schedule.append({
            "step": t,
            "timestamp": req.series.timestamps[t],
            "month": month_key,
            "price_eur_per_mwh": round(prices_eur_per_mwh[t], 6),
            "load_kwh": round(load[t], 6),
            "pv_kwh": round(pv[t], 6),
            "charge_kwh": round(charge_val, 6),
            "discharge_kwh": round(discharge_val, 6),
            "import_kwh": round(import_val, 6),
            "export_kwh": round(export_val, 6),
            "import_kw_avg": round(import_val / dt_hours, 6),
            "soc_end_kwh": round(soc_val, 6),
            "is_charging": int(round(is_charging[t].solution_value())),
            "is_importing": int(round(is_importing[t].solution_value())),
            "market_import_cost_eur": round(market_import_cost, 6),
            "energy_tax_eur": round(energy_tax_cost, 6),
            "import_margin_eur": round(import_margin_cost, 6),
            "export_market_revenue_eur": round(export_market_revenue, 6),
            "export_margin_cost_eur": round(export_margin_cost, 6),
            "degradation_cost_eur": round(degradation_cost, 6),
            "net_cost_excl_kwmax_eur": round(
                market_import_cost
                + energy_tax_cost
                + import_margin_cost
                - export_market_revenue
                + export_margin_cost
                + degradation_cost,
                6
            )
        })

    monthly_kw_max_summary = {}
    total_kw_max_cost_eur = 0.0

    if req.options.include_kw_max:
        for month_key in month_map.keys():
            monthly_peak_kw_val = monthly_peak_kw_vars[month_key].solution_value()
            peak_mid_kw_val = peak_mid_kw_vars[month_key].solution_value()
            peak_high_kw_val = peak_high_kw_vars[month_key].solution_value()

            kw_max_cost_val = (
                peak_mid_kw_val * req.options.kw_max_mid_tariff_eur_per_kw
                + peak_high_kw_val * req.options.kw_max_high_tariff_eur_per_kw
            )
            total_kw_max_cost_eur += kw_max_cost_val

            monthly_kw_max_summary[month_key] = {
                "monthly_peak_kw": round(monthly_peak_kw_val, 6),
                "peak_mid_kw": round(peak_mid_kw_val, 6),
                "peak_high_kw": round(peak_high_kw_val, 6),
                "kw_max_cost_eur": round(kw_max_cost_val, 6)
            }

    return {
        "status": result_status,
        "solver_backend": used_backend,
        "objective_value_eur": round(solver.Objective().Value(), 6),
        "parameters": {
            "mode": req.mode,
            "optimization_timestamp": req.optimization_timestamp,
            "actual_soc_kwh": req.actual_soc_kwh,
            "time_step_minutes": req.time_step_minutes,
            "horizon_steps": req.horizon_steps,
            "battery_power_kw": req.battery.power_kw,
            "battery_capacity_kwh": req.battery.capacity_kwh,
            "soc_min_kwh": round(soc_min_kwh, 6),
            "soc_max_kwh": round(soc_max_kwh, 6),
            "dod": req.options.dod,
            "charge_efficiency": round(charge_eff, 6),
            "discharge_efficiency": round(discharge_eff, 6),
            "energy_tax_eur_per_kwh": req.options.energy_tax_eur_per_kwh,
            "supplier_margin_eur_per_kwh": req.options.supplier_margin_eur_per_kwh,
            "round_trip_efficiency": req.options.round_trip_efficiency,
            "degradation_cost_eur_per_kwh_throughput": req.options.degradation_cost_eur_per_kwh_throughput,
            "mloea": req.options.mloea,
            "final_soc_target_kwh": req.options.final_soc_target_kwh,
            "max_step_energy_kwh": round(max_step_energy, 6),
            "max_import_step_kwh": round(max_import_step, 6),
            "max_export_step_kwh": round(max_export_step, 6),
            "include_kw_max": req.options.include_kw_max,
            "kw_max_free_threshold_kw": req.options.kw_max_free_threshold_kw,
            "kw_max_mid_threshold_kw": req.options.kw_max_mid_threshold_kw,
            "kw_max_mid_tariff_eur_per_kw": req.options.kw_max_mid_tariff_eur_per_kw,
            "kw_max_high_tariff_eur_per_kw": req.options.kw_max_high_tariff_eur_per_kw
        },
        "summary": {
            "total_import_kwh": round(total_import_kwh, 6),
            "total_export_kwh": round(total_export_kwh, 6),
            "total_charge_kwh": round(total_charge_kwh, 6),
            "total_discharge_kwh": round(total_discharge_kwh, 6),
            "final_soc_kwh": round(schedule[-1]["soc_end_kwh"], 6),
            "total_market_import_cost_eur": round(total_market_import_cost_eur, 6),
            "total_energy_tax_eur": round(total_energy_tax_eur, 6),
            "total_import_margin_eur": round(total_import_margin_eur, 6),
            "total_export_market_revenue_eur": round(total_export_market_revenue_eur, 6),
            "total_export_margin_cost_eur": round(total_export_margin_cost_eur, 6),
            "total_degradation_cost_eur": round(total_degradation_cost_eur, 6),
            "total_kw_max_cost_eur": round(total_kw_max_cost_eur, 6),
            "net_total_cost_eur": round(
                total_market_import_cost_eur
                + total_energy_tax_eur
                + total_import_margin_eur
                - total_export_market_revenue_eur
                + total_export_margin_cost_eur
                + total_degradation_cost_eur
                + total_kw_max_cost_eur,
                6
            )
        },
        "monthly_kw_max": monthly_kw_max_summary,
        "schedule": schedule
    }


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/solve/dayahead")
def solve_dayahead(req: SolveRequest):
    if req.mode != "dayahead":
        raise HTTPException(status_code=400, detail="Gebruik mode='dayahead' voor dit endpoint.")
    return solve_battery_dispatch(req)


@app.post("/solve/id2")
def solve_id2(req: SolveRequest):
    if req.mode != "id2":
        raise HTTPException(status_code=400, detail="Gebruik mode='id2' voor dit endpoint.")
    return solve_battery_dispatch(req)


@app.post("/solve/id3")
def solve_id3(req: SolveRequest):
    if req.mode != "id3":
        raise HTTPException(status_code=400, detail="Gebruik mode='id3' voor dit endpoint.")
    return solve_battery_dispatch(req)


@app.post("/solve")
def solve_generic(req: SolveRequest):
    return solve_battery_dispatch(req)

@app.post("/forecast")
def forecast(req: SolveRequest):
    return solve_battery_dispatch(req)
