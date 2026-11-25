"""Visual report generator for a saved PSO solution.

The report is a single self-contained HTML file with embedded PNG charts,
plus standalone PNG charts and CSV tables.  It reads the solution JSON
written by ``main.py`` and the original dataset (through ``data_loader``, the
only CSV reader).

Generated files (default ``output/report/``)
-------------------------------------------
report.html             self-contained HTML report
convergence.png         loss / convergence dashboard
vehicle_timeline.png    vehicle-day Gantt chart
cost_breakdown.png      stacked cost components and total pie
trip_capacity.png       weight/volume utilisation and customer counts
route_map.png           route map and trip distance/cost scatter
route_map.html          interactive map (Leaflet; street map, Esri satellite or
                        satellite + labels, switchable; needs internet
                        for the tiles; routes follow the roads when
                        route_geometry.json is available)
summary.csv             solution, search and configuration summary
vehicles.csv            per-vehicle aggregates
trips.csv               per-trip details
"""

from __future__ import annotations

import base64
import csv
import html
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import ProblemData


VEHICLE_COLORS = ["#2E86AB", "#5DADE2", "#E67E22", "#F1C40F", "#8E44AD"]
OVERTIME_COLOR = "#E74C3C"
COST_COMPONENTS = [
    ("fuel_cost", "Fuel", "#3498DB"),
    ("maintenance_cost", "Maintenance", "#95A5A6"),
    ("wage_base_cost", "Wages (base)", "#2ECC71"),
    # Paid use of the 10 % volume reserve (90-100 % full): a real cost, not a penalty.
    ("overflow_cost", "Volume overflow fee", "#E67E22"),
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _num(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _maybe_num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def _round_row(row: dict, digits: int = 3) -> dict:
    return {
        key: round(value, digits) if isinstance(value, float) else value
        for key, value in row.items()
    }


def _csv_number(value, digits: int = 3):
    out = _maybe_num(value)
    return round(out, digits) if np.isfinite(out) else ""


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _img_tag(path: Path) -> str:
    if not path.exists():
        return f"<p class='missing'>Missing chart: {html.escape(path.name)}</p>"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f"<img alt='{html.escape(path.stem)}' "
        f"src='data:image/png;base64,{encoded}'>"
    )


def _html_table(
    headers: list[str],
    rows: list[list[object]],
    numeric_columns: set[int] | None = None,
) -> str:
    numeric_columns = numeric_columns or set()
    parts = ["<div class='table-wrap'><table><thead><tr>"]
    parts.extend(f"<th>{html.escape(str(h))}</th>" for h in headers)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for index, cell in enumerate(row):
            cls = " class='num'" if index in numeric_columns else ""
            parts.append(f"<td{cls}>{html.escape(str(cell))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------
def _trip_records(data: ProblemData, solution: dict) -> list[dict]:
    records: list[dict] = []
    for index, trip in enumerate(solution.get("trips", []), 1):
        k = int(trip["vehicle_index"])
        vehicle = data.vehicles[k]
        weight_cap = float(vehicle.max_weight_kg)
        volume_cap = float(data.vehicle_volume_cap[k])
        weight = _num(trip.get("weight_kg"))
        volume = _num(trip.get("volume_m3"))
        time_min = _num(trip.get("time_min"))
        records.append(
            {
                "trip": index,
                "vehicle_index": k,
                "vehicle_id": int(vehicle.vehicle_id),
                "vehicle_type": vehicle.type_id,
                "vehicle_label": f"V{vehicle.vehicle_id} {vehicle.type_id}",
                "customers": len(trip.get("customers", [])),
                "distance_km": _num(trip.get("distance_km")),
                "time_min": time_min,
                "time_h": time_min / 60.0,
                "weight_kg": weight,
                "volume_m3": volume,
                "weight_cap_kg": weight_cap,
                "volume_cap_m3": volume_cap,
                "weight_util_pct": 100.0 * weight / weight_cap if weight_cap else 0.0,
                "volume_util_pct": 100.0 * volume / volume_cap if volume_cap else 0.0,
                "fuel_cost": _num(trip.get("fuel_cost")),
                "maintenance_cost": _num(trip.get("maintenance_cost")),
                "wage_base_cost": _num(trip.get("wage_base_cost")),
                "overtime_cost": _num(trip.get("overtime_cost")),
                "base_cost_eur": _num(trip.get("base_cost_eur")),
                "overflow_m3": _num(trip.get("overflow_m3")),
                "overflow_cost": _num(trip.get("overflow_cost")),
                "cost_eur": _num(trip.get("cost_eur")),
                "penalty": _num(trip.get("penalty")),
                "feasible": bool(trip.get("feasible", False)),
            }
        )
    return records


def _vehicle_records(
    data: ProblemData, solution: dict, trips: list[dict]
) -> list[dict]:
    total_min = solution.get("vehicle_total_min") or []
    overtime_min = solution.get("vehicle_overtime_min") or []
    overtime_cost = solution.get("vehicle_overtime_cost") or []

    records: list[dict] = []
    for k, vehicle in enumerate(data.vehicles):
        rows = [row for row in trips if row["vehicle_index"] == k]
        fuel = sum(row["fuel_cost"] for row in rows)
        maintenance = sum(row["maintenance_cost"] for row in rows)
        wages = sum(row["wage_base_cost"] for row in rows)
        ot_cost = _num(overtime_cost[k]) if k < len(overtime_cost) else 0.0
        overflow = sum(row["overflow_cost"] for row in rows)
        records.append(
            {
                "vehicle_index": k,
                "vehicle_id": int(vehicle.vehicle_id),
                "type": vehicle.type_id,
                "model": vehicle.model,
                "trips": len(rows),
                "customers": sum(row["customers"] for row in rows),
                "distance_km": sum(row["distance_km"] for row in rows),
                "time_h": _num(total_min[k]) / 60.0
                if k < len(total_min)
                else sum(row["time_h"] for row in rows),
                "overtime_h": _num(overtime_min[k]) / 60.0
                if k < len(overtime_min)
                else 0.0,
                "fuel_cost": fuel,
                "maintenance_cost": maintenance,
                "wage_base_cost": wages,
                "overtime_cost": ot_cost,
                "overflow_cost": overflow,
                "total_cost": fuel + maintenance + wages + ot_cost + overflow,
                "weight_cap_kg": float(vehicle.max_weight_kg),
                "volume_cap_m3": float(data.vehicle_volume_cap[k]),
            }
        )
    return records


def _day_window_rows(payload: dict, solution: dict) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    window = payload.get("day_window") or {}
    if window.get("hard_day_end"):
        rows.append((
            "Daily window",
            "Rule",
            f"start {window.get('day_start')}, overtime 2x after 8 h, last delivery by "
            f"{window.get('last_delivery_by')}, back by {window.get('hard_day_end')}",
        ))
    if solution.get("longest_day_min") is not None:
        rows.append(("Daily window", "Longest vehicle-day",
                     f"{_num(solution.get('longest_day_min')) / 60:,.2f} h"))
    ots = payload.get("overtime_summary") or {}
    if ots:
        rows.append(("Daily window", "Unavoidable overtime",
                     f"{_num(ots.get('unavoidable_overtime_h')):,.2f} h"))
        rows.append(("Daily window", "Avoidable overtime",
                     f"{_num(ots.get('avoidable_overtime_h')):,.2f} h"))
    for row in payload.get("day_schedule") or []:
        if row.get("finish"):
            rows.append((
                "Daily window",
                f"Vehicle {row['vehicle_id']}",
                f"{row['start']}-{row['finish']}, last delivery {row['last_delivery']}",
            ))
    return rows


def _summary_rows(payload: dict, solution: dict) -> list[tuple[str, str, str]]:
    config = payload.get("config", {})
    check_errors = payload.get("check_errors") or []
    feasible_cost = payload.get("best_feasible_cost")
    rows = [
        ("Solution", "True cost", f"{_num(solution.get('true_cost_eur')):,.2f} EUR"),
        ("Solution", "Penalty", f"{_num(solution.get('penalty')):,.2f} EUR"),
        ("Solution", "Objective", f"{_num(solution.get('objective')):,.2f} EUR"),
        ("Solution", "Feasible", "yes" if solution.get("feasible") else "no"),
        ("Solution", "Trips", f"{int(_num(solution.get('n_trips')))}"),
        ("Solution", "Distance", f"{_num(solution.get('total_distance_km')):,.1f} km"),
        ("Solution", "Working time", f"{_num(solution.get('total_time_min')) / 60:,.2f} h"),
        ("Solution", "Overtime", f"{_num(solution.get('shift_over_min')) / 60:,.2f} h"),
        (
            "Solution",
            "Vehicle-day overtime cost",
            f"{_num(solution.get('day_overtime_cost')):,.2f} EUR",
        ),
        ("Solution", "Total weight", f"{_num(solution.get('total_weight_kg')):,.1f} kg"),
        ("Solution", "Total volume", f"{_num(solution.get('total_volume_m3')):,.2f} m3"),
        ("Solution", "Method", str(solution.get("method", ""))),
        *_day_window_rows(payload, solution),
        ("Search", "Runtime", f"{_num(payload.get('runtime_sec')):,.1f} s"),
        ("Search", "Decoder evaluations", f"{int(_num(payload.get('evaluations'))):,}"),
        (
            "Search",
            "Best feasible cost",
            f"{_num(feasible_cost):,.2f} EUR" if feasible_cost is not None else "n/a",
        ),
        (
            "Search",
            "Final check_solution",
            "no errors" if not check_errors else "; ".join(str(e) for e in check_errors),
        ),
        ("Config", "Swarm size", str(config.get("swarm_size", ""))),
        ("Config", "Iterations", str(config.get("iterations", ""))),
        ("Config", "Seed", str(config.get("seed", ""))),
        (
            "Config",
            "Local search",
            f"every {config.get('local_search_interval', '?')} iterations, "
            f"budget {config.get('local_search_budget', '?')}",
        ),
        (
            "Config",
            "Final local search budget",
            str(config.get("final_local_search_budget", "")),
        ),
    ]
    return rows


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _plot_convergence(
    history: list[dict], path: Path, final_cost: float | None = None
) -> None:
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], hspace=0.38, wspace=0.24)
    ax_cost = fig.add_subplot(grid[0, :])
    ax_penalty = fig.add_subplot(grid[1, 0])
    ax_evals = fig.add_subplot(grid[1, 1])

    if not history:
        for ax in (ax_cost, ax_penalty, ax_evals):
            ax.text(
                0.5,
                0.5,
                "No search history in this output",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6B7280",
            )
            ax.set_axis_off()
        fig.savefig(path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return

    iterations = [int(row.get("iteration", i + 1)) for i, row in enumerate(history)]
    best_score = [_maybe_num(row.get("best_score")) for row in history]
    feasible = [_maybe_num(row.get("best_feasible_cost")) for row in history]
    penalty = [_maybe_num(row.get("best_penalty")) for row in history]
    evaluations = [int(_num(row.get("evaluations"))) for row in history]

    ax_cost.plot(
        iterations, best_score, color="#2C3E50", lw=2.2, label="Best objective"
    )
    if any(np.isfinite(feasible)):
        ax_cost.plot(
            iterations,
            feasible,
            color="#27AE60",
            lw=1.8,
            ls="--",
            label="Best feasible cost",
        )
    ax_cost.set_xlabel("Iteration")
    ax_cost.set_ylabel("EUR")
    ax_cost.set_title("Convergence / loss curve")
    ax_cost.legend(loc="upper right")
    ax_cost.grid(True, alpha=0.25, ls="--")

    final_value = _maybe_num(final_cost)

    if np.isfinite(best_score[-1]) and not np.isfinite(final_value):
        ax_cost.annotate(
            f"{best_score[-1]:,.2f} EUR",
            xy=(iterations[-1], best_score[-1]),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color="#2C3E50",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D1D5DB"},
        )

    # The final trip-level polish runs after the last PSO iteration, so its
    # result is not in the per-iteration history.  Show it as a separate marker
    # instead of silently changing the last history row.
    if np.isfinite(final_value):
        final_x = iterations[-1] + 1
        ax_cost.scatter(
            [final_x],
            [final_value],
            color="#27AE60",
            marker="D",
            s=48,
            zorder=5,
            label="Final polished cost",
        )
        ax_cost.annotate(
            f"final {final_value:,.2f} EUR",
            xy=(final_x, final_value),
            xytext=(-10, 18),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#1E8449",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#A7D7B9"},
        )
        ax_cost.set_xlim(iterations[0] - 1, final_x + 2)
        ax_cost.legend(loc="upper right")

    finite_penalty = [p for p in penalty if np.isfinite(p)]
    if finite_penalty and max(finite_penalty) > 1e-9:
        ax_penalty.plot(iterations, penalty, color=OVERTIME_COLOR, lw=1.8)
        ax_penalty.set_yscale("symlog", linthresh=1.0)
        ax_penalty.set_ylabel("Penalty (EUR, symlog)")
    else:
        ax_penalty.axhline(0, color="#27AE60", lw=2)
        ax_penalty.text(
            0.5,
            0.58,
            "All recorded incumbents feasible\n(penalty = 0)",
            transform=ax_penalty.transAxes,
            ha="center",
            va="center",
            color="#27AE60",
            fontsize=10,
        )
        ax_penalty.set_ylim(-1, 1)
        ax_penalty.set_yticks([0])
    ax_penalty.set_xlabel("Iteration")
    ax_penalty.set_title("Best penalty")
    ax_penalty.grid(True, alpha=0.25, ls="--")

    ax_evals.plot(iterations, evaluations, color="#8E44AD", lw=1.8)
    ax_evals.set_xlabel("Iteration")
    ax_evals.set_ylabel("Decoder calls")
    ax_evals.set_title("Search effort")
    ax_evals.grid(True, alpha=0.25, ls="--")

    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def _plot_vehicle_timeline(data: ProblemData, solution: dict, path: Path) -> None:
    """Gantt chart of each vehicle-day on the wall clock (08:00 start)."""
    p = data.params
    t0 = p.minutes_after_start("00:00")  # negative: minutes from midnight
    start_h = -t0 / 60.0                 # day start as a clock hour (8.0)
    fig, ax = plt.subplots(figsize=(13, 5.0))
    max_end = start_h + 8.5
    trips = solution.get("trips", [])
    last_delivery = solution.get("vehicle_last_delivery_min") or []

    for k, vehicle in enumerate(data.vehicles):
        color = VEHICLE_COLORS[k % len(VEHICLE_COLORS)]
        vehicle_trips = [
            (index, trip)
            for index, trip in enumerate(trips, 1)
            if int(trip["vehicle_index"]) == k
        ]
        start = start_h
        for j, (trip_index, trip) in enumerate(vehicle_trips):
            duration = _num(trip.get("time_min")) / 60.0
            ax.barh(
                k,
                duration,
                left=start,
                height=0.55,
                color=color,
                alpha=0.55 + 0.15 * (j % 3),
                edgecolor="white",
                linewidth=0.8,
            )
            if duration >= 0.45:
                ax.text(
                    start + duration / 2,
                    k,
                    f"T{trip_index}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#1B2631",
                )
            start += duration
        max_end = max(max_end, start)
        if vehicle_trips and k < len(last_delivery):
            ld = start_h + _num(last_delivery[k]) / 60.0
            ax.plot([ld], [k], marker="v", color="#1B2631", ms=6, zorder=5)
        if vehicle_trips:
            ax.text(
                start + 0.08,
                k,
                p.clock((start - start_h) * 60.0),
                va="center",
                fontsize=8,
                color="#2C3E50",
            )

    shift_end = start_h + p.shift_minutes / 60.0
    ax.axvline(shift_end, color=OVERTIME_COLOR, ls="--", lw=1.5,
               label=f"shift ends {p.clock(p.shift_minutes)} (2x after)")
    night = p.night_from_min
    day_end = p.day_end_min
    last_by = p.last_delivery_min
    if night is None and last_by is not None and day_end is not None:
        ld_h = start_h + last_by / 60.0
        ax.axvspan(shift_end, start_h + day_end / 60.0, color=OVERTIME_COLOR, alpha=0.06)
        ax.axvline(ld_h, color="#8E44AD", ls="--", lw=1.2,
                   label=f"{p.last_delivery_by}: last delivery")
    elif night is not None:
        night_h = start_h + night / 60.0
        ax.axvspan(shift_end, night_h, color=OVERTIME_COLOR, alpha=0.06)
        ax.axvline(night_h, color="#8E44AD", ls="--", lw=1.2,
                   label=f"{p.night_rate_from}: last delivery, night rate "
                         f"x{p.night_multiplier:g}")
        if day_end is not None:
            ax.axvspan(night_h, start_h + day_end / 60.0, color="#8E44AD", alpha=0.08)
    elif max_end > shift_end:
        ax.axvspan(shift_end, max_end, color=OVERTIME_COLOR, alpha=0.06)
    if day_end is not None:
        end_h = start_h + day_end / 60.0
        ax.axvline(end_h, color="#C0392B", lw=1.8,
                   label=f"{p.hard_day_end}: hard end of day")
        max_end = max(max_end, end_h)
    ax.plot([], [], "v", color="#1B2631", label="last delivery")
    ax.set_yticks(range(len(data.vehicles)))
    ax.set_yticklabels(
        [f"V{v.vehicle_id}\n{v.type_id}" for v in data.vehicles]
    )
    ax.invert_yaxis()
    ticks = list(range(int(start_h), int(max_end) + 2, 2))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{h:02d}:00" for h in ticks])
    ax.set_xlabel("Clock time")
    ax.set_title("Vehicle-day timeline")
    ax.set_xlim(start_h, max_end + 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25, ls="--")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def _plot_cost_breakdown(vehicles: list[dict], path: Path) -> None:
    fig, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    labels = [f"V{v['vehicle_id']}\n{v['type']}" for v in vehicles]
    x = np.arange(len(vehicles))
    bottom = np.zeros(len(vehicles))

    for key, label, color in COST_COMPONENTS:
        values = np.array([v[key] for v in vehicles], dtype=float)
        ax_bar.bar(
            x,
            values,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
        )
        bottom += values

    overtime = np.array([v["overtime_cost"] for v in vehicles], dtype=float)
    ax_bar.bar(
        x,
        overtime,
        bottom=bottom,
        label="OT premium",
        color=OVERTIME_COLOR,
        edgecolor="white",
    )
    bottom += overtime

    for index, total in enumerate(bottom):
        ax_bar.text(
            index, total, f"{total:.0f}", ha="center", va="bottom", fontsize=8
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("EUR")
    ax_bar.set_title("Cost breakdown by vehicle")
    ax_bar.legend(ncol=2, fontsize=8, loc="upper left")
    ax_bar.grid(axis="y", alpha=0.25, ls="--")

    totals = [sum(v[key] for v in vehicles) for key, _, _ in COST_COMPONENTS]
    totals.append(float(overtime.sum()))
    pie_labels = [label for _, label, _ in COST_COMPONENTS] + ["OT premium"]
    pie_colors = [color for _, _, color in COST_COMPONENTS] + [OVERTIME_COLOR]
    grand = sum(totals)
    if grand > 1e-9:
        # Small parts (e.g. the overflow fee, ~0.1 %) get no wedge label, which
        # would overlap; the legend names every part with its EUR amount.
        big = [t / grand >= 0.01 for t in totals]
        ax_pie.pie(
            totals,
            labels=[lab if b else "" for lab, b in zip(pie_labels, big)],
            colors=pie_colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 1.0 else "",
            startangle=90,
            textprops={"fontsize": 8},
            wedgeprops={"edgecolor": "white"},
        )
        ax_pie.legend(
            [f"{lab}: {t:,.2f} EUR" for lab, t in zip(pie_labels, totals)],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            fontsize=7.5,
            ncol=2,
            frameon=False,
        )
    else:
        ax_pie.text(
            0.5,
            0.5,
            "No cost data",
            transform=ax_pie.transAxes,
            ha="center",
            va="center",
            color="#6B7280",
        )
        ax_pie.set_axis_off()
    ax_pie.set_title("Total cost composition")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def _plot_trip_capacity(trips: list[dict], path: Path) -> None:
    fig, (ax_capacity, ax_count) = plt.subplots(
        1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.1, 1]}
    )
    y = np.arange(len(trips))
    labels = [f"T{row['trip']} V{row['vehicle_id']}" for row in trips]
    weight = np.array([row["weight_util_pct"] for row in trips], dtype=float)
    volume = np.array([row["volume_util_pct"] for row in trips], dtype=float)

    ax_capacity.barh(
        y - 0.18, weight, height=0.36, color="#2E86AB", label="Weight"
    )
    ax_capacity.barh(
        y + 0.18,
        volume,
        height=0.36,
        color="#E67E22",
        label="Volume (usable)",
    )
    ax_capacity.axvline(
        100, color=OVERTIME_COLOR, ls="--", lw=1.2, label="Capacity"
    )
    ax_capacity.set_yticks(y)
    ax_capacity.set_yticklabels(labels, fontsize=8)
    ax_capacity.invert_yaxis()
    ax_capacity.set_xlabel("Utilisation (%)")
    ax_capacity.set_title("Trip capacity utilisation")
    upper = max(
        110.0,
        float(weight.max(initial=0.0)),
        float(volume.max(initial=0.0)),
    )
    ax_capacity.set_xlim(0, upper + 10.0)
    ax_capacity.legend(loc="lower right", fontsize=8)
    ax_capacity.grid(axis="x", alpha=0.25, ls="--")

    counts = np.array([row["customers"] for row in trips], dtype=int)
    colors = [
        VEHICLE_COLORS[row["vehicle_index"] % len(VEHICLE_COLORS)]
        for row in trips
    ]
    ax_count.barh(y, counts, height=0.6, color=colors)
    ax_count.axvline(10, color=OVERTIME_COLOR, ls="--", lw=1.0, label="min 10")
    ax_count.axvline(50, color="#2C3E50", ls=":", lw=1.0, label="max 50")
    for index, count in zip(y, counts):
        ax_count.text(
            count + 0.5, index, str(int(count)), va="center", fontsize=7
        )
    ax_count.set_yticks(y)
    ax_count.set_yticklabels([])
    ax_count.invert_yaxis()
    ax_count.set_xlabel("Customers")
    ax_count.set_title("Customers per trip")
    ax_count.set_xlim(0, max(55.0, float(counts.max(initial=0)) + 5.0))
    ax_count.legend(loc="lower right", fontsize=8)
    ax_count.grid(axis="x", alpha=0.25, ls="--")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def _geometry_by_trip(solution: dict, geometry: dict | None) -> list[list[dict] | None]:
    """Road legs for each solution trip (None = no geometry for that trip)."""
    out: list[list[dict] | None] = []
    geo_trips = (geometry or {}).get("trips", [])
    lookup = {tuple(int(c) for c in g["customers"]): g["legs"] for g in geo_trips}
    for trip in solution.get("trips", []):
        out.append(lookup.get(tuple(int(c) for c in trip.get("customers", []))))
    return out


def route_mode_caption(data: ProblemData, geometry: dict | None) -> str:
    if data.road_mode and geometry:
        s = geometry.get("summary", {})
        return (
            "Routes follow the OpenStreetMap road graph (GA team's road network): "
            f"{s.get('legs', 0)} legs, {s.get('exact', 0)} identical to the cost "
            f"matrix, {s.get('close', 0)} within {100 * s.get('tolerance', 0.02):.0f}%, "
            f"{s.get('differs', 0)} differing, {s.get('unroutable', 0)} not routable "
            "(dashed). Costs and times use the directed road matrices."
        )
    if data.road_mode:
        return (
            "Costs and times use the directed OSM road matrices, but the map draws "
            "straight lines between stops because no route_geometry.json was found "
            "(run python -m experiments.road_geometry)."
        )
    return (
        "Straight-line distance mode (haversine x 1.3): routes are drawn depot to "
        "customers and back as straight lines."
    )


def _plot_route_map(
    data: ProblemData, solution: dict, trips: list[dict], path: Path,
    geometry: dict | None = None,
) -> None:
    legs_by_trip = _geometry_by_trip(solution, geometry)
    fig, (ax_map, ax_scatter) = plt.subplots(
        1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.45, 1]}
    )

    ax_map.scatter(
        data.lon[1:],
        data.lat[1:],
        s=8,
        c="#BDC3C7",
        alpha=0.65,
        zorder=1,
        label="Customers",
    )
    ax_map.scatter(
        [data.lon[0]],
        [data.lat[0]],
        marker="*",
        s=220,
        c="#2C3E50",
        zorder=4,
        label="Depot",
    )

    for k, vehicle in enumerate(data.vehicles):
        color = VEHICLE_COLORS[k % len(VEHICLE_COLORS)]
        vehicle_trips = [
            trip
            for trip in solution.get("trips", [])
            if int(trip["vehicle_index"]) == k
        ]
        for t_index, trip in enumerate(solution.get("trips", [])):
            if int(trip["vehicle_index"]) != k:
                continue
            customers = [int(c) for c in trip.get("customers", [])]
            if not customers:
                continue
            legs = legs_by_trip[t_index]
            if legs:
                for leg in legs:
                    lat, lon = zip(*leg["points"])
                    dashed = leg["status"] == "unroutable"
                    ax_map.plot(lon, lat, color=color, lw=0.9, alpha=0.7, zorder=2,
                                ls="--" if dashed else "-")
                continue
            xs = [data.lon[0]] + [data.lon[c] for c in customers] + [data.lon[0]]
            ys = [data.lat[0]] + [data.lat[c] for c in customers] + [data.lat[0]]
            ax_map.plot(xs, ys, color=color, lw=1.0, alpha=0.55, zorder=2)
        served = [
            int(c)
            for trip in vehicle_trips
            for c in trip.get("customers", [])
        ]
        if served:
            ax_map.scatter(
                data.lon[served],
                data.lat[served],
                s=15,
                c=color,
                alpha=0.9,
                zorder=3,
                label=f"V{vehicle.vehicle_id} {vehicle.type_id}",
            )

    ax_map.set_title(
        "Delivery routes on the road network" if any(legs_by_trip)
        else "Delivery routes (straight lines between stops)"
    )
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.legend(fontsize=7, loc="best")
    ax_map.grid(True, alpha=0.2, ls="--")

    for k, vehicle in enumerate(data.vehicles):
        rows = [row for row in trips if row["vehicle_index"] == k]
        if not rows:
            continue
        ax_scatter.scatter(
            [row["distance_km"] for row in rows],
            [row["cost_eur"] for row in rows],
            s=[max(20.0, 4.0 * row["customers"]) for row in rows],
            color=VEHICLE_COLORS[k % len(VEHICLE_COLORS)],
            alpha=0.75,
            edgecolor="white",
            label=f"V{vehicle.vehicle_id} {vehicle.type_id}",
        )

    ax_scatter.set_xlabel(
        "Trip road distance (km)" if data.road_mode else "Trip distance (km)"
    )
    ax_scatter.set_ylabel("Standalone trip cost (EUR)")
    ax_scatter.set_title("Trip distance vs standalone cost")
    ax_scatter.legend(fontsize=7)
    ax_scatter.grid(True, alpha=0.25, ls="--")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def _build_html(
    payload: dict,
    solution: dict,
    trips: list[dict],
    vehicles: list[dict],
    summary_rows: list[tuple[str, str, str]],
    charts: dict[str, Path],
    out_dir: Path,
    route_caption: str | None = None,
) -> str:
    feasible = bool(solution.get("feasible"))
    check_errors = payload.get("check_errors") or []
    status_class = "good" if feasible and not check_errors else "bad"
    status_text = "Feasible" if feasible else "Infeasible"
    if check_errors:
        status_text += " / check errors"

    cards = [
        ("True cost", f"{_num(solution.get('true_cost_eur')):,.2f} EUR", "good"),
        ("Penalty", f"{_num(solution.get('penalty')):,.2f} EUR", "good"),
        ("Trips", f"{int(_num(solution.get('n_trips')))}", ""),
        ("Distance", f"{_num(solution.get('total_distance_km')):,.1f} km", ""),
        ("Working time", f"{_num(solution.get('total_time_min')) / 60:,.2f} h", ""),
        ("Overtime", f"{_num(solution.get('shift_over_min')) / 60:,.2f} h", "warn"),
        ("Runtime", f"{_num(payload.get('runtime_sec')):,.1f} s", ""),
        ("Evaluations", f"{int(_num(payload.get('evaluations'))):,}", ""),
    ]
    card_html = "".join(
        f"<div class='card {cls}'><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div></div>"
        for label, value, cls in cards
    )

    summary_table = _html_table(
        ["Section", "Metric", "Value"],
        [[section, key, value] for section, key, value in summary_rows],
    )

    vehicle_rows = [
        [
            f"V{v['vehicle_id']}",
            v["type"],
            v["trips"],
            v["customers"],
            f"{v['distance_km']:.1f}",
            f"{v['time_h']:.2f}",
            f"{v['overtime_h']:.2f}",
            f"{v['fuel_cost']:.2f}",
            f"{v['maintenance_cost']:.2f}",
            f"{v['wage_base_cost']:.2f}",
            f"{v['overtime_cost']:.2f}",
            f"{v['total_cost']:.2f}",
        ]
        for v in vehicles
    ]
    vehicle_table = _html_table(
        [
            "Vehicle",
            "Type",
            "Trips",
            "Customers",
            "Distance (km)",
            "Hours",
            "Overtime (h)",
            "Fuel (EUR)",
            "Maint. (EUR)",
            "Wages (EUR)",
            "OT premium (EUR)",
            "Total (EUR)",
        ],
        vehicle_rows,
        numeric_columns=set(range(2, 12)),
    )

    trip_rows = [
        [
            row["trip"],
            f"V{row['vehicle_id']} {row['vehicle_type']}",
            row["customers"],
            f"{row['distance_km']:.1f}",
            f"{row['time_h']:.2f}",
            f"{row['weight_kg']:.1f}",
            f"{row['volume_m3']:.3f}",
            f"{row['weight_util_pct']:.1f}",
            f"{row['volume_util_pct']:.1f}",
            f"{row['overflow_cost']:.2f}",
            f"{row['cost_eur']:.2f}",
            "yes" if row["feasible"] else "no",
        ]
        for row in trips
    ]
    trip_table = _html_table(
        [
            "Trip",
            "Vehicle",
            "Customers",
            "Distance (km)",
            "Time (h)",
            "Weight (kg)",
            "Volume (m3)",
            "Weight util. (%)",
            "Volume util. (%)",
            "Overflow fee (EUR)",
            "Cost (EUR)",
            "Feasible",
        ],
        trip_rows,
        numeric_columns={0, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    )

    chart_sections = [
        (
            "Convergence and loss",
            charts["convergence"],
            "Best objective, best feasible cost, penalty and cumulative decoder "
            "evaluations over the PSO iterations.",
        ),
        (
            "Vehicle-day timeline",
            charts["vehicle_timeline"],
            "Each bar is one trip; trips are shown sequentially in solution order. "
            "Each bar is one trip. Everything after the 8-hour shift is overtime at 2x. The last customer must be served by the last-delivery line; after it only the drive back is allowed, until the hard end of the day. Triangles mark each vehicle's last delivery.",
        ),
        (
            "Cost breakdown",
            charts["cost_breakdown"],
            "Fuel, maintenance, base wages and the volume overflow fee (paid use of "
            "the 10 % volume reserve) per vehicle, plus the vehicle-day overtime "
            "premium and the total cost composition. Bar totals equal each "
            "vehicle's total cost.",
        ),
        (
            "Capacity and trip size",
            charts["trip_capacity"],
            "Weight and usable-volume utilisation per trip, and the number of "
            "customers per trip against the soft minimum of 10 and maximum of 50.",
        ),
        (
            "Route map and trip cost",
            charts["route_map"],
            (route_caption or "Routes are drawn depot to customers and back.")
            + " The scatter plot uses the standalone trip cost; the solution total "
            "adds vehicle-day overtime once. Open route_map.html for the "
            "interactive street map.",
        ),
    ]
    chart_html = "".join(
        f"<section><h2>{html.escape(title)}</h2>{_img_tag(path)}"
        f"<p class='caption'>{html.escape(caption)}</p></section>"
        for title, path, caption in chart_sections
    )

    file_links = [
        "report.html",
        "summary.csv",
        "vehicles.csv",
        "trips.csv",
        "history.csv",
        "convergence.png",
        "vehicle_timeline.png",
        "cost_breakdown.png",
        "trip_capacity.png",
        "route_map.png",
    ]
    links_html = "".join(
        f"<a href='{html.escape(name)}'>{html.escape(name)}</a>"
        for name in file_links
    )

    errors_html = ""
    if check_errors:
        errors_html = (
            "<section class='warning'><h2>Validation errors</h2><ul>"
            + "".join(f"<li>{html.escape(str(e))}</li>" for e in check_errors)
            + "</ul></section>"
        )

    dataset = html.escape(str(payload.get("dataset", "dataset")))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Furniture Delivery - PSO Report</title>
<style>
:root {{
  --bg: #f5f7fa;
  --card: #ffffff;
  --ink: #1f2937;
  --muted: #6b7280;
  --line: #e5e7eb;
  --accent: #2E86AB;
  --good: #16a34a;
  --bad: #dc2626;
  --warn: #d97706;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  line-height: 1.5;
}}
.wrap {{ max-width: 1400px; margin: 0 auto; padding: 28px 22px 48px; }}
header {{ padding: 8px 0 4px; }}
h1 {{ margin: 0 0 4px; font-size: 30px; letter-spacing: -0.02em; }}
.subtitle {{ color: var(--muted); margin: 0; }}
.badge {{
  display: inline-block;
  margin-top: 10px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.02em;
}}
.badge.good {{ background: #dcfce7; color: #166534; }}
.badge.bad {{ background: #fee2e2; color: #991b1b; }}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin: 22px 0 26px;
}}
.card {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}}
.card .label {{
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
.card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
.card.good .value {{ color: var(--good); }}
.card.bad .value {{ color: var(--bad); }}
.card.warn .value {{ color: var(--warn); }}
section {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 20px 22px;
  margin: 18px 0;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}}
section.warning {{ border-color: #fecaca; background: #fff7f7; }}
h2 {{ margin: 0 0 12px; font-size: 20px; }}
h3 {{ margin: 18px 0 8px; font-size: 15px; color: #374151; }}
img {{ max-width: 100%; height: auto; display: block; border-radius: 8px; }}
.caption {{ color: var(--muted); font-size: 13px; margin: 8px 0 0; }}
.table-wrap {{
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
}}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }}
th {{
  text-align: left;
  background: #f8fafc;
  color: #475569;
  font-weight: 600;
  padding: 8px 10px;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}}
td {{ padding: 7px 10px; border-bottom: 1px solid #f1f5f9; white-space: nowrap; }}
tr:last-child td {{ border-bottom: none; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tbody tr:hover {{ background: #f8fafc; }}
.links a {{ color: var(--accent); text-decoration: none; margin-right: 14px; }}
.links a:hover {{ text-decoration: underline; }}
.note {{ color: var(--muted); font-size: 13px; }}
.missing {{ color: var(--bad); }}
@media (max-width: 700px) {{
  .wrap {{ padding: 18px 12px; }}
  h1 {{ font-size: 24px; }}
  section {{ padding: 16px; }}
}}
@media print {{
  body {{ background: #fff; }}
  section {{ box-shadow: none; break-inside: avoid; }}
}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Furniture Delivery - PSO Report</h1>
  <p class="subtitle">Dataset: {dataset} &middot; Generated: {html.escape(generated)}</p>
  <span class="badge {status_class}">{html.escape(status_text)}</span>
</header>

<div class="cards">{card_html}</div>

{chart_html}

<section>
  <h2>Summary</h2>
  {summary_table}
</section>

<section>
  <h2>Fleet summary</h2>
  {vehicle_table}
  <p class="caption">Vehicle-day overtime is charged once per vehicle, on the
  sum of its trip times. The total column includes that overtime premium.</p>
</section>

<section>
  <h2>Trip details</h2>
  {trip_table}
  <p class="caption">Trip cost is the standalone cost of that trip. Weight
  utilisation uses the vehicle weight limit; volume utilisation uses the
  full cargo volume (above 90 % a fee of 100 EUR per m3 is charged and is
  included in the trip cost; 100 % is the hard limit).</p>
</section>

{errors_html}

<section>
  <h2>Files</h2>
  <p class="links">{links_html}</p>
  <p class="note">The HTML report embeds the charts, so it can be opened
  without the PNG files. The CSV files are provided for further analysis in
  Excel or Python.</p>
</section>

<section>
  <h2>Reading notes</h2>
  <ul class="note">
    <li>The convergence chart is the PSO loss curve: lower is better.</li>
    <li>The timeline assumes trips are performed sequentially in solution
    order; it is a workload view, not a clock schedule.</li>
    <li>The route map uses the latitude/longitude in <code>locations.csv</code>;
    in road mode the lines follow the OpenStreetMap road graph when
    <code>route_geometry.json</code> is present.</li>
    <li>The solution is a heuristic result, not a certified global optimum.</li>
  </ul>
</section>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_report(
    data: ProblemData,
    payload: dict,
    output_dir: str | Path = "output/report",
    geometry: dict | None = None,
) -> Path:
    """Generate HTML, PNG and CSV report files.  Returns the HTML path.

    ``geometry`` is the output of experiments/road_geometry.py; with it the
    routes are drawn along the roads, without it as straight lines.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    solution = payload.get("best_solution")
    if not isinstance(solution, dict):
        raise ValueError("payload does not contain a 'best_solution' object")

    trips = _trip_records(data, solution)
    vehicles = _vehicle_records(data, solution, trips)
    summary_rows = _summary_rows(payload, solution)

    _write_csv(
        out_dir / "summary.csv",
        [
            {"section": section, "key": key, "value": value}
            for section, key, value in summary_rows
        ],
        ["section", "key", "value"],
    )
    _write_csv(
        out_dir / "vehicles.csv",
        [
            _round_row(
                {
                    "vehicle_id": v["vehicle_id"],
                    "type": v["type"],
                    "model": v["model"],
                    "trips": v["trips"],
                    "customers": v["customers"],
                    "distance_km": v["distance_km"],
                    "time_h": v["time_h"],
                    "overtime_h": v["overtime_h"],
                    "fuel_cost_eur": v["fuel_cost"],
                    "maintenance_cost_eur": v["maintenance_cost"],
                    "wage_base_cost_eur": v["wage_base_cost"],
                    "overtime_cost_eur": v["overtime_cost"],
                    "total_cost_eur": v["total_cost"],
                }
            )
            for v in vehicles
        ],
        [
            "vehicle_id",
            "type",
            "model",
            "trips",
            "customers",
            "distance_km",
            "time_h",
            "overtime_h",
            "fuel_cost_eur",
            "maintenance_cost_eur",
            "wage_base_cost_eur",
            "overtime_cost_eur",
            "total_cost_eur",
        ],
    )
    _write_csv(
        out_dir / "history.csv",
        [
            {
                "iteration": int(_num(row.get("iteration"), i + 1)),
                "best_score": _csv_number(row.get("best_score")),
                "best_true_cost": _csv_number(row.get("best_true_cost")),
                "best_penalty": _csv_number(row.get("best_penalty")),
                "best_feasible_cost": _csv_number(
                    row.get("best_feasible_cost")
                ),
                "evaluations": int(_num(row.get("evaluations"))),
            }
            for i, row in enumerate(payload.get("history", []))
        ],
        [
            "iteration",
            "best_score",
            "best_true_cost",
            "best_penalty",
            "best_feasible_cost",
            "evaluations",
        ],
    )
    _write_csv(
        out_dir / "trips.csv",
        [
            _round_row(
                {
                    "trip": row["trip"],
                    "vehicle_id": row["vehicle_id"],
                    "vehicle_type": row["vehicle_type"],
                    "customers": row["customers"],
                    "distance_km": row["distance_km"],
                    "time_h": row["time_h"],
                    "weight_kg": row["weight_kg"],
                    "volume_m3": row["volume_m3"],
                    "weight_util_pct": row["weight_util_pct"],
                    "volume_util_pct": row["volume_util_pct"],
                    "fuel_cost_eur": row["fuel_cost"],
                    "maintenance_cost_eur": row["maintenance_cost"],
                    "wage_base_cost_eur": row["wage_base_cost"],
                    "overtime_cost_eur": row["overtime_cost"],
                    "base_cost_eur": row["base_cost_eur"],
                    "standalone_cost_eur": row["cost_eur"],
                    "penalty_eur": row["penalty"],
                    "feasible": row["feasible"],
                }
            )
            for row in trips
        ],
        [
            "trip",
            "vehicle_id",
            "vehicle_type",
            "customers",
            "distance_km",
            "time_h",
            "weight_kg",
            "volume_m3",
            "weight_util_pct",
            "volume_util_pct",
            "fuel_cost_eur",
            "maintenance_cost_eur",
            "wage_base_cost_eur",
            "overtime_cost_eur",
            "base_cost_eur",
            "standalone_cost_eur",
            "penalty_eur",
            "feasible",
        ],
    )

    charts = {
        "convergence": out_dir / "convergence.png",
        "vehicle_timeline": out_dir / "vehicle_timeline.png",
        "cost_breakdown": out_dir / "cost_breakdown.png",
        "trip_capacity": out_dir / "trip_capacity.png",
        "route_map": out_dir / "route_map.png",
    }
    _plot_convergence(
        payload.get("history", []),
        charts["convergence"],
        final_cost=payload.get("best_feasible_cost"),
    )
    _plot_vehicle_timeline(data, solution, charts["vehicle_timeline"])
    _plot_cost_breakdown(vehicles, charts["cost_breakdown"])
    _plot_trip_capacity(trips, charts["trip_capacity"])
    _plot_route_map(data, solution, trips, charts["route_map"], geometry)
    caption = route_mode_caption(data, geometry)
    write_route_map_html(data, solution, geometry, out_dir / "route_map.html", caption)

    report_path = out_dir / "report.html"
    report_path.write_text(
        _build_html(
            payload, solution, trips, vehicles, summary_rows, charts, out_dir,
            route_caption=caption,
        ),
        encoding="utf-8",
    )
    return report_path


def write_route_map_html(
    data: ProblemData,
    solution: dict,
    geometry: dict | None,
    path: Path,
    caption: str,
) -> Path:
    """Interactive street map: one layer per vehicle, popups per stop.

    Leaflet and the Esri street/satellite tiles load from the internet when the page
    is opened; the route data itself is embedded in the file.
    """
    legs_by_trip = _geometry_by_trip(solution, geometry)
    routes = []
    for t_index, trip in enumerate(solution.get("trips", [])):
        k = int(trip["vehicle_index"])
        customers = [int(c) for c in trip.get("customers", [])]
        legs = legs_by_trip[t_index]
        if legs:
            lines = [{"points": leg["points"], "dashed": leg["status"] == "unroutable"}
                     for leg in legs]
        else:
            stops = [0] + customers + [0]
            lines = [{"points": [[float(data.lat[c]), float(data.lon[c])] for c in stops],
                      "dashed": False}]
        routes.append({
            "vehicle": k, "trip": t_index + 1, "customers": customers,
            "km": _num(trip.get("distance_km")), "hours": _num(trip.get("time_min")) / 60.0,
            "cost": _num(trip.get("cost_eur")), "lines": lines,
        })
    payload = {
        "depot": [float(data.lat[0]), float(data.lon[0])],
        "stops": [[float(data.lat[i]), float(data.lon[i])] for i in range(len(data.lat))],
        "vehicles": [f"V{v.vehicle_id} {v.type_id}" for v in data.vehicles],
        "colors": [VEHICLE_COLORS[k % len(VEHICLE_COLORS)] for k in range(data.n_vehicles)],
        "routes": routes,
    }
    data_json = json.dumps(payload).replace("</", "<\\/")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Delivery routes - street map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{margin:0;font:14px system-ui,sans-serif;color:#1d2b36}
header{padding:10px 16px;background:#f4f7fa;border-bottom:1px solid #d6dee6}
h1{font-size:18px;margin:0 0 4px}p{margin:4px 0;max-width:1100px}
#map{height:calc(100vh - 140px);min-height:420px}label{margin-right:14px;font-weight:600}
</style></head><body><header><h1>Delivery routes on the Budapest street map</h1>
<p>__CAPTION__</p><p id="toggles"></p></header><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>const d=__DATA__;
const map=L.map('map');
const esri=s=>'https://server.arcgisonline.com/ArcGIS/rest/services/'+s+'/MapServer/tile/{z}/{y}/{x}';
const esriAttr='Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community';
// Esri street tiles: tile.openstreetmap.org blocks pages opened as local
// files (no Referer, see osm.wiki/Blocked), which is how this report is opened.
const street=L.tileLayer(esri('World_Street_Map'),{maxZoom:19,
attribution:'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, USGS, &copy; OpenStreetMap contributors, and the GIS User Community'}).addTo(map);
const sat=()=>L.tileLayer(esri('World_Imagery'),{maxZoom:19,attribution:esriAttr});
const hybrid=L.layerGroup([sat(),L.tileLayer(esri('Reference/World_Transportation'),{maxZoom:19,opacity:.7}),
L.tileLayer(esri('Reference/World_Boundaries_and_Places'),{maxZoom:19})]);
L.control.layers({'Street map':street,'Satellite':sat(),'Satellite + labels':hybrid},null,{collapsed:false}).addTo(map);
const groups=d.vehicles.map(()=>L.layerGroup().addTo(map));const bounds=[d.depot];
d.routes.forEach(r=>{const c=d.colors[r.vehicle];r.lines.forEach(l=>{bounds.push(...l.points);
L.polyline(l.points,{color:'#fff',weight:7,opacity:.6,interactive:false}).addTo(groups[r.vehicle]);
L.polyline(l.points,{color:c,weight:4,opacity:.85,dashArray:l.dashed?'6 6':null})
.bindPopup(d.vehicles[r.vehicle]+' - trip '+r.trip+'<br>'+r.customers.length+' stops, '+r.km.toFixed(1)+' km, '+r.hours.toFixed(2)+' h, EUR '+r.cost.toFixed(2))
.addTo(groups[r.vehicle]);});
r.customers.forEach((cu,i)=>L.circleMarker(d.stops[cu],{radius:4,color:c,fillOpacity:.9})
.bindPopup('Customer '+cu+'<br>'+d.vehicles[r.vehicle]+', trip '+r.trip+', stop '+(i+1)).addTo(groups[r.vehicle]));});
L.marker(d.depot).bindPopup('Depot').addTo(map);map.fitBounds(bounds,{padding:[20,20]});
const t=document.getElementById('toggles');d.vehicles.forEach((v,k)=>{const lab=document.createElement('label');
lab.style.color=d.colors[k];const cb=document.createElement('input');cb.type='checkbox';cb.checked=true;
cb.onchange=()=>cb.checked?groups[k].addTo(map):map.removeLayer(groups[k]);lab.append(cb,' '+v);t.append(lab);});
</script></body></html>"""
    page = page.replace("__CAPTION__", html.escape(caption)).replace("__DATA__", data_json)
    path.write_text(page, encoding="utf-8")
    return path


def load_payload(path: str | Path = "output/best_solution.json") -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
