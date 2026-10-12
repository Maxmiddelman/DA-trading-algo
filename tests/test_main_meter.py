from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import SolveRequest, solve_battery_dispatch


def request(**changes):
    data = dict(mode='dayahead',time_step_minutes=15,horizon_steps=2,
        optimization_timestamp='2026-06-01T10:00:00Z',actual_soc_kwh=9,
        battery=dict(power_kw=4,capacity_kwh=10),site=dict(max_import_kw=10,max_export_kw=20),
        series=dict(timestamps=['2026-06-01T10:00:00Z','2026-06-01T10:15:00Z'],
            load_kwh=[.25,.25],pv_kwh=[3,3],da_price_eur_per_mwh=[100,100],
            id2_price_eur_per_mwh=[100,100],id3_price_eur_per_mwh=[100,100]),
        options=dict(min_soc_pct=10,max_soc_pct=90,include_kw_max=False,round_trip_efficiency=.9))
    data.update(changes)
    return SolveRequest(**data)


def test_zero_program_curtails_without_physical_export_limit():
    req=request(main_meter_program_kw=[0,0]); result=solve_battery_dispatch(req)
    assert result['status']=='optimal'
    previous=req.actual_soc_kwh
    for i,s in enumerate(result['schedule']):
        assert s['export_kwh']==pytest.approx(0,abs=1e-5)
        assert s['import_kwh']==pytest.approx(0,abs=1e-5)
        assert s['pv_curtail_kwh']>2
        assert s['import_kwh']+s['discharge_kwh']+3-s['pv_curtail_kwh']==pytest.approx(.25+s['charge_kwh']+s['export_kwh'],abs=1e-5)
        assert s['soc_end_kwh']==pytest.approx(previous+s['charge_kwh']*.9**.5-s['discharge_kwh']/.9**.5,abs=1e-5)
        previous=s['soc_end_kwh']


def test_free_grid_exports_instead_of_throwing_away_valuable_pv():
    r=solve_battery_dispatch(request())
    assert r['status']=='optimal'
    assert sum(s['export_kwh'] for s in r['schedule'])>0
    assert sum(s['pv_curtail_kwh'] for s in r['schedule'])==pytest.approx(0,abs=1e-5)


def test_impossible_program_returns_infeasible():
    assert solve_battery_dispatch(request(main_meter_program_kw=[-100,-100]))['status']=='infeasible'


@pytest.mark.parametrize('change',[ 'missing', 'duplicate', 'nan', 'naive' ])
def test_bad_input_is_rejected(change):
    req=request()
    if change=='missing': req.series.pv_kwh.pop()
    if change=='duplicate': req.series.timestamps[1]=req.series.timestamps[0]
    if change=='nan': req.series.load_kwh[0]=float('nan')
    if change=='naive': req.series.timestamps=[t.rstrip('Z') for t in req.series.timestamps]
    with pytest.raises(Exception): solve_battery_dispatch(req)


def test_hard_terminal_target_keeps_priority_over_terminal_credit():
    req=request();req.options.final_soc_target_kwh=9;req.options.terminal_soc_value_eur_per_kwh=1
    req.options.grid_slew_cost_eur_per_kw=.001;req.options.previous_grid_kw=0
    r=solve_battery_dispatch(req)
    assert r['status']=='optimal';assert r['schedule'][-1]['soc_end_kwh']==pytest.approx(9)
    assert r['objective_adjustments']['terminal_energy_credit_eur']==pytest.approx(9)
