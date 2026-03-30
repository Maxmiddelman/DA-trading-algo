# Battery MILP Solver API

Snelle setup:

## 1. Maak een virtual environment
```bash
python -m venv .venv
```

## 2. Activeer hem
Mac/Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

## 3. Installeer dependencies
```bash
pip install -r requirements.txt
```

## 4. Start de API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 5. Open de docs
Ga naar:
```text
http://127.0.0.1:8000/docs
```

## 6. Test day-ahead
Gebruik `test_payload_dayahead.json` in `/docs` of via curl:
```bash
curl -X POST http://127.0.0.1:8000/solve/dayahead \
  -H "Content-Type: application/json" \
  -d @test_payload_dayahead.json
```

## Belangrijke logica
- `locked_schedule` is 0 bij day-ahead.
- Voor ID2 vul je hier de output van DA in.
- Voor ID3 vul je hier de cumulatieve output van ID2 in.
- `solver_backend` probeert eerst SCIP en valt terug op CBC.

## Belangrijk voor jouw batterij
- 200 kW bij 15 minuten = 50 kWh max per stap.
- Zet in het begin bijna altijd `final_soc_target_kwh` aan.
- Als je infeasible krijgt, check dan eerst:
  - horizon lengte
  - arrays even lang
  - SOC target
  - locked schedule
  - reserve SOC
