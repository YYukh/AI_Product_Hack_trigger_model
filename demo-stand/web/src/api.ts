import type {
  Corridor,
  Evaluation,
  Health,
  Persona,
  RatePoint,
  ReserveView,
  Scenario,
  StandEvent,
} from "./types";

const BASE = "/api";

async function j<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    throw new Error(`${r.status} ${path} ${detail}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => j<Health>("/health"),
  corridors: () => j<Corridor[]>("/corridors"),
  personas: () => j<Persona[]>("/personas"),
  scenarios: () => j<Scenario[]>("/scenarios"),
  texts: () => j<any[]>("/texts"),
  rates: (corridor: string, from?: string, to?: string) =>
    j<{ corridor: string; points: RatePoint[] }>(
      `/rates?corridor=${corridor}` + (from ? `&from=${from}` : "") + (to ? `&to=${to}` : "")
    ),
  signals: (asOf: string, corridor?: string) =>
    j<{ as_of: string; model_version: string; signals: any[] }>(
      `/signals?as_of=${asOf}` + (corridor ? `&corridor=${corridor}` : "")
    ),
  mlEngineSignals: (asOf: string, corridor?: string) =>
    j<any>(`/ml/engine-signals?as_of=${asOf}` + (corridor ? `&corridor=${corridor}` : "")),
  mlDecisions: (asOf: string, corridor?: string) =>
    j<any>(`/ml/decisions?as_of=${asOf}` + (corridor ? `&corridor=${corridor}` : "")),
  runScenario: (
    id: string,
    opts: { mechanic?: string; driftThresholdBp?: number; amountRub?: number; simDate?: string; openDelayMin?: number } = {}
  ) => {
    const q = new URLSearchParams();
    if (opts.mechanic) q.set("drift_mechanic", opts.mechanic);
    if (opts.driftThresholdBp != null) q.set("drift_threshold_bp", String(opts.driftThresholdBp));
    if (opts.amountRub != null) q.set("amount_rub", String(opts.amountRub));
    if (opts.simDate) q.set("sim_date", opts.simDate);
    if (opts.openDelayMin != null) q.set("open_delay_min", String(opts.openDelayMin));
    return j<Evaluation>(`/scenario/${id}/run?` + q.toString());
  },
  evaluate: (body: Record<string, unknown>) =>
    j<Evaluation>("/evaluate", { method: "POST", body: JSON.stringify(body) }),
  createReserve: (body: Record<string, unknown>) =>
    j<ReserveView>("/reserve", { method: "POST", body: JSON.stringify(body) }),
  viewReserve: (id: string, asOf: string) => j<ReserveView>(`/reserve/${id}?as_of=${asOf}`),
  cancelReserve: (id: string, asOf: string, sessionId: string) =>
    j<ReserveView>(`/reserve/${id}/cancel`, {
      method: "POST",
      body: JSON.stringify({ as_of: asOf, session_id: sessionId }),
    }),
  transferNowReserve: (id: string, asOf: string, sessionId: string) =>
    j<ReserveView>(`/reserve/${id}/transfer-now`, {
      method: "POST",
      body: JSON.stringify({ as_of: asOf, session_id: sessionId }),
    }),
  events: (sessionId: string, limit = 200) =>
    j<StandEvent[]>(`/events?session_id=${sessionId}&limit=${limit}`),
  postEvent: (body: Record<string, unknown>) =>
    j<StandEvent>("/events", { method: "POST", body: JSON.stringify(body) }),
  policyPreview: (body: Record<string, unknown>) =>
    j<any[]>("/policy/preview", { method: "POST", body: JSON.stringify(body) }),
  recipientLimitCheck: (persona: string, amountRub: number, corridor: string, asOf: string) =>
    j<any>(
      `/recipient-limit/check?persona=${persona}&amount_rub=${amountRub}&corridor=${corridor}&as_of=${asOf}`
    ),
};
