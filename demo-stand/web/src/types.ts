export type ScreenState = "OK" | "DRIFT" | "BETTER" | "NEUTRAL";
export type Mechanic = "A" | "B" | "C";
export type Mode = "demo" | "review" | "data";

export interface Corridor {
  corridor: string;
  country: string;
  flag: string;
  currency_code: string;
  currency_name: string;
  currency_gen: string;
  currency_dat: string;
  currency_acc: string;
  currency_short: string;
}

export interface Persona {
  id: string;
  name: string;
  corridor: string;
  city: string;
  timezone: string;
  typical_amount_rub: number;
  recipient_name: string;
  recipient_phone: string;
  recipient_bank: string;
  open_delay_min: number;
  rate_sensitivity: string;
  recipient_limit: null | {
    per_operation_kgs: number;
    per_month_kgs: number;
    reason: string;
  };
  note: string;
  assumption: string;
}

export interface Scenario {
  id: string;
  title: string;
  persona: string;
  corridor: string;
  as_of_date: string;
  push_sent_at: string | null;
  open_delay_min: number;
  entry: "PUSH" | "SELF";
  push_rate: number | null;
  expected_state: ScreenState;
  scenario_code: string;
  summary: string;
  optional?: boolean;
  amount_rub_override?: number;
  panel_hint?: Record<string, number>;
}

export interface Plaque {
  scenario_code: string;
  text: string | null;
  context: string | null;
  forbidden: string | null;
  why_forbidden: string | null;
}

export interface Evaluation {
  state: ScreenState;
  entry: "PUSH" | "SELF";
  push_text: string | null;
  delta_bp: number;
  current_rate: number;
  current_rate_is_stale: boolean;
  push_rate: number | null;
  recipient_gets: number | null;
  recipient_gets_at_push: number | null;
  recipient_delta: number | null;
  recipient_currency: string;
  recipient_currency_short: string;
  percentile_now: number;
  prefill: boolean;
  plaque: Plaque;
  actions: { primary: "TRANSFER"; secondary: "RESERVE" | "RETURN" | null };
  signal: null | {
    date: string;
    corridor: string;
    indicator: string;
    direction: string;
    speed: string;
    strength: number;
    scenario_code: string;
    facts: Record<string, number | string>;
  };
  disclaimer: string;
  scenario?: Scenario;
  persona?: Persona;
  expected_state?: ScreenState;
  state_matches_expected?: boolean;
}

export interface ReserveView {
  id: string;
  corridor: string;
  amount_rub: number;
  created_on: string;
  percentile: number;
  window_days: number;
  ttl_days: number;
  fallback_send_on_expiry: boolean;
  state: "ACTIVE" | "EXECUTED" | "CANCELLED" | "EXPIRED" | "SUPERSEDED";
  rate_at_creation: number;
  executed_on: string | null;
  exec_rate: number | null;
  waited_days: number | null;
  gain_bp: number | null;
  as_of: string;
  current_rate: number;
  threshold_rate: number;
  distance_bp: number;
  days_left: number;
  condition_text: string;
  recipient_gets_now: number;
  recipient_gets_at_creation: number;
}

export interface StandEvent {
  ts: string;
  type: string;
  sim_date: string | null;
  session_id: string | null;
  payload: Record<string, unknown>;
  known_type: boolean;
}

export interface MlServiceProbe {
  configured: boolean;
  reachable?: boolean;
  error?: string;
  health?: Record<string, unknown>;
}

export interface Health {
  status: string;
  version: string;
  signals_source: { active: string; model_version: string | null; fell_back?: boolean; error?: string };
  rates_source?: { active: string; error?: string | null };
  ml_services?: {
    parser: MlServiceProbe;
    moment_model: MlServiceProbe;
    push_model: MlServiceProbe;
  };
  dates_available: { from: string; to: string };
  features: { reserve: boolean; recipient_limit: boolean };
  params: Record<string, number>;
}

export interface RatePoint {
  date: string;
  rate: number;
  is_stale: boolean;
}
