// Состояние прогона в query-строке.
//   ?scenario=S2                 — вкладка «Сценарии», проигрывается S2
//   ?corridor=RUB_TJS&date=...&entry=PUSH  — вкладка «Песочница» с параметрами
export type Tab = "sandbox" | "scenarios";
export type Entry = "SELF" | "PUSH";

export interface UrlState {
  tab: Tab;
  scenario: string | null;
  corridor: string | null;
  date: string | null;
  entry: Entry | null;
}

export function readUrl(): UrlState {
  const p = new URLSearchParams(location.search);
  const scenario = p.get("scenario");
  const tabRaw = p.get("tab");
  const tab: Tab =
    tabRaw === "scenarios" || (scenario && tabRaw !== "sandbox") ? "scenarios" : "sandbox";
  const entryRaw = p.get("entry");
  return {
    tab,
    scenario,
    corridor: p.get("corridor"),
    date: p.get("date"),
    entry: entryRaw === "PUSH" || entryRaw === "SELF" ? entryRaw : null,
  };
}

export function writeUrl(s: Partial<UrlState>) {
  const p = new URLSearchParams();
  if (s.tab === "scenarios") {
    p.set("tab", "scenarios");
    if (s.scenario) p.set("scenario", s.scenario);
  } else {
    if (s.corridor) p.set("corridor", s.corridor);
    if (s.date) p.set("date", s.date);
    if (s.entry) p.set("entry", s.entry);
  }
  const qs = p.toString();
  history.replaceState(null, "", qs ? `?${qs}` : location.pathname);
}

// Анонимный идентификатор сессии для группировки событий. Не ПДн.
export function sessionId(): string {
  const k = "stand_session";
  let v = sessionStorage.getItem(k);
  if (!v) {
    v = "s_" + Math.random().toString(36).slice(2, 10);
    sessionStorage.setItem(k, v);
  }
  return v;
}

// --- форматирование ---------------------------------------------------
const nf = new Intl.NumberFormat("ru-RU");

export function money(n: number | null | undefined, suffix = "₽"): string {
  if (n == null) return "—";
  return `${nf.format(Math.round(n))} ${suffix}`;
}

export function units(n: number | null | undefined, currencyShort: string): string {
  if (n == null) return "—";
  return `${nf.format(Math.round(n))} ${currencyShort}`;
}

export function rate(n: number | null | undefined): string {
  if (n == null) return "—";
  // адаптивная точность: у мелких валют (UZS ≈ 0,0069) 5 знаков теряют цифры
  const abs = Math.abs(n);
  const digits = abs >= 100 ? 2 : abs >= 1 ? 4 : abs >= 0.01 ? 5 : 6;
  return n
    .toFixed(digits)
    .replace(/(\.\d*?)0+$/, "$1")
    .replace(/\.$/, "")
    .replace(".", ",");
}

export function pct(bp: number): string {
  return (Math.abs(bp) / 100).toFixed(2).replace(".", ",") + "%";
}

export function ddmm(iso: string): string {
  const [, m, d] = iso.split("-");
  return `${d}.${m}`;
}

export function ddmmyyyy(iso: string): string {
  const [y, m, d] = iso.split("-");
  return `${d}.${m}.${y}`;
}

export function minutesToHHMM(min: number): string {
  const h = Math.floor(min / 60) % 24;
  const m = min % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}
