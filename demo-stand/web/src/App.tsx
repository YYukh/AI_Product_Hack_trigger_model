import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { readUrl, writeUrl, sessionId } from "./store";
import type { Entry, Tab } from "./store";
import type {
  Corridor,
  Evaluation,
  Health,
  Persona,
  ReserveView,
  Scenario,
} from "./types";
import { PhoneFrame } from "./components/Shell";
import { TopTabs, SandboxBar, ScenarioBar } from "./components/Chrome";
import { BackButton } from "./components/ui";
import { PushToast } from "./components/PushToast";
import { Launcher } from "./Launcher";
import { Home, CountryList } from "./screens/Home";
import { TransferScreen, TransferConfirm, TransferSuccess } from "./screens/Transfer";
import { NotificationSettings } from "./screens/Settings";
import {
  ReserveSetup,
  ReserveConfirm,
  ReserveManage,
  ReserveExecuted,
  ReserveExpired,
} from "./screens/Reserve";
import { RecipientLimitWarning } from "./screens/RecipientLimit";

// Механика подачи DRIFT и порог — фиксированы (пульт стенда убран).
const MECHANIC = "C";
const DRIFT_THR = 20;
const DEFAULT_CORRIDOR = "RUB_TJS";

type Screen =
  | "push"
  | "home"
  | "country"
  | "transfer"
  | "confirm"
  | "success"
  | "settings"
  | "reserve-setup"
  | "reserve-confirm"
  | "reserve-manage"
  | "reserve-executed"
  | "reserve-expired"
  | "recipient-limit";

const hhmm = (s: string) => {
  const [h, m] = s.split(":").map(Number);
  return h * 60 + m;
};

function nextTradingDay(iso: string): string {
  const d = new Date(iso + "T00:00:00Z");
  do {
    d.setUTCDate(d.getUTCDate() + 1);
  } while (d.getUTCDay() === 0 || d.getUTCDay() === 6);
  return d.toISOString().slice(0, 10);
}

function prevTradingDay(iso: string): string {
  const d = new Date(iso + "T00:00:00Z");
  do {
    d.setUTCDate(d.getUTCDate() - 1);
  } while (d.getUTCDay() === 0 || d.getUTCDay() === 6);
  return d.toISOString().slice(0, 10);
}

function shiftIso(iso: string, days: number): string {
  const d = new Date(iso + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

const clampDate = (iso: string, lo: string, hi: string) =>
  iso < lo ? lo : iso > hi ? hi : iso;

export default function App() {
  const url0 = useMemo(() => readUrl(), []);
  const sid = useMemo(() => sessionId(), []);

  const [tab, setTab] = useState<Tab>(url0.tab);
  const [corridors, setCorridors] = useState<Corridor[]>([]);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [health, setHealth] = useState<Health | null>(null);

  // --- вкладка «Сценарии»
  const [scenarioId, setScenarioId] = useState<string>(url0.scenario || "");

  // --- вкладка «Песочница»
  const [sbCorridor, setSbCorridor] = useState<string>(url0.corridor || "");
  const [sbEntry, setSbEntry] = useState<Entry>(url0.entry || "SELF");
  const [sbPushDaysAgo, setSbPushDaysAgo] = useState<number>(4);
  const [sbPushRate, setSbPushRate] = useState<number | null>(null);
  const [sbPushDate, setSbPushDate] = useState<string | null>(null);

  // --- общий поток
  const [simDate, setSimDate] = useState<string>(url0.date || "");
  const [screen, setScreen] = useState<Screen>("home");
  const [ev, setEv] = useState<Evaluation | null>(null);
  const [pushEval, setPushEval] = useState<Evaluation | null>(null);
  const [amountRub, setAmountRub] = useState<number>(0);
  const [reserve, setReserve] = useState<ReserveView | null>(null);
  const [limitCheck, setLimitCheck] = useState<any | null>(null);

  const dateMin = health?.dates_available.from || "2020-01-02";
  const dateMax = health?.dates_available.to || "2026-09-06";

  const inScenario = tab === "scenarios" && !!scenarioId;

  const activeScenario = useMemo(
    () => scenarios.find((s) => s.id === scenarioId) || null,
    [scenarios, scenarioId]
  );

  const corridorCode = inScenario
    ? activeScenario?.corridor || sbCorridor
    : sbCorridor;

  const corridor = useMemo(
    () =>
      corridors.find((c) => c.corridor === corridorCode) || corridors[0] || null,
    [corridors, corridorCode]
  );

  const persona = useMemo<Persona | null>(() => {
    if (inScenario)
      return personas.find((p) => p.id === activeScenario?.persona) || personas[0] || null;
    const p = personas.find((x) => x.corridor === corridorCode);
    if (p) return p;
    if (!corridor) return null;
    return {
      id: "sandbox",
      name: "Гость",
      corridor: corridorCode,
      city: "",
      timezone: "",
      typical_amount_rub: 20000,
      recipient_name: "",
      recipient_phone: "",
      recipient_bank: corridor.country,
      open_delay_min: 0,
      rate_sensitivity: "mid",
      recipient_limit: null,
      note: "",
      assumption: "",
    } as Persona;
  }, [inScenario, personas, activeScenario, corridorCode, corridor]);

  // --- начальная загрузка --------------------------------------------
  useEffect(() => {
    Promise.all([api.corridors(), api.personas(), api.scenarios(), api.health()])
      .then(([c, p, s, h]) => {
        setCorridors(c);
        setPersonas(p);
        setScenarios(s);
        setHealth(h);
      })
      .catch((e) => console.error("bootstrap failed", e));
  }, []);

  // дефолты песочницы, когда пришли данные
  useEffect(() => {
    if (!corridors.length || !health) return;
    setSbCorridor((prev) =>
      prev && corridors.some((c) => c.corridor === prev) ? prev : DEFAULT_CORRIDOR
    );
    setSimDate((prev) => prev || health.dates_available.to);
  }, [corridors, health]);

  // --- запрос к /evaluate -------------------------------------------
  const evalBody = useCallback(
    (
      entry: Entry,
      date: string,
      opts?: { corridor?: string; pushRate?: number | null }
    ) => {
      const corr = opts?.corridor ?? corridorCode;
      if (inScenario && activeScenario) {
        const sc = activeScenario;
        const pushMin = sc.push_sent_at ? hhmm(sc.push_sent_at) : null;
        const simMinutes =
          entry === "PUSH" && pushMin != null
            ? pushMin + (sc.open_delay_min || 0)
            : 12 * 60;
        return {
          corridor: sc.corridor,
          sim_date: date,
          sim_minutes: simMinutes,
          entry,
          push_sent_at_minutes: entry === "PUSH" ? pushMin : null,
          push_rate: entry === "PUSH" ? sc.push_rate : null,
          amount_rub:
            amountRub ||
            sc.amount_rub_override ||
            persona?.typical_amount_rub ||
            20000,
          drift_mechanic: MECHANIC,
          drift_threshold_bp: DRIFT_THR,
          session_id: sid,
        };
      }
      // песочница
      const pr = opts?.pushRate !== undefined ? opts.pushRate : sbPushRate;
      return {
        corridor: corr,
        sim_date: date,
        sim_minutes: 12 * 60,
        entry,
        push_sent_at_minutes: entry === "PUSH" ? 9 * 60 : null,
        push_rate: entry === "PUSH" ? pr : null,
        amount_rub: amountRub || persona?.typical_amount_rub || 20000,
        drift_mechanic: MECHANIC,
        drift_threshold_bp: DRIFT_THR,
        session_id: sid,
      };
    },
    [inScenario, activeScenario, corridorCode, sbPushRate, amountRub, persona, sid]
  );

  // --- песочница: курс «в пуше» = курс за N торговых дней до даты ----
  useEffect(() => {
    if (inScenario || sbEntry !== "PUSH" || !corridorCode || !simDate) return;
    let dead = false;
    api
      .rates(corridorCode, shiftIso(simDate, -45), simDate)
      .then((res) => {
        if (dead) return;
        const pts = res.points.filter((p) => p.date <= simDate);
        const pt = pts[Math.max(0, pts.length - 1 - sbPushDaysAgo)];
        setSbPushRate(pt ? pt.rate : null);
        setSbPushDate(pt ? pt.date : null);
      })
      .catch(() => {
        if (!dead) {
          setSbPushRate(null);
          setSbPushDate(null);
        }
      });
    return () => {
      dead = true;
    };
  }, [inScenario, sbEntry, corridorCode, simDate, sbPushDaysAgo]);

  // --- песочница: держим стартовый экран в синхроне с параметрами ---
  useEffect(() => {
    if (inScenario) return;
    if (screen !== "home" && screen !== "push") return;
    if (sbEntry === "SELF") {
      setPushEval(null);
      setEv(null);
      setScreen("home");
      return;
    }
    if (sbPushRate == null) return;
    api
      .evaluate(evalBody("PUSH", simDate, { pushRate: sbPushRate }))
      .then((e) => {
        setPushEval(e);
        setScreen("push");
      })
      .catch((e) => console.error(e));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inScenario, sbEntry, corridorCode, simDate, sbPushRate]);

  // --- проиграть сценарий с начала --------------------------------
  const play = useCallback(async () => {
    const sc = scenarios.find((s) => s.id === scenarioId);
    if (!sc) return;
    setReserve(null);
    setAmountRub(0);
    setLimitCheck(null);
    setSimDate(sc.as_of_date);

    if (sc.entry === "PUSH" && sc.push_sent_at) {
      const body = {
        corridor: sc.corridor,
        sim_date: sc.as_of_date,
        sim_minutes: hhmm(sc.push_sent_at),
        entry: "PUSH",
        push_sent_at_minutes: hhmm(sc.push_sent_at),
        push_rate: sc.push_rate,
        amount_rub:
          sc.amount_rub_override ||
          personas.find((p) => p.id === sc.persona)?.typical_amount_rub ||
          20000,
        drift_mechanic: MECHANIC,
        drift_threshold_bp: DRIFT_THR,
        session_id: sid,
      };
      try {
        setPushEval(await api.evaluate(body));
      } catch (e) {
        console.error(e);
      }
      setEv(null);
      setScreen("push");
    } else {
      const body = {
        corridor: sc.corridor,
        sim_date: sc.as_of_date,
        sim_minutes: 12 * 60,
        entry: "SELF",
        push_sent_at_minutes: null,
        push_rate: null,
        amount_rub:
          personas.find((p) => p.id === sc.persona)?.typical_amount_rub || 20000,
        drift_mechanic: MECHANIC,
        drift_threshold_bp: DRIFT_THR,
        session_id: sid,
      };
      try {
        setEv(await api.evaluate(body));
      } catch (e) {
        console.error(e);
      }
      setPushEval(null);
      setScreen("home");
    }
  }, [scenarios, scenarioId, personas, sid]);

  // прогон при готовности данных и при каждой смене сценария
  useEffect(() => {
    if (tab !== "scenarios" || !scenarioId) return;
    if (scenarios.length && personas.length && corridors.length) play();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenarios.length, personas.length, corridors.length, scenarioId, tab]);

  // синхронизация URL
  useEffect(() => {
    if (tab === "scenarios") writeUrl({ tab: "scenarios", scenario: scenarioId || null });
    else
      writeUrl({
        tab: "sandbox",
        corridor: sbCorridor || null,
        date: simDate || null,
        entry: sbEntry,
      });
  }, [tab, scenarioId, sbCorridor, simDate, sbEntry]);

  // --- переходы ----------------------------------------------------
  function resetFlow() {
    setEv(null);
    setPushEval(null);
    setReserve(null);
    setAmountRub(0);
    setLimitCheck(null);
    setScreen("home");
  }

  function switchTab(t: Tab) {
    if (t === tab) return;
    setTab(t);
    resetFlow();
    if (t === "sandbox") setSimDate(dateMax);
    else setScenarioId("");
  }

  function pickScenario(id: string) {
    resetFlow();
    setScenarioId(id);
  }

  function sandboxReset() {
    setAmountRub(0);
    setReserve(null);
    setLimitCheck(null);
    setEv(null);
    setScreen("home");
  }

  async function openPush() {
    try {
      setEv(
        await api.evaluate(
          inScenario
            ? evalBody("PUSH", simDate)
            : evalBody("PUSH", simDate, { pushRate: sbPushRate })
        )
      );
      setScreen("transfer");
    } catch (e) {
      console.error(e);
    }
  }

  async function goTransferFromHome() {
    if (!corridor) return;
    try {
      setEv(await api.evaluate(evalBody("SELF", simDate)));
    } catch (e) {
      console.error(e);
    }
    setScreen("country");
  }

  async function onTransfer(amount: number) {
    setAmountRub(amount);
    if (persona?.recipient_limit && health?.features.recipient_limit && corridor) {
      try {
        const chk = await api.recipientLimitCheck(
          persona.id === "sandbox" ? "ainura" : persona.id,
          amount,
          corridor.corridor,
          simDate
        );
        if (chk.applies && (chk.exceeds_operation || chk.exceeds_month)) {
          setLimitCheck(chk);
          setScreen("recipient-limit");
          return;
        }
      } catch (e) {
        console.error(e);
      }
    }
    setScreen("confirm");
  }

  async function onConfirm() {
    await api
      .postEvent({
        type: "transfer_confirmed",
        payload: { amount_rub: amountRub, state: ev?.state },
        sim_date: simDate,
        session_id: sid,
      })
      .catch(() => {});
    setScreen("success");
  }

  async function onSecondary() {
    if (!ev) return;
    if (ev.actions.secondary === "RESERVE") setScreen("reserve-setup");
    else setScreen("home");
  }

  async function createReserve(opts: {
    amountRub: number;
    percentile: number;
    windowDays: number;
    fallback: boolean;
  }) {
    if (!corridor) return;
    try {
      const rv = await api.createReserve({
        corridor: corridor.corridor,
        amount_rub: opts.amountRub,
        created_on: simDate,
        percentile: opts.percentile,
        window_days: opts.windowDays,
        ttl_days: 7,
        fallback_send_on_expiry: opts.fallback,
        session_id: sid,
      });
      setReserve(rv);
      setScreen("reserve-confirm");
    } catch (e) {
      console.error(e);
    }
  }

  async function shiftSandboxDay(delta: number) {
    setSimDate((d) => {
      const nd = clampDate(
        delta > 0 ? nextTradingDay(d) : prevTradingDay(d),
        dateMin,
        dateMax
      );
      return nd;
    });
  }

  async function plusDay() {
    if (!simDate) return;
    const nd = clampDate(nextTradingDay(simDate), dateMin, dateMax);
    setSimDate(nd);
    if (reserve && reserve.state === "ACTIVE") {
      try {
        const rv = await api.viewReserve(reserve.id, nd);
        setReserve(rv);
        if (rv.state === "EXECUTED") setScreen("reserve-executed");
        else if (rv.state === "EXPIRED") setScreen("reserve-expired");
      } catch (e) {
        console.error(e);
      }
    }
    if (screen === "transfer") {
      try {
        setEv(await api.evaluate(evalBody(ev?.entry || "SELF", nd)));
      } catch (e) {
        console.error(e);
      }
    }
  }

  async function cancelReserve() {
    if (!reserve) return;
    const rv = await api.cancelReserve(reserve.id, simDate, sid).catch(() => null);
    if (rv) setReserve(rv);
    setScreen("home");
  }

  async function reserveTransferNow() {
    if (!reserve) return;
    await api.transferNowReserve(reserve.id, simDate, sid).catch(() => {});
    try {
      setEv(await api.evaluate(evalBody("SELF", simDate)));
    } catch (e) {
      console.error(e);
    }
    setScreen("transfer");
  }

  // --- рендер ----------------------------------------------------
  const showList = tab === "scenarios" && !scenarioId;
  const loading = !showList && (!corridor || !persona || !simDate || (inScenario && !activeScenario));

  return (
    <div className="flex h-[100dvh] flex-col items-center bg-[#15161A] px-0 py-0 sm:h-auto sm:min-h-screen sm:px-4 sm:py-5">
      <div className="w-full max-w-[393px] shrink-0 space-y-2 px-3 pt-3 sm:px-0 sm:pt-0">
        <TopTabs tab={tab} onTab={switchTab} />

        {tab === "sandbox" && corridor && (
          <SandboxBar
            corridors={corridors}
            corridor={corridorCode}
            onCorridor={(c) => {
              setSbCorridor(c);
              sandboxReset();
            }}
            date={simDate || dateMax}
            dateMin={dateMin}
            dateMax={dateMax}
            onShiftDay={shiftSandboxDay}
            onToday={() => setSimDate(dateMax)}
            entry={sbEntry}
            onEntry={(e) => {
              setSbEntry(e);
              setScreen("home");
            }}
            pushDaysAgo={sbPushDaysAgo}
            onPushDaysAgo={setSbPushDaysAgo}
            pushRate={sbPushRate}
            pushDate={sbPushDate}
            onReset={sandboxReset}
            onPlusDay={plusDay}
          />
        )}

        {tab === "scenarios" && inScenario && activeScenario && (
          <ScenarioBar
            title={`${activeScenario.id} · ${activeScenario.title}`}
            onList={() => setScenarioId("")}
            onReplay={play}
            onPlusDay={reserve?.state === "ACTIVE" ? plusDay : undefined}
          />
        )}
      </div>

      {showList ? (
        <div className="w-full flex-1 overflow-y-auto sm:mt-3 sm:flex-none">
          {scenarios.length ? (
            <Launcher scenarios={scenarios} corridors={corridors} onPick={pickScenario} />
          ) : (
            <div className="p-10 text-center text-white/50">Загрузка…</div>
          )}
        </div>
      ) : loading ? (
        <div className="p-10 text-center text-white/50">Загрузка…</div>
      ) : (
        <PhoneFrame>
          {screen === "push" && pushEval && (
            <PushToast
              text={pushEval.push_text || "Курс изменился"}
              sentAtMinutes={
                inScenario && activeScenario?.push_sent_at
                  ? hhmm(activeScenario.push_sent_at)
                  : 9 * 60
              }
              onOpen={openPush}
              onClose={() => setScreen("home")}
              stacked={inScenario && activeScenario?.id === "S6"}
              onSettings={
                inScenario && activeScenario?.id === "S6"
                  ? () => setScreen("settings")
                  : undefined
              }
            />
          )}

          {screen !== "home" && screen !== "push" && (
            <div className="px-5 pt-4">
              <BackButton onClick={() => setScreen("home")} />
            </div>
          )}

          {(screen === "home" || screen === "push") && persona && corridor && (
            <Home
              persona={persona}
              corridor={corridor}
              reserve={reserve}
              onStartTransfer={goTransferFromHome}
              onManageReserve={() => setScreen("reserve-manage")}
            />
          )}

          {screen === "country" && corridor && (
            <CountryList
              corridors={corridors}
              current={corridor.corridor}
              onPick={async (c) => {
                if (!inScenario && c !== corridorCode) {
                  setSbCorridor(c);
                  try {
                    setEv(
                      await api.evaluate(evalBody("SELF", simDate, { corridor: c }))
                    );
                  } catch (e) {
                    console.error(e);
                  }
                }
                setScreen("transfer");
              }}
            />
          )}

          {screen === "transfer" && ev && persona && corridor && (
            <TransferScreen
              ev={ev}
              persona={persona}
              corridor={corridor}
              simDate={simDate}
              onTransfer={onTransfer}
              onSecondary={onSecondary}
              onBack={() => setScreen("home")}
            />
          )}

          {screen === "recipient-limit" && limitCheck && (
            <RecipientLimitWarning
              check={limitCheck}
              onSplit={() => setScreen("confirm")}
              onProceed={() => setScreen("confirm")}
              onBack={() => setScreen("transfer")}
            />
          )}

          {screen === "confirm" && ev && corridor && persona && (
            <TransferConfirm
              ev={ev}
              corridor={corridor}
              amountRub={amountRub || persona.typical_amount_rub}
              simDate={simDate}
              onConfirm={onConfirm}
              onBack={() => setScreen("transfer")}
            />
          )}

          {screen === "success" && ev && corridor && persona && (
            <TransferSuccess
              ev={ev}
              corridor={corridor}
              amountRub={amountRub || persona.typical_amount_rub}
              onDone={() => (inScenario ? setScenarioId("") : sandboxReset())}
            />
          )}

          {screen === "settings" && corridor && (
            <NotificationSettings
              corridors={corridors}
              activeCorridor={corridor.corridor}
              overloaded={inScenario && activeScenario?.id === "S6"}
              onSave={() => setScreen("home")}
            />
          )}

          {screen === "reserve-setup" && corridor && persona && (
            <ReserveSetup
              corridor={corridor}
              defaultAmount={amountRub || persona.typical_amount_rub}
              onCreate={createReserve}
              onBack={() => setScreen("transfer")}
            />
          )}
          {screen === "reserve-confirm" && reserve && corridor && (
            <ReserveConfirm rv={reserve} corridor={corridor} onDone={() => setScreen("home")} />
          )}
          {screen === "reserve-manage" && reserve && (
            <ReserveManage
              rv={reserve}
              onCancel={cancelReserve}
              onTransferNow={reserveTransferNow}
              onBack={() => setScreen("home")}
            />
          )}
          {screen === "reserve-executed" && reserve && corridor && (
            <ReserveExecuted
              rv={reserve}
              corridor={corridor}
              onDone={() => (inScenario ? setScenarioId("") : sandboxReset())}
            />
          )}
          {screen === "reserve-expired" && reserve && (
            <ReserveExpired
              rv={reserve}
              onTransferNow={reserveTransferNow}
              onDone={() => (inScenario ? setScenarioId("") : sandboxReset())}
            />
          )}
        </PhoneFrame>
      )}
    </div>
  );
}
