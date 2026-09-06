import { useState, type ReactNode } from "react";
import type { Corridor } from "../types";
import type { Entry, Tab } from "../store";
import { ddmm, ddmmyyyy, rate } from "../store";

// Переключатель верхнего уровня: Песочница ⇄ Сценарии.
export function TopTabs({ tab, onTab }: { tab: Tab; onTab: (t: Tab) => void }) {
  const items: [Tab, string][] = [
    ["sandbox", "Песочница"],
    ["scenarios", "Сценарии"],
  ];
  return (
    <div className="mx-auto flex w-full max-w-[393px] gap-1 rounded-full bg-white/10 p-1">
      {items.map(([t, label]) => (
        <button
          key={t}
          onClick={() => onTab(t)}
          className={
            "flex-1 rounded-full py-1.5 text-[13px] font-medium transition " +
            (tab === t ? "bg-white text-[#15161A]" : "text-white/70 hover:text-white")
          }
        >
          {label}
        </button>
      ))}
    </div>
  );
}

// Панель «Сценарии» в режиме проигрывания: назад к списку, заголовок, повтор, +день.
export function ScenarioBar({
  title,
  onList,
  onReplay,
  onPlusDay,
}: {
  title: string;
  onList: () => void;
  onReplay: () => void;
  onPlusDay?: () => void;
}) {
  return (
    <div className="mx-auto flex w-full max-w-[393px] items-center gap-2 text-[13px] text-white/70">
      <button
        onClick={onList}
        className="shrink-0 rounded-full bg-white/10 px-3 py-1 font-medium text-white/85 transition active:scale-95"
      >
        ‹ К списку
      </button>
      <span className="min-w-0 flex-1 truncate text-center">{title}</span>
      {onPlusDay && (
        <button
          onClick={onPlusDay}
          className="shrink-0 rounded-full bg-white/10 px-3 py-1 font-medium text-white/85 transition active:scale-95"
        >
          +1 день
        </button>
      )}
      <button
        onClick={onReplay}
        aria-label="Проиграть заново"
        className="shrink-0 rounded-full bg-white/10 px-2.5 py-1 font-medium text-white/85 transition active:scale-95"
      >
        ↻
      </button>
    </div>
  );
}

// Панель «Песочница»: краткая сводка + раскрывающиеся параметры сессии.
export function SandboxBar({
  corridors,
  corridor,
  onCorridor,
  date,
  dateMin,
  dateMax,
  onShiftDay,
  onToday,
  entry,
  onEntry,
  pushDaysAgo,
  onPushDaysAgo,
  pushRate,
  pushDate,
  onReset,
  onPlusDay,
}: {
  corridors: Corridor[];
  corridor: string;
  onCorridor: (c: string) => void;
  date: string;
  dateMin: string;
  dateMax: string;
  onShiftDay: (delta: number) => void;
  onToday: () => void;
  entry: Entry;
  onEntry: (e: Entry) => void;
  pushDaysAgo: number;
  onPushDaysAgo: (n: number) => void;
  pushRate: number | null;
  pushDate: string | null;
  onReset: () => void;
  onPlusDay: () => void;
}) {
  const [open, setOpen] = useState(false);
  const c = corridors.find((x) => x.corridor === corridor);
  const summary = `${c?.flag ?? ""} ${c?.country ?? corridor} · ${ddmm(date)} · ${
    entry === "PUSH" ? "по пушу" : "сам зашёл"
  }`;

  return (
    <div className="mx-auto w-full max-w-[393px] text-white/80">
      <div className="flex items-center gap-2 text-[13px]">
        <button
          onClick={() => setOpen((v) => !v)}
          className="min-w-0 flex-1 truncate rounded-full bg-white/10 px-3 py-1 text-left font-medium text-white/85 transition active:scale-[.99]"
        >
          {open ? "▾ " : "▸ "}
          {summary}
        </button>
        <button
          onClick={onPlusDay}
          className="shrink-0 rounded-full bg-white/10 px-3 py-1 font-medium text-white/85 transition active:scale-95"
        >
          +1 день
        </button>
        <button
          onClick={onReset}
          aria-label="На главный экран"
          className="shrink-0 rounded-full bg-white/10 px-2.5 py-1 font-medium text-white/85 transition active:scale-95"
        >
          ↻
        </button>
      </div>

      {open && (
        <div className="mt-2 space-y-3 rounded-2xl bg-white/[.06] p-3 text-[13px]">
          <Row label="Страна получателя">
            <div className="flex flex-wrap gap-1.5">
              {corridors.map((x) => (
                <button
                  key={x.corridor}
                  onClick={() => onCorridor(x.corridor)}
                  className={
                    "rounded-full px-2.5 py-1 transition " +
                    (x.corridor === corridor
                      ? "bg-white text-[#15161A]"
                      : "bg-white/10 text-white/75 hover:text-white")
                  }
                >
                  {x.flag} {x.country}
                </button>
              ))}
            </div>
          </Row>

          <Row label="Дата курса">
            <div className="flex items-center gap-1.5">
              <Step onClick={() => onShiftDay(-1)} disabled={date <= dateMin}>
                −
              </Step>
              <span className="min-w-[92px] text-center font-semibold text-white">
                {ddmmyyyy(date)}
              </span>
              <Step onClick={() => onShiftDay(1)} disabled={date >= dateMax}>
                +
              </Step>
              <button
                onClick={onToday}
                className="ml-1 rounded-full bg-white/10 px-2.5 py-1 text-white/75 transition hover:text-white"
              >
                последняя
              </button>
            </div>
          </Row>

          <Row label="Как открыл приложение">
            <div className="flex gap-1.5">
              {(
                [
                  ["SELF", "Сам зашёл"],
                  ["PUSH", "Пришёл по пушу"],
                ] as [Entry, string][]
              ).map(([e, label]) => (
                <button
                  key={e}
                  onClick={() => onEntry(e)}
                  className={
                    "rounded-full px-3 py-1 transition " +
                    (entry === e
                      ? "bg-white text-[#15161A]"
                      : "bg-white/10 text-white/75 hover:text-white")
                  }
                >
                  {label}
                </button>
              ))}
            </div>
          </Row>

          {entry === "PUSH" && (
            <Row label="Курс в пуше — с даты">
              <div className="flex items-center gap-1.5">
                <Step onClick={() => onPushDaysAgo(Math.max(1, pushDaysAgo - 1))}>−</Step>
                <span className="min-w-[92px] text-center font-semibold text-white">
                  {pushDaysAgo} дн. назад
                </span>
                <Step onClick={() => onPushDaysAgo(Math.min(15, pushDaysAgo + 1))}>+</Step>
                <span className="ml-1 text-white/55">
                  {pushRate != null
                    ? `${rate(pushRate)} ₽${pushDate ? ` · ${ddmm(pushDate)}` : ""}`
                    : "…"}
                </span>
              </div>
            </Row>
          )}
        </div>
      )}
    </div>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-1 text-white/45">{label}</div>
      {children}
    </div>
  );
}

function Step({
  children,
  onClick,
  disabled,
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="flex h-7 w-7 items-center justify-center rounded-full bg-white/10 text-[15px] text-white/80 transition active:scale-95 disabled:opacity-30"
    >
      {children}
    </button>
  );
}
