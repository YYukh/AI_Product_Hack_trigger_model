import type { Corridor, Scenario } from "./types";

// Стартовый экран стенда: список сценариев. Тап по карточке запускает отыгрыш
// в мобильном приложении (пуш → экран перевода → подтверждение).
export function Launcher({
  scenarios,
  corridors,
  onPick,
}: {
  scenarios: Scenario[];
  corridors: Corridor[];
  onPick: (id: string) => void;
}) {
  const corr = (c: string) => corridors.find((x) => x.corridor === c);

  return (
    <div className="mx-auto w-full max-w-[460px] px-4 py-7 text-white">
      <h1 className="text-[22px] font-bold">Триггерный слой переводов</h1>
      <p className="mt-1.5 text-[13px] leading-snug text-white/55">
        Демонстрационный стенд. Выберите сценарий — он проигрывается как в мобильном
        приложении: уведомление, экран перевода, подтверждение.
      </p>

      <div className="mt-6 space-y-2.5">
        {scenarios.map((s) => {
          const c = corr(s.corridor);
          return (
            <button
              key={s.id}
              onClick={() => onPick(s.id)}
              className="block w-full rounded-2xl bg-white/[.06] p-4 text-left transition hover:bg-white/[.10] active:scale-[.99]"
            >
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[12px] text-white/45">
                <span className="rounded-md bg-white/10 px-1.5 py-0.5 font-semibold text-white/70">
                  {s.id}
                </span>
                <span>
                  {c?.flag} {c?.country ?? s.corridor}
                </span>
                <span>·</span>
                <span>{s.entry === "PUSH" ? "приходит пуш" : "клиент сам зашёл"}</span>
                {s.optional && <span className="text-white/30">· опционально</span>}
              </div>
              <div className="mt-1.5 text-[15px] font-semibold">{s.title}</div>
              <div className="mt-1 text-[13px] leading-snug text-white/55">{s.summary}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
