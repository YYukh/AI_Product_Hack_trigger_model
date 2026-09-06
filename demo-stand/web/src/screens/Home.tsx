import { ChevronRight } from "lucide-react";
import type { Corridor, Persona, ReserveView } from "../types";
import { Card } from "../components/ui";
import { ddmm, units } from "../store";

// Главная приложения. Единственное живое действие — «За рубеж»; остальные
// пункты — визуальный фон, не кликаются (не «сломанные кнопки»).
export function Home({
  corridor,
  reserve,
  onStartTransfer,
  onManageReserve,
}: {
  persona: Persona;
  corridor: Corridor;
  reserve: ReserveView | null;
  onStartTransfer: () => void;
  onManageReserve: () => void;
}) {
  const showReserve =
    reserve && ["ACTIVE", "EXECUTED", "EXPIRED"].includes(reserve.state);

  return (
    <div className="animate-fade-in px-5 py-4">
      <div className="flex items-center justify-between">
        <h1 className="text-[26px] font-bold">Платежи</h1>
        <div className="h-9 w-9 rounded-full bg-field" />
      </div>

      <div className="mt-4 rounded-product bg-surface p-4 shadow-card">
        <div className="text-[13px] text-text-muted">Текущий счёт</div>
        <div className="text-[24px] font-bold">1 248 300 ₽</div>
      </div>

      <section className="mt-5">
        <h2 className="mb-2 text-[14px] font-semibold text-text-muted">Быстрые переводы</h2>
        <div className="grid grid-cols-3 gap-2.5">
          <button
            onClick={onStartTransfer}
            className="flex h-24 flex-col justify-between rounded-product bg-field p-3 text-left transition active:scale-[.98]"
          >
            <span className="text-2xl">🌍</span>
            <span className="text-[13px] font-medium leading-tight">За рубеж</span>
          </button>
          {["По телефону", "Между счетами"].map((t) => (
            <div
              key={t}
              className="flex h-24 flex-col justify-between rounded-product bg-field/70 p-3 text-text-muted"
            >
              <span className="text-2xl opacity-50">↦</span>
              <span className="text-[13px] font-medium leading-tight">{t}</span>
            </div>
          ))}
        </div>
      </section>

      {showReserve && (
        <button onClick={onManageReserve} className="mt-4 block w-full text-left">
          <Card className="p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="text-[13px] text-text-muted">Ожидание выгодного курса</div>
                <div className="truncate text-[15px] font-semibold">
                  {units(reserve!.amount_rub, "₽")} · {corridor.country}
                </div>
                <div className="mt-1 text-[13px] text-text-muted">
                  {reserve!.state === "ACTIVE" &&
                    `${reserve!.condition_text} · осталось ${reserve!.days_left} дн.`}
                  {reserve!.state === "EXECUTED" &&
                    `Исполнен ${ddmm(reserve!.executed_on!)} · ждали ${reserve!.waited_days} дн.`}
                  {reserve!.state === "EXPIRED" && "Срок истёк, деньги разблокированы"}
                </div>
              </div>
              <ChevronRight className="shrink-0 text-text-muted" />
            </div>
          </Card>
        </button>
      )}

      <section className="mt-5 space-y-2.5">
        {["Деньги за рекомендации", "Альфа-Выгодно"].map((t) => (
          <div
            key={t}
            className="rounded-product bg-field/70 p-4 text-[15px] text-text-muted"
          >
            {t}
          </div>
        ))}
      </section>
    </div>
  );
}

// «За рубеж»: список стран получателя.
export function CountryList({
  corridors,
  current,
  onPick,
}: {
  corridors: Corridor[];
  current: string;
  onPick: (c: string) => void;
}) {
  const order = ["RUB_TJS", "RUB_UZS", "RUB_KGS", "RUB_KZT", "RUB_AMD"];
  const sorted = [...corridors].sort(
    (a, b) => order.indexOf(a.corridor) - order.indexOf(b.corridor)
  );
  return (
    <div className="animate-fade-in px-5 py-4">
      <h1 className="text-[22px] font-bold">Перевод за рубеж</h1>
      <p className="mt-1 text-[13px] text-text-muted">Выберите страну получателя</p>
      <div className="mt-4 space-y-1">
        {sorted.map((c) => (
          <button
            key={c.corridor}
            onClick={() => onPick(c.corridor)}
            className={
              "flex w-full items-center gap-3 rounded-field px-4 py-3 text-left text-[15px] transition " +
              (c.corridor === current ? "bg-field" : "hover:bg-field/60")
            }
          >
            <span className="text-xl">{c.flag}</span>
            <span className="font-medium">{c.country}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
