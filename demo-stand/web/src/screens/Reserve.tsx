import { useState } from "react";
import type { Corridor, ReserveView } from "../types";
import { PrimaryButton, SecondaryButton, GhostButton, Actions } from "../components/ui";
import { ddmm, money, rate, units } from "../store";

const PRESETS = [
  { id: "careful", label: "Осторожный", percentile: 40, window: 30, note: "обычно за 2–3 дня" },
  { id: "normal", label: "Обычный", percentile: 25, window: 30, note: "обычно за 5–7 дней" },
  { id: "patient", label: "Терпеливый", percentile: 10, window: 60, note: "не всегда, зато выигрыш больше" },
];

// R1 — настройка резерва. Резерв никогда не предвыбран и не основное действие.
export function ReserveSetup({
  defaultAmount,
  onCreate,
  onBack,
}: {
  corridor: Corridor;
  defaultAmount: number;
  onCreate: (opts: {
    amountRub: number;
    percentile: number;
    windowDays: number;
    fallback: boolean;
  }) => void;
  onBack: () => void;
}) {
  const [amount, setAmount] = useState(String(defaultAmount));
  const [preset, setPreset] = useState(PRESETS[1]);
  const [fallback, setFallback] = useState(false);
  const amountNum = Number(amount.replace(/\D/g, "")) || 0;

  return (
    <div className="animate-fade-in px-5 py-4">
      <h1 className="text-[22px] font-bold">Дождаться выгодного курса</h1>

      <div className="mt-4 rounded-field bg-plaque p-4 text-[14px] leading-relaxed">
        <div>
          <span className="text-text-muted">Заблокируем на счёте: </span>
          <span className="font-semibold">{money(amountNum)}</span>
        </div>
        <div>
          <span className="text-text-muted">Вернём, если условие не наступит за: </span>
          <span className="font-semibold">7 дней</span>
        </div>
        <div>
          <span className="text-text-muted">Отмена: </span>
          <span className="font-semibold">в один тап, мгновенно</span>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        <div className="rounded-field bg-field px-4 py-2.5">
          <div className="text-[13px] text-text-muted">Сумма</div>
          <input
            className="w-full bg-transparent text-[17px] font-semibold outline-none"
            inputMode="numeric"
            value={amount}
            onChange={(e) => setAmount(e.target.value.replace(/\D/g, ""))}
          />
        </div>

        <div className="text-[13px] text-text-muted">Условие</div>
        <div className="space-y-2">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => setPreset(p)}
              className={
                "block w-full rounded-field px-4 py-3 text-left transition " +
                (preset.id === p.id ? "bg-cta text-white" : "bg-field text-text")
              }
            >
              <div className="text-[15px] font-semibold">{p.label}</div>
              <div
                className={
                  "text-[13px] " + (preset.id === p.id ? "text-white/70" : "text-text-muted")
                }
              >
                курс в нижних {p.percentile}% за {p.window} дней · {p.note}
              </div>
            </button>
          ))}
        </div>

        <label className="flex items-start gap-3 rounded-field bg-field px-4 py-3 text-[14px]">
          <input
            type="checkbox"
            checked={fallback}
            onChange={(e) => setFallback(e.target.checked)}
            className="mt-0.5"
          />
          <span>
            Если не сработает — отправить по текущему курсу
            <span className="block text-[13px] text-text-muted">
              курс может оказаться хуже сегодняшнего. По умолчанию выключено.
            </span>
          </span>
        </label>
      </div>

      <Actions>
        <SecondaryButton
          onClick={() =>
            onCreate({
              amountRub: amountNum,
              percentile: preset.percentile,
              windowDays: preset.window,
              fallback,
            })
          }
        >
          Зарезервировать
        </SecondaryButton>
        <GhostButton onClick={onBack}>Назад</GhostButton>
      </Actions>
    </div>
  );
}

// R2 — подтверждение резерва.
export function ReserveConfirm({
  rv,
  onDone,
}: {
  rv: ReserveView;
  corridor: Corridor;
  onDone: () => void;
}) {
  return (
    <div className="animate-fade-in px-5 py-4">
      <h1 className="text-[22px] font-bold">Резерв создан</h1>
      <div className="mt-4 rounded-field bg-field p-4 text-[15px]">
        <Row k="Заблокировано" v={money(rv.amount_rub)} />
        <Row k="Условие" v={rv.condition_text} />
        <Row k="Дата истечения" v={`через ${rv.ttl_days} дн.`} />
        <Row
          k="Если условие не наступит"
          v={rv.fallback_send_on_expiry ? "отправим по текущему курсу" : "деньги вернутся"}
        />
      </div>
      <p className="mt-3 text-[13px] text-text-muted">
        Отменить можно в любой момент на главной — карточка «Ожидание выгодного курса».
        Кнопкой «+1 день» вверху можно промотать время и увидеть исполнение.
      </p>
      <Actions>
        <PrimaryButton onClick={onDone}>Готово</PrimaryButton>
      </Actions>
    </div>
  );
}

// R3 — управление резервом.
export function ReserveManage({
  rv,
  onCancel,
  onTransferNow,
  onBack,
}: {
  rv: ReserveView;
  onCancel: () => void;
  onTransferNow: () => void;
  onBack: () => void;
}) {
  const above = rv.distance_bp > 0;
  return (
    <div className="animate-fade-in px-5 py-4">
      <h1 className="text-[22px] font-bold">Ожидание выгодного курса</h1>
      <div className="mt-4 rounded-field bg-field p-4 text-[15px]">
        <Row k="Сумма" v={money(rv.amount_rub)} />
        <Row k="Условие" v={rv.condition_text} />
        <Row k="Осталось" v={`${rv.days_left} дн.`} />
        <Row
          k="Курс относительно порога"
          v={
            above
              ? `выше порога на ${Math.abs(rv.distance_bp).toFixed(0)} б.п.`
              : `достиг порога`
          }
        />
      </div>
      <div className="mt-4 rounded-field bg-plaque p-4 text-[13px] text-text-muted">
        Порог: {rate(rv.threshold_rate)} ₽ · сейчас {rate(rv.current_rate)} ₽. Условие
        проверяется раз в сутки после публикации курса ЦБ.
      </div>
      <Actions>
        <PrimaryButton onClick={onTransferNow}>Перевести сейчас</PrimaryButton>
        <SecondaryButton onClick={onCancel}>Отменить резерв</SecondaryButton>
        <GhostButton onClick={onBack}>Назад</GhostButton>
      </Actions>
    </div>
  );
}

// R4 — исполнение резерва.
export function ReserveExecuted({
  rv,
  corridor,
  onDone,
}: {
  rv: ReserveView;
  corridor: Corridor;
  onDone: () => void;
}) {
  return (
    <div className="animate-fade-in px-5 py-8 text-center">
      <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-field text-2xl">
        ✓
      </div>
      <h1 className="mt-4 text-[22px] font-bold">Резерв исполнен</h1>
      <div className="mt-4 rounded-field bg-field p-4 text-left text-[15px]">
        <Row k="Отправлено" v={money(rv.amount_rub)} />
        <Row k="Курс исполнения" v={`${rate(rv.exec_rate)} ₽`} />
        <Row k="Получатель получит" v={units(rv.recipient_gets_now, corridor.currency_name)} />
        <Row k="Дней ожидания" v={String(rv.waited_days)} />
        <Row
          k="Выигрыш к дню оформления"
          v={`${rv.gain_bp && rv.gain_bp > 0 ? "+" : ""}${rv.gain_bp?.toFixed(0)} б.п.`}
        />
      </div>
      <Actions>
        <PrimaryButton onClick={onDone}>Готово</PrimaryButton>
      </Actions>
    </div>
  );
}

// R5 — истечение резерва. Плохой сценарий не прячем.
export function ReserveExpired({
  rv,
  onTransferNow,
  onDone,
}: {
  rv: ReserveView;
  onTransferNow: () => void;
  onDone: () => void;
}) {
  return (
    <div className="animate-fade-in px-5 py-4">
      <h1 className="text-[22px] font-bold">Срок резерва истёк</h1>
      <div className="mt-4 rounded-field bg-plaque p-4 text-[15px] leading-relaxed">
        Курс не опускался до порога {rate(rv.threshold_rate)} ₽ в течение {rv.ttl_days} дней.
        Деньги разблокированы, перевод не выполнен.
      </div>
      <Actions>
        <PrimaryButton onClick={onTransferNow}>Перевести сейчас</PrimaryButton>
        <GhostButton onClick={onDone}>Готово</GhostButton>
      </Actions>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4 py-2">
      <span className="text-text-muted">{k}</span>
      <span className="text-right font-semibold">{v}</span>
    </div>
  );
}
