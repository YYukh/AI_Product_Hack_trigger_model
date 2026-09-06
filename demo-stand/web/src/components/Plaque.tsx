import { ArrowDownRight, Equal, Info, TrendingUp } from "lucide-react";
import type { Evaluation } from "../types";
import { rate, units } from "../store";

// Один компонент, четыре состояния OK / DRIFT / BETTER / NEUTRAL.
// Заливка одинакова во всех состояниях (ТЗ §9.6): зелёного и красного нет,
// цвет читался бы как вердикт, а вердикта мы не даём. Различие — иконка и текст.

const ICON = {
  OK: Info,
  BETTER: TrendingUp,
  NEUTRAL: Equal,
  DRIFT: ArrowDownRight,
} as const;

export function Plaque({ ev }: { ev: Evaluation }) {
  const Icon = ICON[ev.state];
  const p = ev.plaque;

  return (
    <div className="rounded-field bg-plaque p-5">
      <div className="flex gap-3">
        <Icon size={20} strokeWidth={1.75} className="mt-0.5 shrink-0 text-text-muted" />
        <div className="min-w-0">
          {ev.state === "DRIFT" ? <DriftBody ev={ev} /> : (
            <p className="text-[15px] font-medium leading-snug text-text">{p.text}</p>
          )}
          {p.context && ev.state === "DRIFT" && (
            <p className="mt-1 text-[13px] leading-snug text-text-muted">{p.context}</p>
          )}
        </div>
      </div>
    </div>
  );
}

// Для DRIFT всегда показываем обе величины и дельту в валюте получателя,
// а не только в процентах (BFR-10).
function DriftBody({ ev }: { ev: Evaluation }) {
  return (
    <div>
      <p className="text-[15px] font-medium leading-snug text-text">{ev.plaque.text}</p>
      {ev.push_rate != null && (
        <div className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[13px] text-text-muted">
          <span>В уведомлении</span>
          <span className="font-semibold text-text">{rate(ev.push_rate)} ₽</span>
          <span>Сейчас</span>
          <span className="font-semibold text-text">{rate(ev.current_rate)} ₽</span>
          <span>Разница</span>
          <span className="font-semibold text-text">
            {ev.delta_bp > 0 ? "+" : ""}
            {(Math.abs(ev.delta_bp) / 100).toFixed(2).replace(".", ",")}%
            {ev.recipient_delta != null && (
              <>
                {" · "}
                −{units(ev.recipient_delta, ev.recipient_currency_short)} у получателя
              </>
            )}
          </span>
        </div>
      )}
    </div>
  );
}
