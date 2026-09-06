import { useState } from "react";
import type { Corridor, Evaluation, Persona } from "../types";
import { PrimaryButton, SecondaryButton, Field, Chip, GhostButton, Actions } from "../components/ui";
import { Plaque } from "../components/Plaque";
import { ddmm, money, rate, units } from "../store";

const DISCLAIMER = "Курс перевода может отличаться от официального";

// B3 — экран перевода. Один экран; над формой — плашка, под ней — действия.
export function TransferScreen({
  ev,
  persona,
  corridor,
  simDate,
  onTransfer,
  onSecondary,
  onBack,
}: {
  ev: Evaluation;
  persona: Persona;
  corridor: Corridor;
  simDate: string;
  onTransfer: (amountRub: number) => void;
  onSecondary: () => void;
  onBack: () => void;
}) {
  // Предзаполнение только при state OK/BETTER и входе по пушу.
  const [phone, setPhone] = useState(ev.prefill ? persona.recipient_phone : "");
  const [purpose, setPurpose] = useState(ev.prefill ? "Помощь семье" : "");
  const [amount, setAmount] = useState(
    ev.prefill ? String(persona.typical_amount_rub) : ""
  );
  const amountNum = Number(amount.replace(/\s/g, "")) || 0;

  const recipientGets = amountNum > 0 ? Math.round(amountNum / ev.current_rate) : null;
  const canSend = amountNum >= 100 && phone.trim() !== "" && purpose.trim() !== "";

  const primaryLabel = ev.state === "DRIFT" ? "Перевести всё равно" : "Перевести";
  const secondaryLabel =
    ev.actions.secondary === "RESERVE"
      ? "Дождаться выгодного курса"
      : ev.actions.secondary === "RETURN"
        ? "Вернуться"
        : null;

  return (
    <div className="animate-fade-in px-5 py-4">
      <h1 className="text-[22px] font-bold leading-tight">Перевод по телефону за рубеж</h1>
      <p className="mt-1 text-[13px] text-text-muted">
        {corridor.flag} {corridor.country} · без комиссии, от 100 ₽ до 800 000 ₽
      </p>

      <div className="mt-4">
        <Plaque ev={ev} />
      </div>

      <div className="mt-4 space-y-3">
        <Field label="Счёт списания" value="Текущий счёт · 1 248 300 ₽" />
        <Field
          label="Номер телефона получателя"
          value={phone}
          onChange={setPhone}
          placeholder="+000 00 000-00-00"
          hint={`${persona.recipient_bank}`}
          inputMode="text"
        />
        <Field
          label="Назначение перевода"
          value={purpose}
          onChange={setPurpose}
          placeholder="обязательное поле"
        />
        <Field
          label="Сумма"
          value={amount}
          onChange={(v) => setAmount(v.replace(/[^\d]/g, ""))}
          placeholder="0"
          inputMode="numeric"
          right={<span className="text-[15px] text-text-muted">₽</span>}
        />
        <div className="flex flex-wrap gap-2">
          {[5000, 10000, 20000].map((s) => (
            <Chip key={s} active={amountNum === s} onClick={() => setAmount(String(s))}>
              {money(s)}
            </Chip>
          ))}
          <Chip onClick={() => setAmount(String(persona.typical_amount_rub))}>
            обычная сумма
          </Chip>
        </div>

        <div className="pt-1">
          <div className="text-[15px]">
            Получатель получит ≈{" "}
            <span className="font-semibold">
              {recipientGets != null
                ? `${units(recipientGets, corridor.currency_name)}`
                : "—"}
            </span>{" "}
            <span className="text-text-muted">
              по курсу ЦБ на {ddmm(simDate)}
              {ev.current_rate_is_stale ? " (перенесён с прошлого дня)" : ""}
            </span>
          </div>
          <div className="text-[13px] text-text-muted">{DISCLAIMER}</div>
        </div>
      </div>

      <Actions>
        <PrimaryButton disabled={!canSend} onClick={() => onTransfer(amountNum)}>
          {primaryLabel}
        </PrimaryButton>
        {secondaryLabel ? (
          <SecondaryButton onClick={onSecondary}>{secondaryLabel}</SecondaryButton>
        ) : (
          <GhostButton onClick={onBack}>Не сейчас</GhostButton>
        )}
      </Actions>
    </div>
  );
}

// B4 — подтверждение.
export function TransferConfirm({
  ev,
  corridor,
  amountRub,
  simDate,
  onConfirm,
  onBack,
}: {
  ev: Evaluation;
  corridor: Corridor;
  amountRub: number;
  simDate: string;
  onConfirm: () => void;
  onBack: () => void;
}) {
  const recipientGets = Math.round(amountRub / ev.current_rate);
  const rows: [string, string][] = [
    ["Сумма списания", money(amountRub)],
    ["Курс ЦБ на " + ddmm(simDate), `${rate(ev.current_rate)} ₽ за ${corridor.currency_acc}`],
    ["Получатель получит", units(recipientGets, corridor.currency_name)],
    ["Комиссия", "Без комиссии"],
    ["Срок зачисления", "обычно в течение часа"],
  ];
  return (
    <div className="animate-fade-in px-5 py-4">
      <h1 className="text-[22px] font-bold">Подтверждение перевода</h1>
      <div className="mt-4 rounded-field bg-field p-4">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3 py-2 text-[15px]">
            <span className="text-text-muted">{k}</span>
            <span className="text-right font-semibold">{v}</span>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[13px] text-text-muted">
        Показан официальный курс ЦБ РФ. Курс перевода может отличаться.
      </p>
      <Actions>
        <PrimaryButton onClick={onConfirm}>Подтвердить</PrimaryButton>
        <GhostButton onClick={onBack}>Назад</GhostButton>
      </Actions>
    </div>
  );
}

// B5 — успех.
export function TransferSuccess({
  ev,
  corridor,
  amountRub,
  onDone,
}: {
  ev: Evaluation;
  corridor: Corridor;
  amountRub: number;
  onDone: () => void;
}) {
  const recipientGets = Math.round(amountRub / ev.current_rate);
  return (
    <div className="animate-fade-in px-5 py-8 text-center">
      <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-field text-2xl">
        ✓
      </div>
      <h1 className="mt-4 text-[22px] font-bold">Перевод отправлен</h1>
      <p className="mt-2 text-[17px] font-semibold">
        {units(recipientGets, corridor.currency_name)} получателю
      </p>
      <p className="mt-1 text-[13px] text-text-muted">Зачисление обычно в течение часа</p>
      <Actions>
        <PrimaryButton onClick={onDone}>Готово</PrimaryButton>
      </Actions>
    </div>
  );
}
