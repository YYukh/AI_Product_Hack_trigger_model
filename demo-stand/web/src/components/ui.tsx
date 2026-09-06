import React from "react";

// Примитивы по дизайн-системе: обводок нет, только заливки и отступы.
// На телефоне действия — во всю ширину, стопкой.

export function PrimaryButton({
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={
        "flex h-12 w-full items-center justify-center rounded-btn bg-cta px-6 text-[16px] font-semibold text-white " +
        "transition active:scale-[.99] disabled:opacity-40 " +
        (rest.className || "")
      }
    >
      {children}
    </button>
  );
}

export function SecondaryButton({
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={
        "flex h-12 w-full items-center justify-center rounded-btn bg-field px-6 text-[16px] font-semibold text-text " +
        "transition active:scale-[.99] disabled:opacity-40 " +
        (rest.className || "")
      }
    >
      {children}
    </button>
  );
}

export function GhostButton({
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={
        "mx-auto block py-2 text-[14px] text-text-muted underline-offset-2 hover:underline " +
        (rest.className || "")
      }
    >
      {children}
    </button>
  );
}

// Вертикальная стопка действий внизу экрана.
export function Actions({ children }: { children: React.ReactNode }) {
  return <div className="mt-6 space-y-2">{children}</div>;
}

export function Field({
  label,
  value,
  hint,
  right,
  onChange,
  placeholder,
  inputMode,
}: {
  label: string;
  value: string;
  hint?: string;
  right?: React.ReactNode;
  onChange?: (v: string) => void;
  placeholder?: string;
  inputMode?: "numeric" | "text";
}) {
  const readOnly = !onChange;
  return (
    <div className="rounded-field bg-field px-4 py-2.5">
      <div className="text-[13px] text-text-muted">{label}</div>
      <div className="flex items-center gap-2">
        <input
          className="w-full bg-transparent text-[17px] font-semibold text-text outline-none placeholder:font-normal placeholder:text-text-muted"
          value={value}
          placeholder={placeholder}
          inputMode={inputMode}
          readOnly={readOnly}
          onChange={(e) => onChange?.(e.target.value)}
        />
        {right}
      </div>
      {hint && <div className="mt-0.5 text-[13px] text-text-muted">{hint}</div>}
    </div>
  );
}

export function Chip({
  children,
  active,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return (
    <button
      {...rest}
      className={
        "h-10 rounded-full px-4 text-[14px] transition " +
        (active ? "bg-cta text-white " : "bg-field text-text ") +
        (rest.className || "")
      }
    >
      {children}
    </button>
  );
}

export function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={"rounded-content bg-surface shadow-card " + className}>{children}</div>
  );
}

export function BackButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      aria-label="Назад"
      className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-surface text-text shadow-card transition active:scale-95"
    >
      <span className="text-xl leading-none">‹</span>
    </button>
  );
}
