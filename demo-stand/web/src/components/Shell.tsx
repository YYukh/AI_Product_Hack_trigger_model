import type { ReactNode } from "react";

// Рамка телефона: на десктопе — устройство по центру, на узком экране — во весь
// вьюпорт без бортика. Внутри — статус-бар, прокручиваемый контент, home-бар.
export function PhoneFrame({
  statusDark,
  children,
}: {
  statusDark?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="relative mx-auto flex w-full min-h-0 max-w-[393px] flex-1 flex-col overflow-hidden bg-bg shadow-2xl sm:h-[852px] sm:max-h-[852px] sm:flex-none sm:rounded-[52px] sm:border-[10px] sm:border-black">
      <StatusBar dark={statusDark} />
      <div className="relative min-h-0 flex-1 overflow-y-auto overscroll-contain">
        {children}
      </div>
      <div className="flex justify-center pb-2 pt-1">
        <div className="h-1 w-32 rounded-full bg-black/20" />
      </div>
    </div>
  );
}

function StatusBar({ dark }: { dark?: boolean }) {
  return (
    <div
      className={
        "flex items-center justify-between px-7 pb-1 pt-3 text-[13px] font-semibold " +
        (dark ? "text-white" : "text-text")
      }
    >
      <span>9:41</span>
      <span className="flex items-center gap-1.5">
        <span className="flex items-end gap-[2px]">
          {[3, 5, 7, 9].map((h) => (
            <span
              key={h}
              className="w-[3px] rounded-[1px] bg-current"
              style={{ height: h }}
            />
          ))}
        </span>
        <span className="text-[11px]">5G</span>
        <span className="relative inline-block h-[11px] w-[22px] rounded-[3px] border border-current">
          <span className="absolute inset-[2px] right-[4px] rounded-[1px] bg-current" />
        </span>
      </span>
    </div>
  );
}
