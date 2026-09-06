import { X } from "lucide-react";
import { minutesToHHMM } from "../store";

// B1 — системное уведомление, съезжает сверху экрана телефона. Ровно один факт
// с числом и периодом, без глаголов будущего и «успейте».
export function PushToast({
  text,
  sentAtMinutes,
  onOpen,
  onClose,
  onSettings,
  stacked,
}: {
  text: string;
  sentAtMinutes: number;
  onOpen: () => void;
  onClose: () => void;
  onSettings?: () => void;
  stacked?: boolean;
}) {
  return (
    <div className="animate-toast-in absolute inset-x-0 top-0 z-30 px-3 pt-2">
      <div className="relative mx-auto max-w-[380px]">
        {stacked && (
          <>
            <div className="absolute inset-x-3 top-3 h-full scale-[.97] rounded-[22px] bg-[#dfe1e6] shadow-toast" />
            <div className="absolute inset-x-1.5 top-1.5 h-full scale-[.985] rounded-[22px] bg-[#eceef1] shadow-toast" />
          </>
        )}
        <div className="relative rounded-[22px] bg-surface p-3.5 shadow-toast ring-1 ring-black/5">
          <div className="flex items-start gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[9px] bg-cta text-[13px] font-bold text-white">
              A
            </div>
            <button className="min-w-0 flex-1 text-left" onClick={onOpen}>
              <div className="flex items-center justify-between gap-2">
                <span className="text-[14px] font-semibold text-text">Альфа-Онлайн</span>
                <span className="text-[11px] text-text-muted">
                  {minutesToHHMM(sentAtMinutes)}
                </span>
              </div>
              <p className="mt-0.5 text-[13px] leading-snug text-text">{text}</p>
            </button>
            <button
              aria-label="Закрыть уведомление"
              onClick={onClose}
              className="shrink-0 text-text-muted transition hover:text-text"
            >
              <X size={15} />
            </button>
          </div>
          {onSettings && (
            <button
              onClick={onSettings}
              className="mt-2.5 w-full rounded-btn bg-field py-2 text-[12px] font-medium text-accent transition active:scale-[.99]"
            >
              Слишком часто — настроить уведомления
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
