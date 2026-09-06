import { useState } from "react";
import type { Corridor } from "../types";
import { PrimaryButton, Actions } from "../components/ui";

// B6 — настройки уведомлений о курсе. Путь без выхода «пишите реже» не
// соответствует коммуникационной политике, поэтому экран есть всегда.
export function NotificationSettings({
  corridors,
  activeCorridor,
  overloaded,
  onSave,
}: {
  corridors: Corridor[];
  activeCorridor: string;
  overloaded?: boolean;
  onSave: () => void;
}) {
  const [enabled, setEnabled] = useState<Record<string, boolean>>(
    Object.fromEntries(corridors.map((c) => [c.corridor, c.corridor === activeCorridor]))
  );
  const [freq, setFreq] = useState<"low" | "normal" | "off">(
    overloaded ? "off" : "normal"
  );
  const [quiet, setQuiet] = useState(true);

  return (
    <div className="animate-fade-in px-5 py-4">
      <h1 className="text-[22px] font-bold">Уведомления о курсе</h1>

      {overloaded && (
        <div className="mt-4 rounded-field bg-plaque p-4 text-[14px] leading-relaxed">
          За последние 3 дня вы получили 3 уведомления по этому коридору. Это выше
          обычного бюджета канала (не более 1–2 в неделю). Можно снизить частоту или
          отключить уведомления полностью.
        </div>
      )}

      <div className="mt-5">
        <div className="mb-2 text-[13px] text-text-muted">Коридоры</div>
        <div className="space-y-1">
          {corridors.map((c) => (
            <label
              key={c.corridor}
              className="flex items-center justify-between rounded-field bg-field px-4 py-3 text-[15px]"
            >
              <span>
                {c.flag} {c.country}
              </span>
              <input
                type="checkbox"
                checked={!!enabled[c.corridor]}
                onChange={(e) => setEnabled({ ...enabled, [c.corridor]: e.target.checked })}
              />
            </label>
          ))}
        </div>
      </div>

      <div className="mt-5">
        <div className="mb-2 text-[13px] text-text-muted">Частота</div>
        <div className="flex gap-2">
          {(
            [
              ["low", "Реже"],
              ["normal", "Обычно"],
              ["off", "Отключить"],
            ] as const
          ).map(([v, l]) => (
            <button
              key={v}
              onClick={() => setFreq(v)}
              className={
                "h-10 flex-1 rounded-full px-3 text-[14px] transition " +
                (freq === v ? "bg-cta text-white" : "bg-field text-text")
              }
            >
              {l}
            </button>
          ))}
        </div>
      </div>

      <label className="mt-4 flex items-center justify-between rounded-field bg-field px-4 py-3 text-[15px]">
        <span>Тихие часы 22:00–09:00</span>
        <input type="checkbox" checked={quiet} onChange={(e) => setQuiet(e.target.checked)} />
      </label>

      <Actions>
        <PrimaryButton onClick={onSave}>Сохранить</PrimaryButton>
      </Actions>
    </div>
  );
}
