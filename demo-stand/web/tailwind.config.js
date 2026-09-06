/** @type {import('tailwindcss').Config} */
// Токены сняты со скриншотов веб-приложения (ТЗ §9.6). Менять только по
// согласованию — кейс прямо оценивает попадание в существующий стиль.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#EBECEF",
        surface: "#FFFFFF",
        field: "#F0F1F3",
        text: { DEFAULT: "#0A0A0A", muted: "#8A8F98" },
        cta: "#000000",
        accent: "#EF3124",
        info: { bg: "#E9F1FC", text: "#2C4A6E" },
        plaque: "#EEF1F5",
      },
      borderRadius: {
        content: "24px",
        product: "20px",
        field: "16px",
        btn: "14px",
      },
      boxShadow: {
        card: "0 2px 8px rgba(0,0,0,.04)",
        toast: "0 8px 24px rgba(0,0,0,.10)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
      },
      transitionDuration: { DEFAULT: "180ms" },
    },
  },
  plugins: [],
};
