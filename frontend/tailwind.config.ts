import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        forest: {
          50: "#eef7f0",
          100: "#d9ebdc",
          300: "#8fc49b",
          500: "#367d4a",
          700: "#1f5b35",
          800: "#17462a",
          900: "#113820"
        },
        cream: "#f7f5ee",
        ink: "#18211b",
        sand: "#e8e2d4"
      },
      boxShadow: {
        soft: "0 18px 50px rgba(31, 72, 48, 0.10)"
      }
    }
  },
  plugins: []
};

export default config;
