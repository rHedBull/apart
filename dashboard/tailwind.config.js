/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        danger: {
          power: '#ef4444',      // red-500
          deception: '#f97316',  // orange-500
          exploit: '#eab308',    // yellow-500
        },
      },
    },
  },
  plugins: [],
}
