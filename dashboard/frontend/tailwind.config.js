/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'bg-primary': '#0a0e17',
                'bg-secondary': '#111827',
                'bg-card': '#1f2937',
                'border': '#374151',
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
        },
    },
    plugins: [],
}
