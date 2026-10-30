# Dodge dashboard frontend

React, Vite, Tailwind and shadcn/ui; charts are Vega-Lite specs that
`dodge/charts.py` builds with Altair and the server sends ready to render.

    pnpm install
    pnpm run build     # writes dist/, which dashboard.py serves
    pnpm run dev       # hot reload; proxies /api to dashboard.py on :8765
