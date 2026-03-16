import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#FF9500", light: "#FFB340", dark: "#E08500", contrastText: "#fff" },
    secondary: { main: "#764ba2" },
    success: { main: "#22c55e" },
    error: { main: "#ef4444" },
    background: { default: "#0A0E17", paper: "#111827" },
    text: { primary: "#F1F5F9", secondary: "#94A3B8" },
    divider: "rgba(255, 255, 255, 0.12)",
  },
  typography: {
    fontFamily: "'Inter', sans-serif",
    h6: { fontWeight: 600 },
  },
  shape: { borderRadius: 8 },
  components: {
    MuiCssBaseline: { styleOverrides: { body: { margin: 0, overflow: "hidden" } } },
    MuiButton: {
      defaultProps: { disableRipple: false },
      styleOverrides: {
        root: {
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          textTransform: "none" as const,
          fontWeight: 600,
          borderRadius: "10px",
          fontSize: "0.8125rem",
        },
        outlined: {
          borderColor: "#FF9500",
          color: "#FF9500",
          "&:hover": { borderColor: "#E08500", backgroundColor: "rgba(255, 149, 0, 0.08)" },
        },
      },
    },
    MuiPaper: { styleOverrides: { root: { backgroundImage: "none" } } },
    MuiCard: { styleOverrides: { root: { borderRadius: "10px" } } },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          fontSize: "11px",
          fontWeight: 500,
          padding: "10px 14px",
          borderRadius: "8px",
          backgroundColor: "rgba(255, 255, 255, 0.95)",
          color: "#1d1d1f",
        },
      },
    },
    MuiOutlinedInput: { styleOverrides: { root: { borderRadius: "8px" } } },
    MuiToggleButton: { styleOverrides: { root: { textTransform: "none" as const, borderRadius: "8px" } } },
  },
});

export default theme;
