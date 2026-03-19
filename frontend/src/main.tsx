import React, { useMemo } from "react";
import ReactDOM from "react-dom/client";
import CssBaseline from "@mui/material/CssBaseline";
import { ThemeProvider } from "@mui/material/styles";
import useMediaQuery from "@mui/material/useMediaQuery";
import { lightTheme, darkTheme } from "./theme";
import App from "./App";

function useThemeFromContext() {
  const params = useMemo(() => new URLSearchParams(window.location.search), []);
  const explicitTheme = params.get("theme");
  const systemDark = useMediaQuery("(prefers-color-scheme: dark)");

  if (explicitTheme === "dark") return darkTheme;
  if (explicitTheme === "light") return lightTheme;
  return systemDark ? darkTheme : lightTheme;
}

function Root() {
  const theme = useThemeFromContext();

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
