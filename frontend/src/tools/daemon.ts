let _url = "";

export function setDaemonUrl(url: string): void {
  _url = url.replace(/\/$/, "");
}

export function getDaemonBase(): string {
  return _url || window.location.origin;
}
