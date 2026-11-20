export type BackendConfig = { port: number; token: string };
export const DEFAULT_READINESS_TIMEOUT_MS = 60000;
export type BackendStartupDeps = {
  isDevelopment: boolean;
  developmentBase?: string;
  listen: (handler: (config: BackendConfig) => void) => Promise<() => void>;
  invoke: () => Promise<BackendConfig>;
  waitUntilReady: (base: string, signal: AbortSignal, timeoutMs?: number) => Promise<void>;
  readinessTimeoutMs?: number;
  onReady: (base: string, token?: string) => void;
  onError: (message: string) => void;
};

export function startBackend(deps: BackendStartupDeps): () => void {
  let active = true;
  let disposeListener: (() => void) | undefined;
  const controller = new AbortController();
  const run = async () => {
    try {
      if (deps.isDevelopment) {
        const base = deps.developmentBase;
        if (!base) throw new Error('Development API base URL is not configured.');
        await deps.waitUntilReady(base, controller.signal, deps.readinessTimeoutMs ?? DEFAULT_READINESS_TIMEOUT_MS);
        if (active) deps.onReady(base);
        return;
      }

      let eventConfig: BackendConfig | undefined;
      const listenerReady = deps.listen(config => { eventConfig = config; });
      disposeListener = await listenerReady;
      if (!active) { disposeListener(); return; }

      const config = await deps.invoke();
      // The command is authoritative; the event is only a recovery signal.
      const selected = config || eventConfig;
      if (!selected || !Number.isInteger(selected.port) || !selected.token) throw new Error('Tauri did not provide backend connection details.');
      const base = `http://127.0.0.1:${selected.port}`;
      await deps.waitUntilReady(base, controller.signal, deps.readinessTimeoutMs ?? DEFAULT_READINESS_TIMEOUT_MS);
      if (active) deps.onReady(base, selected.token);
    } catch (error) {
      if (active && !controller.signal.aborted) deps.onError(`Backend startup failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };
  void run();
  return () => { active = false; controller.abort(); disposeListener?.(); };
}
