// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useState } from "react";
import { authFetch } from "@/features/auth";

type LocalModel = { name: string; path: string; bytes: number };

export function ModelsPage() {
  const [root, setRoot] = useState("");
  const [models, setModels] = useState<LocalModel[]>([]);
  const [status, setStatus] = useState("Select a local folder of GGUF files, then load one.");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.title = "Helix Harness";
  }, []);

  async function scan() {
    setError(null);
    const response = await authFetch(
      `/api/helix-engine/hub/local?root=${encodeURIComponent(root)}`,
    );
    const body = (await response.json()) as { models?: LocalModel[] };
    setModels(body.models ?? []);
    setStatus(`${(body.models ?? []).length} local GGUF models`);
  }

  async function select(path: string) {
    setError(null);
    const selected = await authFetch("/api/helix-engine/hub/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const chosen = (await selected.json()) as { ok?: boolean; error?: string; path?: string };
    if (!chosen.ok || !chosen.path) {
      setError(chosen.error ?? "Could not select model");
      return;
    }
    setStatus(`Loading ${chosen.path}`);
    const response = await authFetch("/api/helix-engine/hub/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: chosen.path }),
    });
    const body = (await response.json()) as {
      ok?: boolean;
      error?: string;
      path?: string;
      model?: string;
      loaded?: boolean;
    };
    if (!body.ok || !body.loaded) {
      setError(body.error ?? "Could not load model");
      return;
    }
    setStatus(`Loaded ${body.model ?? body.path}`);
  }

  async function download(path: string) {
    setError(null);
    const dest = root.trim() || ".";
    const response = await authFetch("/api/helix-engine/hub/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: path, dest }),
    });
    const body = (await response.json()) as { ok?: boolean; error?: string; path?: string };
    if (!body.ok) {
      setError(body.error ?? "Could not stage model");
      return;
    }
    setStatus(`Ready ${body.path}`);
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-4 p-6" data-testid="helix-simple-hub">
      <div>
        <h1 className="text-lg font-semibold">Helix Harness models</h1>
        <p className="text-sm text-muted-foreground">
          Simpler hub: list local GGUF files, select one, load through Chat. Built on upstream Unsloth components.
        </p>
      </div>
      <label className="flex flex-col gap-1 text-sm">
        Models folder
        <input
          className="rounded-md border border-border bg-background px-3 py-2"
          value={root}
          onChange={(event) => setRoot(event.target.value)}
          placeholder="/path/to/gguf"
        />
      </label>
      <button type="button" className="w-fit rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground" onClick={() => void scan()}>
        Scan
      </button>
      {error ? <p className="text-sm text-destructive">{error}</p> : <p className="text-sm text-muted-foreground">{status}</p>}
      <ul className="min-h-0 flex-1 space-y-2 overflow-y-auto">
        {models.map((model) => (
          <li key={model.path} className="flex items-center justify-between rounded-lg border border-border p-3">
            <span className="truncate text-sm">{model.name}</span>
            <span className="flex gap-3">
              <button type="button" className="text-sm text-primary" onClick={() => void download(model.path)}>
                Stage
              </button>
              <button type="button" className="text-sm text-primary" onClick={() => void select(model.path)}>
                Select
              </button>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
