// Client for the fortify service. Mirrors src/fortify/service.py; the README's
// "API contract" section is the source of truth for both.

export type Strength = "low" | "medium" | "high";

export type Rect = { x: number; y: number; w: number; h: number };

export type VaccinateGroup = {
  /** Echoed back so the caller can match results (e.g. "stay-3"). */
  id: string;
  /** Where the mark sits inside the crop, in crop pixels. */
  logo: Rect;
  /** PNG bytes of the watermarked crops; all the same size, at most 8. */
  frames: Uint8Array[];
  /**
   * Where the crop sits in the full frame. Optional, but without it detectors that run on
   * whole frames (Florence-2) are attacked at the wrong scale and the shield misses them.
   */
  view?: View;
};

export type View = {
  /** Full frame size in pixels. */
  frame: { w: number; h: number };
  /** The crop's top-left corner in the frame. */
  at: { x: number; y: number };
  /** PNG bytes of the whole frame at any size; a ≤ 1024 px thumbnail is plenty. */
  background?: Uint8Array;
};

export type Delta = {
  id: string;
  /** RGB PNG, same size as the crops. Pixel value = offset + δ·255. */
  png: Uint8Array;
  /** |δ| bound in 8-bit levels. */
  eps: number;
  ms: number;
};

export type VaccinateResult = { offset: number; surrogates: string; deltas: Delta[]; ms: number };

export type FortifyErrorCode = "unavailable" | "timeout" | "rejected" | "bad_response";

export class FortifyError extends Error {
  constructor(
    readonly code: FortifyErrorCode,
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "FortifyError";
  }
}

/** The δ wire offset: a PNG value of 128 means "no change". */
export const DELTA_OFFSET = 128;

export type ClientOptions = {
  url: string;
  token?: string;
  /** Whole-request budget. On timeout the call throws FortifyError("timeout"). */
  timeoutMs?: number;
  fetch?: typeof fetch;
};

export function createClient({ url, token, timeoutMs = 120_000, fetch: doFetch = fetch }: ClientOptions) {
  const base = url.replace(/\/+$/, "");

  async function call(path: string, init: RequestInit, signal?: AbortSignal): Promise<unknown> {
    const timeout = AbortSignal.timeout(timeoutMs);
    const signals = signal ? AbortSignal.any([signal, timeout]) : timeout;
    let res: Response;
    try {
      res = await doFetch(base + path, {
        ...init,
        signal: signals,
        headers: {
          "content-type": "application/json",
          ...(token ? { authorization: `Bearer ${token}` } : {}),
        },
      });
    } catch (error) {
      if (timeout.aborted) throw new FortifyError("timeout", `fortify: no answer in ${timeoutMs} ms`);
      if (signal?.aborted) throw error;
      throw new FortifyError("unavailable", `fortify: ${(error as Error).message}`);
    }
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      const code = res.status >= 500 ? "unavailable" : "rejected";
      throw new FortifyError(code, `fortify ${res.status}: ${detail.slice(0, 300)}`, res.status);
    }
    return res.json().catch(() => {
      throw new FortifyError("bad_response", "fortify: response is not JSON");
    });
  }

  return {
    async health(signal?: AbortSignal) {
      return (await call("/health", { method: "GET" }, signal)) as Record<string, unknown>;
    },

    async vaccinate({
      groups,
      strength = "medium",
      signal,
    }: {
      groups: VaccinateGroup[];
      strength?: Strength;
      signal?: AbortSignal;
    }): Promise<VaccinateResult> {
      const body = JSON.stringify({
        version: 1,
        strength,
        groups: groups.map((g) => ({
          id: g.id,
          logo: g.logo,
          frames: g.frames.map(toBase64),
          ...(g.view ? { view: toWireView(g.view) } : {}),
        })),
      });
      const out = (await call("/v1/vaccinate", { method: "POST", body }, signal)) as {
        offset?: number;
        surrogates?: string;
        ms?: number;
        groups?: { id: string; delta: string; eps: number; ms: number }[];
      };
      if (!Array.isArray(out?.groups) || out.groups.length !== groups.length) {
        throw new FortifyError("bad_response", "fortify: group count does not match the request");
      }
      return {
        offset: out.offset ?? DELTA_OFFSET,
        surrogates: out.surrogates ?? "",
        ms: out.ms ?? 0,
        deltas: out.groups.map((g) => ({ id: g.id, png: fromBase64(g.delta), eps: g.eps, ms: g.ms })),
      };
    },
  };
}

export type FortifyClient = ReturnType<typeof createClient>;

function toWireView({ frame, at, background }: View) {
  return { frame, at, ...(background ? { background: toBase64(background) } : {}) };
}

function toBase64(bytes: Uint8Array): string {
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(text);
}

function fromBase64(text: string): Uint8Array {
  const raw = atob(text);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}
