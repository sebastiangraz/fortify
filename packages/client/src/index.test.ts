import { expect, test } from "bun:test";
import { createClient, FortifyError } from "./index.ts";

const group = { id: "s0", logo: { x: 1, y: 2, w: 3, h: 4 }, frames: [new Uint8Array([1, 2, 255])] };

function fake(handler: (url: string, init: RequestInit) => Response | Promise<Response>) {
  return ((url: string, init: RequestInit) => Promise.resolve(handler(url, init))) as typeof fetch;
}

test("sends the contract and decodes deltas", async () => {
  let sent: any;
  const client = createClient({
    url: "http://x/",
    token: "t",
    fetch: fake((url, init) => {
      expect(url).toBe("http://x/v1/vaccinate");
      expect((init.headers as Record<string, string>).authorization).toBe("Bearer t");
      sent = JSON.parse(init.body as string);
      return Response.json({ offset: 128, surrogates: "toy", ms: 5, groups: [{ id: "s0", delta: "gICA", eps: 8, ms: 4 }] });
    }),
  });
  const out = await client.vaccinate({ groups: [group], strength: "high" });
  expect(sent).toEqual({ version: 1, strength: "high", groups: [{ id: "s0", logo: group.logo, frames: ["AQL/"] }] });
  expect([...out.deltas[0]!.png]).toEqual([128, 128, 128]);
  expect(out.deltas[0]!.eps).toBe(8);
});

test("maps failures to typed errors", async () => {
  const rejected = createClient({ url: "http://x", fetch: fake(() => new Response("bad", { status: 400 })) });
  await expect(rejected.vaccinate({ groups: [group] })).rejects.toMatchObject({ code: "rejected", status: 400 });

  const down = createClient({ url: "http://x", fetch: fake(() => new Response("", { status: 503 })) });
  await expect(down.vaccinate({ groups: [group] })).rejects.toMatchObject({ code: "unavailable" });

  const offline = createClient({ url: "http://x", fetch: (() => Promise.reject(new TypeError("refused"))) as any });
  await expect(offline.vaccinate({ groups: [group] })).rejects.toBeInstanceOf(FortifyError);
});

test("times out", async () => {
  const slow = createClient({
    url: "http://x",
    timeoutMs: 20,
    fetch: ((_: string, init: RequestInit) =>
      new Promise((_, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason)))) as any,
  });
  await expect(slow.vaccinate({ groups: [group] })).rejects.toMatchObject({ code: "timeout" });
});
