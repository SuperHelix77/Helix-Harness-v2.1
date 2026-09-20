// SPDX-License-Identifier: AGPL-3.0-only
// Screenshot-only release showcase. Every name, prompt, path, and receipt is synthetic.

import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleDot,
  Database,
  FileCheck2,
  GitBranch,
  LockKeyhole,
  MessageSquareText,
  PanelRight,
  Search,
  Send,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TerminalSquare,
  Wrench,
  X,
} from "lucide-react";
import type { CSSProperties } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

declare global {
  interface Window {
    __releaseShowcaseReady?: boolean;
  }
}

const css = String.raw`
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body, #root { width: 100%; height: 100%; margin: 0; overflow: hidden; }
  body { font-family: "Inter Variable", -apple-system, BlinkMacSystemFont, sans-serif; background: #0a1115; }
  button { font: inherit; }
  .release-stage {
    width: 100%; height: 100%; padding: 42px;
    display: grid; place-items: center;
    color: #f4f7f8;
    background:
      radial-gradient(circle at 76% 20%, rgba(107, 160, 174, .7), transparent 28%),
      radial-gradient(circle at 29% 72%, rgba(50, 91, 104, .78), transparent 32%),
      linear-gradient(135deg, #12232b 0%, #315963 42%, #18303a 73%, #091117 100%);
  }
  .release-stage::before {
    content: ""; position: fixed; inset: 0; pointer-events: none;
    background: linear-gradient(112deg, transparent 0 46%, rgba(255,255,255,.1) 46.2%, transparent 46.5% 100%);
    opacity: .38;
  }
  .window {
    width: min(1360px, calc(100vw - 84px)); height: min(816px, calc(100vh - 84px));
    position: relative; display: grid; grid-template-columns: 278px minmax(0, 1fr);
    overflow: hidden; border-radius: 22px;
    border: 1px solid rgba(255,255,255,.14);
    background: rgba(10, 16, 20, .38);
    box-shadow: 0 34px 90px rgba(0,0,0,.45), 0 2px 8px rgba(0,0,0,.24);
    backdrop-filter: blur(34px) saturate(128%);
  }
  .sidebar {
    position: relative; min-width: 0; padding: 23px 15px 15px;
    display: flex; flex-direction: column;
    background: rgba(13, 21, 26, .6); border-right: 1px solid rgba(255,255,255,.08);
  }
  .traffic { display: flex; gap: 8px; margin: 0 0 24px 4px; }
  .traffic i { width: 11px; height: 11px; border-radius: 50%; display: block; box-shadow: inset 0 0 0 1px rgba(0,0,0,.15); }
  .traffic i:nth-child(1) { background:#ff5f57 } .traffic i:nth-child(2){background:#febc2e}.traffic i:nth-child(3){background:#28c840}
  .brand { display:flex; align-items:center; gap:10px; padding: 0 8px 18px; letter-spacing:.12em; font-weight:650; font-size:12px; }
  .brand img { width:25px; height:25px; filter: invert(1) drop-shadow(0 4px 10px rgba(0,0,0,.26)); }
  .nav-action { display:flex; align-items:center; gap:9px; height:34px; margin: 0 2px 12px; padding:0 11px; border:1px solid rgba(255,255,255,.08); border-radius:11px; background:rgba(255,255,255,.045); color:rgba(244,247,248,.78); font-size:12px; }
  .nav-action svg { width:14px; }
  .section-label { padding: 10px 10px 7px; color:rgba(229,237,240,.39); text-transform:uppercase; letter-spacing:.13em; font-size:9px; font-weight:700; }
  .thread { display:grid; grid-template-columns: 18px minmax(0,1fr) auto; gap:8px; align-items:center; min-height:44px; padding:7px 9px; border-radius:11px; color:rgba(239,244,246,.62); }
  .thread.active { color:#f7fafb; background:rgba(255,255,255,.075); box-shadow:inset 0 0 0 1px rgba(255,255,255,.04); }
  .thread svg { width:14px; opacity:.65; }
  .thread-copy { min-width:0; display:flex; flex-direction:column; gap:2px; }
  .thread-copy b { overflow:hidden; white-space:nowrap; text-overflow:ellipsis; font-size:11.5px; font-weight:550; }
  .thread-copy span { color:rgba(229,237,240,.34); font-size:9.5px; }
  .thread time { align-self:start; padding-top:1px; color:rgba(229,237,240,.3); font-size:9px; }
  .sidebar-bottom { margin-top:auto; padding-top:12px; border-top:1px solid rgba(255,255,255,.07); }
  .engine { display:flex; align-items:center; gap:10px; padding:8px; border-radius:12px; }
  .engine img { width:27px; height:27px; filter:invert(1); } .engine b {display:block;font-size:10px;letter-spacing:.1em}.engine span{font-size:9px;color:rgba(229,237,240,.36)}
  .workspace { position:relative; min-width:0; display:flex; flex-direction:column; background:rgba(13,19,23,.2); }
  .topbar { height:58px; padding:0 21px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,.065); }
  .title { display:flex; align-items:center; gap:10px; font-size:12px; font-weight:590; }
  .title .dot { width:6px;height:6px;border-radius:50%;background:#78bba8;box-shadow:0 0 0 4px rgba(120,187,168,.1); }
  .top-actions { display:flex; align-items:center; gap:7px; }
  .icon-button { width:31px;height:31px;display:grid;place-items:center;border:0;border-radius:10px;color:rgba(239,244,246,.62);background:transparent; }
  .icon-button.active {color:#f7fafb;background:rgba(255,255,255,.08)} .icon-button svg{width:15px}
  .chat-layout { min-height:0; flex:1; display:grid; grid-template-columns:minmax(0,1fr); }
  .chat-layout.rail-open { grid-template-columns:minmax(0,1fr) 330px; }
  .conversation { min-width:0; min-height:0; display:flex; flex-direction:column; }
  .messages { width:min(760px, calc(100% - 84px)); margin:0 auto; padding:58px 0 146px; display:flex; flex-direction:column; gap:34px; }
  .user-message { align-self:flex-end; max-width:590px; padding:14px 17px; border:1px solid rgba(255,255,255,.09); border-radius:18px 18px 6px 18px; background:rgba(255,255,255,.07); font-size:13px; line-height:1.55; box-shadow:0 10px 24px rgba(0,0,0,.08); }
  .assistant { display:grid; grid-template-columns:27px minmax(0,1fr); gap:13px; font-size:13px; line-height:1.62; color:rgba(244,247,248,.86); }
  .assistant-mark { width:27px;height:27px;padding:4px;border-radius:8px;background:rgba(255,255,255,.07);filter:invert(1); }
  .assistant h2 { margin:0 0 12px; color:#fff; font-size:17px; letter-spacing:-.025em; }
  .assistant p { margin:0 0 13px; }
  .evidence-row { display:flex; flex-wrap:wrap; gap:7px; margin-top:17px; }
  .evidence { display:flex; align-items:center; gap:6px; padding:6px 9px; border-radius:9px; background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.07); color:rgba(231,239,242,.62); font-size:10px; }
  .evidence svg{width:12px}.verified{color:#a6d4c8}
  .composer-wrap { position:absolute; left:0; right:0; bottom:0; padding:24px 40px 26px; pointer-events:none; background:linear-gradient(transparent,rgba(9,14,18,.5) 56%); }
  .composer-wrap.rail-open { right:330px; }
  .composer { pointer-events:auto; width:min(820px,100%); min-height:84px; margin:0 auto; padding:14px 15px 12px; display:flex; flex-direction:column; justify-content:space-between; border-radius:21px; border:1px solid rgba(255,255,255,.14); background:rgba(22,29,34,.54); backdrop-filter:blur(26px) saturate(135%); box-shadow:0 18px 44px rgba(0,0,0,.24); }
  .composer .prompt { color:rgba(240,245,247,.42); font-size:13px; }
  .composer-footer { display:flex; align-items:center; justify-content:space-between; }
  .pills {display:flex;gap:7px}.pill{display:flex;align-items:center;gap:5px;padding:5px 8px;border-radius:8px;background:rgba(255,255,255,.045);color:rgba(231,239,242,.48);font-size:9.5px}.model-pill{border:1px solid rgba(255,255,255,.09);color:rgba(240,246,247,.72)}.model-pill b{font-size:8px;letter-spacing:.1em;color:#98c9bd}.model-pill svg{width:10px}.send{width:30px;height:30px;display:grid;place-items:center;border-radius:50%;border:0;background:#edf3f4;color:#132029}.send svg{width:14px}
  .right-rail { min-width:0; padding:19px 17px; border-left:1px solid rgba(255,255,255,.075); background:rgba(12,19,23,.58); }
  .rail-head {display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}.rail-head b{font-size:12px}.rail-head span{font-size:9px;color:rgba(229,237,240,.4)}
  .graph-node {position:relative;margin:0 0 12px 16px;padding:11px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.045)}
  .graph-node::before{content:"";position:absolute;left:-17px;top:20px;width:10px;border-top:1px solid rgba(140,192,181,.45)}
  .graph-node::after{content:"";position:absolute;left:-17px;top:-13px;bottom:calc(100% - 20px);border-left:1px solid rgba(140,192,181,.35)}
  .graph-node small{display:block;color:#83c6b4;font-size:8px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}.graph-node b{font-size:10.5px;font-weight:580}.graph-node p{margin:5px 0 0;color:rgba(229,237,240,.42);font-size:9px;line-height:1.45}
  .settings-scrim {position:absolute;inset:0;display:grid;place-items:center;background:rgba(4,8,10,.35);backdrop-filter:blur(8px)}
  .settings-panel {width:980px;height:680px;display:grid;grid-template-columns:225px minmax(0,1fr);overflow:hidden;border-radius:20px;border:1px solid rgba(255,255,255,.14);background:rgba(15,22,27,.9);box-shadow:0 30px 80px rgba(0,0,0,.48);backdrop-filter:blur(38px)}
  .settings-nav{padding:18px 12px;background:rgba(255,255,255,.025);border-right:1px solid rgba(255,255,255,.07)}
  .settings-heading{display:flex;align-items:center;justify-content:space-between;padding:1px 9px 17px}.settings-heading b{font-size:13px}.settings-heading svg{width:14px;color:rgba(255,255,255,.45)}
  .settings-group{margin:8px 0 5px;padding:0 9px;color:rgba(229,237,240,.32);font-size:8.5px;letter-spacing:.11em;text-transform:uppercase}
  .settings-item{height:34px;display:flex;align-items:center;gap:9px;padding:0 10px;border-radius:9px;color:rgba(237,243,245,.56);font-size:10.5px}.settings-item svg{width:13px}.settings-item.active{background:rgba(255,255,255,.075);color:#fff}
  .settings-content{padding:30px 38px;overflow:hidden}.settings-content h1{margin:0;font-size:22px;letter-spacing:-.03em}.settings-content>.sub{margin:6px 0 26px;color:rgba(231,239,242,.42);font-size:11px}
  .setting-card{padding:18px 19px;margin-bottom:14px;border-radius:15px;border:1px solid rgba(255,255,255,.075);background:rgba(255,255,255,.025)}
  .setting-card h3{margin:0 0 14px;font-size:11px;font-weight:620}.preset-row{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.preset{padding:12px;border-radius:11px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);color:rgba(240,245,247,.55);font-size:10px}.preset.active{border-color:rgba(135,190,179,.55);background:rgba(110,166,157,.09);color:#f6faf9}.slider-row{display:grid;grid-template-columns:145px 1fr 38px;gap:12px;align-items:center;min-height:32px;color:rgba(240,245,247,.58);font-size:10px}.track{height:3px;border-radius:3px;background:rgba(255,255,255,.1);position:relative}.track i{position:absolute;inset:0 auto 0 0;width:var(--amount);border-radius:inherit;background:#83bdb0}.track::after{content:"";position:absolute;left:var(--amount);top:50%;width:9px;height:9px;border-radius:50%;background:#dfe8e6;transform:translate(-50%,-50%)}
  .privacy-note {position:absolute;right:53px;bottom:16px;color:rgba(235,241,243,.38);font-size:9px;letter-spacing:.04em}
`;

const threads = [
  ["Reliability release brief", "Synthetic workspace", "now"],
  ["Review recovery receipts", "Durable tools", "12m"],
  ["Context policy comparison", "Runtime", "1h"],
  ["Prepare public documentation", "Release", "3h"],
  ["Audit capability boundaries", "Security", "1d"],
] as const;

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="traffic"><i /><i /><i /></div>
      <div className="brand"><img src="/helix-mark.svg" alt="" />HELIX HARNESS</div>
      <div className="nav-action"><Search /> Search conversations <span style={{ marginLeft: "auto", opacity: .45 }}>⌘K</span></div>
      <div className="section-label">Workspace</div>
      {threads.map(([title, scope, time], index) => (
        <div className={`thread ${index === 0 ? "active" : ""}`} key={title}>
          <MessageSquareText /><div className="thread-copy"><b>{title}</b><span>{scope}</span></div><time>{time}</time>
        </div>
      ))}
      <div className="section-label">Archived</div>
      <div className="thread"><Database /><div className="thread-copy"><b>Benchmark evidence</b><span>4 archived threads</span></div><ChevronRight /></div>
      <div className="sidebar-bottom">
        <div className="engine"><img src="/helix-mark.svg" alt="" /><div><b>HELIX ENGINE</b><span>in-app control plane</span></div></div>
      </div>
    </aside>
  );
}

function Header({ rail }: { rail: boolean }) {
  return <header className="topbar"><div className="title"><span className="dot" />Reliability release brief <span style={{ color: "rgba(230,238,241,.3)", fontSize: 9 }}>SYNTHETIC DEMO</span></div><div className="top-actions"><button className="icon-button"><GitBranch /></button><button className={`icon-button ${rail ? "active" : ""}`}><PanelRight /></button><button className="icon-button"><Settings2 /></button></div></header>;
}

function Chat({ rail = false }: { rail?: boolean }) {
  return <><Header rail={rail} /><div className={`chat-layout ${rail ? "rail-open" : ""}`}><main className="conversation"><div className="messages"><div className="user-message">Prepare a release-readiness summary. Verify the durable tool receipts, account boundaries, and packaged backend identity. Keep every claim traceable to evidence.</div><div className="assistant"><img className="assistant-mark" src="/helix-mark.svg" alt="Helix" /><div><h2>The release candidate is internally consistent.</h2><p>I verified the recovery path against the persisted execution frontier: completed tool receipts remain replay-only, approval identity is bound to the exact account and payload, and ambiguous mutating outcomes still fail closed.</p><p>The packaged runtime reports the expected Helix backend contract. The public snapshot contains source and synthetic documentation only; local accounts, memories, benchmark traces, and machine paths are excluded.</p><div className="evidence-row"><div className="evidence verified"><CheckCircle2 /> VERIFIED · 156 focused tests</div><div className="evidence"><FileCheck2 /> receipt · run_7f3a…91c2</div><div className="evidence"><ShieldCheck /> account boundary intact</div><div className="evidence"><Wrench /> 3 durable tool calls</div></div></div></div></div></main>{rail && <ExecutionRail />}</div><Composer rail={rail} /></>;
}

function Composer({ rail }: { rail: boolean }) {
  return <div className={`composer-wrap ${rail ? "rail-open" : ""}`}><div className="composer"><div className="prompt">Ask Helix to inspect, build, test, or explain…</div><div className="composer-footer"><div className="pills"><span className="pill model-pill"><b>Model</b>Local Senior · 27B<ChevronDown /></span><span className="pill">Durable tools</span><span className="pill">32K Auto</span></div><button className="send"><Send /></button></div></div></div>;
}

function ExecutionRail() {
  return <aside className="right-rail"><div className="rail-head"><div><b>Execution Graph</b><br/><span>Evidence-linked causal path</span></div><X style={{ width: 14, opacity: .45 }} /></div><div className="graph-node"><small>Intent · verified input</small><b>Qualify release candidate</b><p>Constraints preserved from the durable task state.</p></div><div className="graph-node"><small>Action · read only</small><b>Inspect packaged identity</b><p>backend manifest · sha256: 7bd705…dced</p></div><div className="graph-node"><small>Observation · receipt</small><b>Recovery suite passed</b><p>156 focused assertions · no duplicate side effects</p></div><div className="graph-node"><small>Claim · supported</small><b>Candidate is internally consistent</b><p>Supported by receipts above; final installation smoke pending.</p></div><div style={{ marginTop: 20, padding: 12, borderRadius: 12, background: "rgba(126,185,172,.07)", border: "1px solid rgba(126,185,172,.15)", fontSize: 9.5, color: "rgba(226,239,235,.58)", lineHeight: 1.5 }}><LockKeyhole style={{ width: 13, display: "inline", marginRight: 7, verticalAlign: -3 }} />The graph records provenance. It does not fabricate causality.</div></aside>;
}

const settingsGroups = [
  ["Workspace", [[SlidersHorizontal, "Appearance"], [MessageSquareText, "Chat & context"], [Sparkles, "Voice"]]],
  ["Identity & data", [[CircleDot, "Profile & memory"], [ShieldCheck, "Accounts & isolation"], [Database, "Memory & data"]]],
  ["Runtime & access", [[TerminalSquare, "Runtime & models"], [Wrench, "Model providers"], [LockKeyhole, "API access"]]],
] as const;

function AppearanceSettings() {
  return <div className="settings-scrim"><section className="settings-panel"><nav className="settings-nav"><div className="settings-heading"><b>Settings</b><X /></div>{settingsGroups.map(([group, rows]) => <div key={group}><div className="settings-group">{group}</div>{rows.map(([Icon, label]) => <div key={label} className={`settings-item ${label === "Appearance" ? "active" : ""}`}><Icon />{label}</div>)}</div>)}</nav><main className="settings-content"><h1>Appearance</h1><p className="sub">A quiet, native workspace with independently controlled materials.</p><div className="setting-card"><h3>Color and material</h3><div className="preset-row"><div className="preset">Opaque<br/><small>Deterministic solid surfaces</small></div><div className="preset active">Balanced<br/><small>Readable native translucency</small></div><div className="preset">Airy<br/><small>Wallpaper-forward workspace</small></div></div></div><div className="setting-card"><h3>Advanced material controls</h3>{[["Main window", "66%", "66%"], ["Sidebar", "54%", "54%"], ["Chat surface", "28%", "28%"], ["Composer", "82%", "82%"], ["Right rail", "62%", "62%"]].map(([label, value, amount]) => <div className="slider-row" key={label}><span>{label}</span><div className="track" style={{ "--amount": amount } as CSSProperties}><i /></div><span>{value}</span></div>)}</div><div className="setting-card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}><div><h3 style={{ marginBottom: 5 }}>Accessibility fallback</h3><span style={{ color: "rgba(231,239,242,.42)", fontSize: 10 }}>Reduced Transparency returns every surface to an opaque, readable material.</span></div><ShieldCheck style={{ width: 20, color: "#8fc8bb" }} /></div></main></section></div>;
}

function Showcase() {
  const view = new URLSearchParams(location.search).get("view") ?? "chat";
  return <><style>{css}</style><div className="release-stage"><div className="window"><Sidebar /><section className="workspace">{view === "settings" ? <><Chat /><AppearanceSettings /></> : <Chat rail={view === "graph"} />}</section>{view !== "settings" && <span className="privacy-note">Synthetic release content · no user data</span>}</div></div></>;
}

const root = document.getElementById("root");
if (!root) throw new Error("Root element not found");
createRoot(root).render(<Showcase />);
requestAnimationFrame(() => { window.__releaseShowcaseReady = true; });
