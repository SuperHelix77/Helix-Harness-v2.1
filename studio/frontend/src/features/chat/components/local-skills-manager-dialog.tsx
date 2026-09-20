// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState, type ReactElement } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import {
  createLocalSkill,
  installLocalSkill,
  listLocalSkills,
  type LocalSkill,
} from "../api/local-skills-api";
import { LOCAL_SKILLS_CHANGED_EVENT, LOCAL_SKILLS_OPEN_EVENT } from "../lib/local-skills";

export function LocalSkillsManagerDialog(): ReactElement {
  const [open, setOpen] = useState(false);
  const [skills, setSkills] = useState<LocalSkill[]>([]);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [instructions, setInstructions] = useState("");
  const [source, setSource] = useState("");
  const [target, setTarget] = useState<"codex" | "claude" | "both">("codex");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => {
    void listLocalSkills().then(setSkills).catch((error) => {
      toast.error("Could not list local skills", { description: error instanceof Error ? error.message : undefined });
    });
  }, []);

  useEffect(() => {
    const openManager = () => {
      setOpen(true);
      refresh();
    };
    window.addEventListener(LOCAL_SKILLS_OPEN_EVENT, openManager);
    return () => window.removeEventListener(LOCAL_SKILLS_OPEN_EVENT, openManager);
  }, [refresh]);

  const run = async (action: () => Promise<LocalSkill[]>, message: string) => {
    setBusy(true);
    try {
      setSkills(await action());
      window.dispatchEvent(new Event(LOCAL_SKILLS_CHANGED_EVENT));
      toast.success(message);
    } catch (error) {
      toast.error("Skill operation failed", { description: error instanceof Error ? error.message : undefined });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="corner-squircle dialog-soft-surface max-h-[85vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Local skills</DialogTitle>
          <DialogDescription>
            Install read-only SKILL.md packages from GitHub or create a skill usable by this app, Codex, and Claude Code.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-5">
          <section className="grid gap-2">
            <h3 className="text-sm font-medium">Installed</h3>
            <div className="max-h-32 overflow-y-auto rounded-xl border border-border/70 bg-muted/20 p-2 text-xs">
              {skills.length ? skills.map((skill) => (
                <div key={`${skill.path}:${skill.name}`} className="flex items-center justify-between gap-2 border-b border-border/50 py-1.5 last:border-0">
                  <span className="truncate"><span className="font-medium">{skill.name}</span> · {skill.description}</span>
                  <span className="shrink-0 text-muted-foreground">{skill.ecosystems.join(" + ")}</span>
                </div>
              )) : <span className="text-muted-foreground">No local skills found yet.</span>}
            </div>
          </section>
          <section className="grid gap-2">
            <h3 className="text-sm font-medium">Install from GitHub or a local folder</h3>
            <Input value={source} onChange={(event) => setSource(event.target.value)} placeholder="https://github.com/owner/skills-repo" />
            <div className="flex gap-2">
              <select value={target} onChange={(event) => setTarget(event.target.value as typeof target)} className="h-9 rounded-md border border-input bg-background px-2 text-sm">
                <option value="codex">Codex</option>
                <option value="claude">Claude Code</option>
                <option value="both">Codex + Claude</option>
              </select>
              <Button disabled={busy || !source.trim()} onClick={() => void run(() => installLocalSkill({ source: source.trim(), target }), "Skill installed")}>Install</Button>
            </div>
          </section>
          <section className="grid gap-2">
            <h3 className="text-sm font-medium">Create a skill</h3>
            <Input value={name} onChange={(event) => setName(event.target.value)} placeholder="skill-name" />
            <Input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="What this skill does" />
            <Textarea value={instructions} onChange={(event) => setInstructions(event.target.value)} placeholder="Instructions for the agent..." className="min-h-28" />
            <div className="flex justify-end">
              <Button disabled={busy || !name.trim() || !description.trim() || !instructions.trim()} onClick={() => void run(() => createLocalSkill({ name: name.trim(), description: description.trim(), instructions: instructions.trim(), target }), "Skill created")}>Create</Button>
            </div>
          </section>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
