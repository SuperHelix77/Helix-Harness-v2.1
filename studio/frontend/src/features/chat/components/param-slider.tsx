// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { NumericValueInput, snapToStep } from "@/features/model-picker";
import { InfoHint } from "@/components/ui/info-hint";
import { Slider } from "@/components/ui/slider";
import type { ReactNode } from "react";

/** Shared sampling slider used by Chat and the media pages. Kept outside the
 * Run Settings sheet so importing one slider does not pull the entire closed
 * settings surface into first paint. */
export function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  displayValue,
  info,
  valueSize,
  disabled,
  inline,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  displayValue?: string;
  info?: ReactNode;
  valueSize?: number;
  disabled?: boolean;
  /** Label, track and value on one row, for narrow settings columns. */
  inline?: boolean;
}) {
  if (inline) {
    return (
      <div className="flex items-center gap-3">
        <div className="flex min-w-[104px] shrink-0 items-center gap-1.5">
          <span className="text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg">
            {label}
          </span>
          {info && <InfoHint>{info}</InfoHint>}
        </div>
        <Slider
          min={min}
          max={max}
          step={step}
          value={[value]}
          onValueChange={([v]) => onChange(snapToStep(v, step, min, max))}
          className="panel-slider min-w-0 flex-1"
          disabled={disabled}
        />
        <NumericValueInput
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
          displayValue={displayValue}
          ariaLabel={label}
          size={valueSize ?? 4}
          className="panel-number-input"
          disabled={disabled}
        />
      </div>
    );
  }
  return (
    <div className="space-y-3.5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg">
            {label}
          </span>
          {info && <InfoHint>{info}</InfoHint>}
        </div>
        <NumericValueInput
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
          displayValue={displayValue}
          ariaLabel={label}
          size={valueSize ?? 4}
          className="panel-number-input"
          disabled={disabled}
        />
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(snapToStep(v, step, min, max))}
        className="panel-slider"
        disabled={disabled}
      />
    </div>
  );
}

