import { LogEntry } from "./logentry.model";

export class HeatmapDetails{
  constructor(public name: string, public minValue: number, public maxValue: number, public selector: (n: LogEntry) => any) {}
}