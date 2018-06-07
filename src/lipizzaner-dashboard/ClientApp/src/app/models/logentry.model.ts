import { GridPosition } from "./gridposition.model";
import { Individual } from "./individual.model";

export class LogEntry {
  constructor(public id?: string, public experimentId?: string, public iteration?: number, public gridPosition?: GridPosition,
    public nodeName?: string, public mixtureWeightsGenerators?: number[],  public mixtureWeightsDiscriminators?: number[],
    public inceptionScore?: number, public durationSec?: number, public generators?: Individual[], public discriminators?: Individual[],
    public individuals?: any[], public realImages?: any, public fakeImages?: any) {
    if (!generators) this.generators = [];
    if (!discriminators) this.discriminators = [];
    if (!individuals) this.individuals = [];
  }
}
