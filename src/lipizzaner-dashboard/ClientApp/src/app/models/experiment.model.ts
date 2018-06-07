import { Topology } from './topology.model'

export class Experiment {
  constructor(public id: string, public name: string, public master: string, public topology: Topology, public settings: any,
    public startTime: Date, public endTime: Date, public duration: string) { }
}
