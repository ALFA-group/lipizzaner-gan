import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Experiment } from '../models/experiment.model'
import { LogEntry } from '../models/logentry.model'
import { ActivatedRoute, Router } from '@angular/router';
import { HeatmapDetails } from '../models/heatmap-details.model';
import { ToastrService } from 'ngx-toastr';
import { ExperimentResult } from '../models/experiment-result.model';

@Component({
  selector: 'app-grid-nav',
  templateUrl: './grid-nav.component.html',
  styleUrls: ['./grid-nav.component.css']
})
export class GridNavComponent implements OnInit {

  // For now: Values for CelebA
  heatmapDetails: HeatmapDetails[] = [
    new HeatmapDetails('Generator loss', 0, 3, (entry) => entry.generators[0].loss),
    new HeatmapDetails('Learning rate', 0, 0.001, (entry) => entry.generators[0].hyperParams['lr']),
    new HeatmapDetails('Inception score', 1, 2, (entry) => entry.inceptionScore)
  ]
  selectedHeatmapDetails = this.heatmapDetails[0];

  experiments: Experiment[];
  selectedExperiment: Experiment;
  selectedExperimentResults: ExperimentResult[];

  server_url = "128.52.181.99:27017";
  gridCellSize = 100;
  currentIteration = 0;
  maxIterations = 0;
  logEntries: LogEntry[] = [];
  currentIterationEntries: LogEntry[][] = [];

  detailedThreshold = 3;
  detailedGrid = true;

  @ViewChild('gridContainer') gridContainer: ElementRef;

  width: number[];
  height: number[];

  constructor(private http: HttpClient, private activatedRoute: ActivatedRoute, private toastr: ToastrService,
    private router: Router) { }

  ngOnInit(): void {
    this.activatedRoute.queryParams.subscribe(params => {
      let experimentId = params['experimentId'];
      this.init(experimentId);
    });
  }

  init(experimentId: any = undefined): any {
    this.http.get<Experiment[]>('api/experiments').subscribe(result => {
      // Map datetimes from strings to JS objects
      result.forEach(e => {
        e.startTime = new Date(e.startTime);
        e.endTime = !e.endTime || e.endTime.toString() === '0001-01-01T00:00:00' ? new Date(0) : new Date(e.endTime);
        e.duration = this.calcDuration(e)
      });
      this.experiments = result.sort((n1, n2) => {
        if (n1.name > n2.name) return -1;
        if (n1.name < n2.name) return 1;
        return 0;
      });

      if (experimentId)
        this.selectedExperiment = this.experiments.filter(x => x.id == experimentId)[0];
      else
        this.selectedExperiment = this.experiments[0];

      this.changeExperiment();
    }, error => console.error(error));
  }

  changeExperiment(): void {
    this.width = this.selectedExperiment ? Array.from(Array(this.selectedExperiment.topology.width).keys()) : [];
    this.height = this.selectedExperiment ? Array.from(Array(this.selectedExperiment.topology.height).keys()) : [];

    this.http.get<LogEntry[]>('api/experiments/logentries/' + this.selectedExperiment.id).subscribe(result => {

      this.currentIteration = result && result.length != 0 ? Math.max.apply(Math, result.map(x => x.iteration)) : 0;
      this.maxIterations = this.currentIteration;
      this.logEntries = result;

      this.updateGridContent();

      const gridContainer = this.gridContainer.nativeElement as HTMLElement;
      const style = getComputedStyle(gridContainer);
      let gridWidth = gridContainer.offsetWidth - (parseFloat(style.paddingLeft));
      this.gridCellSize = gridWidth / this.selectedExperiment.topology.width;
    });

    this.http.get<ExperimentResult[]>('api/experiments/results/' + this.selectedExperiment.id).subscribe(results => {
      for (let result of results) {
        result.images = result.images.map(x => 'data:image/png;base64,' + x);
      }

      this.selectedExperimentResults = results.sort((n1, n2) => {
        if (n1.mixtureCenter > n2.mixtureCenter) return 1;
        if (n1.mixtureCenter < n2.mixtureCenter) return -1;
        return 0;
      });
    });
  }

  updateGridContent(): void {
    this.detailedGrid = this.selectedExperiment.topology.width < this.detailedThreshold;

    let currentIterationEntries = []
    for (let y = 0; y < this.selectedExperiment.topology.width; y++) {
      currentIterationEntries[y] = [];
      for (let x = 0; x < this.selectedExperiment.topology.height; x++) {
        let element = this.logEntries.filter(entry => entry.iteration === this.currentIteration &&
          entry.gridPosition.x == x && entry.gridPosition.y == y)[0];
        currentIterationEntries[y][x] = element ? element : new LogEntry();

        // Zip for easier view logic
        currentIterationEntries[y][x].individuals =
          currentIterationEntries[y][x].generators.map((gen, i) => ({
            generator: gen,
            discriminator: currentIterationEntries[y][x].discriminators[i]
          }));
      }
    }

    this.currentIterationEntries = currentIterationEntries;
  }

  dumpCurrentExperiment(): void {
    window.open(`/api/experiments/csv/${this.selectedExperiment.id}`);
  }

  dumpAllExperiments(): void {
    window.open('/api/experiments/csv');
  }

  selectHeatmapDetails(heatmapDetails: HeatmapDetails) {
    this.selectedHeatmapDetails = heatmapDetails;
  }

  deleteCurrentExperiment(): void {
    let name = this.selectedExperiment.name;

    if (confirm(`Are you sure you want to delete the currently selected experiment (${name})?`)) {
      this.http.delete('api/experiments/' + this.selectedExperiment.id).subscribe(result => {
        this.init();
        this.toastr.success(`Successfully set experiment status of ${name} to deleted.`, 'Success');
      }, error => this.toastr.success('Error', error));
    }
  }

  hasExperimentEnded(): boolean {
    return this.selectedExperiment.endTime && this.selectedExperiment.endTime.getTime() !== new Date(0).getTime();
  }

  heatMapColorforValue(entry: LogEntry): string {
    // Normalize to [0, 1], currently optimized for CelebA
    let min = this.selectedHeatmapDetails.minValue;
    let max = this.selectedHeatmapDetails.maxValue;
    let value = this.selectedHeatmapDetails.selector(entry);
    value = (value - min) / (max - min);

    let s = (1.0 - value) * 100
    return "hsl(204, " + s + "%, 56%)";
  }

  calcDuration(experiment: Experiment): string {
    let durationMsTotal = experiment.endTime.getTime() != new Date(0).getTime()
      ? experiment.endTime.getTime() - experiment.startTime.getTime()
      : new Date().getTime() - experiment.startTime.getTime();

    let durationMinutesTotal = durationMsTotal / 1000 / 60;
    let hours = Math.floor(durationMinutesTotal / 60);
    let minutes = Math.round(durationMinutesTotal % 60);

    return `${hours}:${minutes.toString().padStart(2, "0")} hours`;
  }

  openImage(image: string): void {
    window.open(image, '_blank');
  }
}
