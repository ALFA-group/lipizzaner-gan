import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { LogEntry } from '../models/logentry.model';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-details',
  templateUrl: './details.component.html',
  styleUrls: ['./details.component.css']
})
export class DetailsComponent implements OnInit {

  logEntries: LogEntry[] = [];
  lossChartData: Array<any>;
  genMixtureChartData: Array<any>;
  disMixtureChartData: Array<any>;
  hyperParamsChartData: Array<any>;
  inceptionScoreChartData: Array<any>;
  durationChartData: Array<any>;

  lineChartLabels: Array<any>;

  lineChartOptions: any = {
    responsive: true,
    elements: {
      point: {
        radius: 0,
        hitRadius: 5,
        hoverRadius: 5
      }
    },
    scales: {
      yAxes: [{
        ticks: {
          beginAtZero: true,
        }
      }]
    }
  };

  constructor(private http: HttpClient, private activatedRoute: ActivatedRoute, private cdRef: ChangeDetectorRef) { }

  ngOnInit(): void {
    this.activatedRoute.queryParams.subscribe(params => {
      let experimentId = params['experimentId'];
      let x = params['x'];
      let y = params['y'];

      this.http.get<LogEntry[]>(`api/experiments/logentries/${experimentId}/${x}/${y}`).subscribe(result => {
        this.logEntries = result;

        this.lossChartData = this.logEntries.length == 0 ? [] : [
          { data: this.logEntries.map(x => x.generators[0].loss), label: 'Generator', fill: false },
          { data: this.logEntries.map(x => x.discriminators[0].loss), label: 'Discriminator', fill: false }
        ]

        this.genMixtureChartData = []
        this.disMixtureChartData = []
        if(this.logEntries.length && this.logEntries[0].mixtureWeightsGenerators && this.logEntries[0].mixtureWeightsGenerators.length) {
          for (let i = 0; i < this.logEntries[0].mixtureWeightsGenerators.length; i++) {
            this.genMixtureChartData.push({ data: this.logEntries.map(x => x.mixtureWeightsGenerators[i]), label: `Cell ${i}`, fill: false });
            this.disMixtureChartData.push({ data: this.logEntries.map(x => x.mixtureWeightsDiscriminators[i]), label: `Cell ${i}`, fill: false });
          }
        }

        this.hyperParamsChartData = []
        let paramsToDisplay = [].concat.apply([], this.logEntries.map(x => Object.keys(x.generators[0].hyperParams)))
          .filter((v, i, a) => a.indexOf(v) === i);
        for (let param of paramsToDisplay) {
          this.hyperParamsChartData.push({ data: this.logEntries.map(x => x.generators[0].hyperParams[param]), label: `Generator - ${param}`, fill: false });
          this.hyperParamsChartData.push({ data: this.logEntries.map(x => x.discriminators[0].hyperParams[param]), label: `Discriminator - ${param}`, fill: false });
        }

        this.inceptionScoreChartData = this.logEntries.length == 0 ? [] : [
          { data: this.logEntries.map(x => x.inceptionScore), label: 'Score' },
        ];

        this.durationChartData = this.logEntries.length == 0 ? [] : [
          { data: this.logEntries.map(x => x.durationSec / 60), label: 'Duration in minutes' },
        ];

        // Needed to refresh axis, which does somehow not work by default
        this.cdRef.detectChanges();
        this.lineChartLabels = this.logEntries.map(x => x.iteration);

        for (let entry of this.logEntries) {
          if (entry.realImages)
            entry.realImages = 'data:image/png;base64,' + entry.realImages;
          if (entry.fakeImages)
            entry.fakeImages = 'data:image/png;base64,' + entry.fakeImages;
        }

        console.log(this.logEntries);
      });
    });
  }

  openImage(image: string): void {
    window.open(image, '_blank');
  }
}
