using System;
using System.Collections.Generic;
using System.Linq;
using LipizzanerDashboard.Models;

namespace LipizzanerDashboard.Helpers
{
  public static class CsvHelperExtensions
  {
    public static string ToCsv(this IList<LogEntry> entries)
    {
      if (!entries.Any()) return string.Empty;

      var mapped = entries.Select(MapEntryToString).ToList();
      mapped.Insert(0, CreateCsvHeader(entries.First()));

      return string.Join(Environment.NewLine, mapped);
    }

    private static string CreateCsvHeader(LogEntry entry)
    {
      return $"id;experiment_id;iteration;inception_score;node_name;grid_pos_x;grid_pos_y;" +
             $"{CreateIndividualHeader(entry.Generators, "g")};{CreateIndividualHeader(entry.Discriminators, "d")};" +
             $"{string.Join(';', entry.MixtureWeightsGenerators.Select((_, i) => $"mw_gen_{i}"))}" +
             $"{string.Join(';', entry.MixtureWeightsDiscriminators.Select((_, i) => $"mw_dis_{i}"))}";
    }

    private static string MapEntryToString(LogEntry entry)
    {
      return $"{entry.Id};{entry.ExperimentId};{entry.Iteration};{entry.InceptionScore};{entry.NodeName};" +
             $"{entry.GridPosition.X};{entry.GridPosition.Y};{MapIndividuals(entry.Generators)};" +
             $"{MapIndividuals(entry.Discriminators)};{string.Join(';', entry.MixtureWeightsGenerators)}" +
             $"{string.Join(';', entry.MixtureWeightsDiscriminators)}";
    }

    private static string MapIndividuals(IEnumerable<Individual> individuals)
    {
      return string.Join(';', individuals.Select(x => $"{x.Loss};{string.Join(';', x.HyperParams.Values)}"));
    }

    private static string CreateIndividualHeader(IEnumerable<Individual> individuals, string prefix)
    {
      string HyperParams(Dictionary<string, double> hyperparams, int idx)
      {
        return string.Join(';', hyperparams.Keys.Select(x => $"{prefix}{idx}_param_{x}"));
      }

      return string.Join(';', individuals.Select((x, i) => $"{prefix}{i}_loss;{HyperParams(x.HyperParams, i)}"));
    }
  }
}
