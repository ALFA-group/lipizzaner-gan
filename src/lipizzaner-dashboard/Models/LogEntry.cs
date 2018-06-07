using System.Collections.Generic;
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace LipizzanerDashboard.Models
{
  [BsonIgnoreExtraElements]
  public class LogEntry
  {
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id { get; set; }

    [BsonElement("experiment_id")]
    [BsonRepresentation(BsonType.ObjectId)]
    public string ExperimentId { get; set; }

    [BsonElement("iteration")]
    public int Iteration { get; set; }

    [BsonElement("grid_position")]
    public GridPosition GridPosition { get; set; }

    [BsonElement("node_name")]
    public string NodeName { get; set; }

    [BsonElement("mixture_weights_gen")]
    public double[] MixtureWeightsGenerators { get; set; }

     [BsonElement("mixture_weights_dis")]
    public double[] MixtureWeightsDiscriminators { get; set; }

    [BsonElement("inception_score")]
    public double InceptionScore { get; set; }

    [BsonElement("duration_sec")]
    public double DurationSec { get; set; }

    [BsonElement("generators")]
    public IEnumerable<Individual> Generators { get; set; }

    [BsonElement("discriminators")]
    public IEnumerable<Individual> Discriminators { get; set; }

    [BsonElement("real_images")]
    public string RealImages { get; set; }

    [BsonElement("fake_images")]
    public string FakeImages { get; set; }
  }
}
