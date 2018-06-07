using System;
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace LipizzanerDashboard.Models
{
  [BsonIgnoreExtraElements]
  public class ExperimentResult
  {
    [BsonElement("mixture_center")]
    public string MixtureCenter { get; set; }

    [BsonElement("inception_score")]
    public double[] InceptionScore { get; set; }

    [BsonElement("images")]
    public string[] Images { get; set; }
  }
}
