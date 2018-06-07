using System.Collections.Generic;
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace LipizzanerDashboard.Models
{
  [BsonIgnoreExtraElements]
  public class Individual
  {
    [BsonElement("cell_id")]
    public string CellId { get; set; }

    [BsonElement("loss")]
    public double Loss { get; set; }

    [BsonElement("hyper_params")]
    public Dictionary<string, double> HyperParams { get; set; }
  }
}
