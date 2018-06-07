using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace LipizzanerDashboard.Models
{
  public class Topology
  {
    [BsonElement("type")]
    public string Type { get; set; }

    [BsonElement("width")]
    public int Width { get; set; }

    [BsonElement("height")]
    public int Height { get; set; }
  }
}
