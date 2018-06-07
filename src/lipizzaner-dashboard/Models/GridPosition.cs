using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace LipizzanerDashboard.Models
{
  public class GridPosition
  {
    [BsonElement("x")]
    public int X { get; set; }

    [BsonElement("y")]
    public int Y { get; set; }
  }
}
