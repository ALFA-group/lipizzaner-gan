using System;
using System.Collections.Generic;
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace LipizzanerDashboard.Models
{
  [BsonIgnoreExtraElements]
  public class Experiment
  {
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id { get; set; }

    [BsonElement("name")]
    public string Name { get; set; }

    [BsonElement("master")]
    public string Master { get; set; }

    [BsonElement("topology")]
    public Topology Topology { get; set; }

    [BsonElement("settings")]
    public object Settings { get; set; }

    [BsonElement("is_deleted")]
    [BsonDefaultValue(false)]
    public bool IsDeleted { get; set; }

    [BsonIgnore]
    public DateTime StartTime => TimeZoneInfo.ConvertTimeFromUtc(new ObjectId(Id).CreationTime, TimeZoneInfo.Local);

    [BsonElement("end_time")]
    public DateTime EndTime { get; set; }

    [BsonElement("results")]
    public IEnumerable<ExperimentResult>  Results { get; internal set; }
  }
}
