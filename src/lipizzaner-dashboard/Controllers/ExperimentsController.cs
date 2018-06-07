using System.Collections.Generic;
using Microsoft.AspNetCore.Mvc;
using MongoDB.Driver;
using LipizzanerDashboard.Models;
using Microsoft.Extensions.Configuration;
using System.Text;
using LipizzanerDashboard.Helpers;
using MongoDB.Bson;
using System;
using System.Linq;

namespace LipizzanerDashboard.Controllers
{
  [Route("api/[controller]")]
  public class ExperimentsController : Controller
  {
    private readonly IConfiguration _configuration;

    public ExperimentsController(IConfiguration configuration)
    {
      _configuration = configuration;
    }

    [HttpGet]
    public IEnumerable<Experiment> GetExperiments()
    {
      var fieldsBuilder = Builders<Experiment>.Projection;
      var fields = fieldsBuilder.Exclude(d => d.Results);

      var db = GetDatabase();
      var collection = db.GetCollection<Experiment>("experiments");
      var experiments = collection.Find(x => !x.IsDeleted).Project<Experiment>(fields).ToList();

      return experiments;
    }

    [HttpGet("results/{experimentId}")]
    public IEnumerable<ExperimentResult> GetExperimentResults(string experimentId)
    {
      var db = GetDatabase();
      var collection = db.GetCollection<Experiment>("experiments");
      var experiment = collection.Find(x => x.Id == experimentId).FirstOrDefault();

      return experiment?.Results ?? Enumerable.Empty<ExperimentResult>();
    }


    [HttpGet("logentries/{experimentId}")]
    public IEnumerable<LogEntry> GetLogEntries(string experimentId)
    {
      var fieldsBuilder = Builders<LogEntry>.Projection;
      var fields = fieldsBuilder.Exclude(d => d.RealImages).Exclude(d => d.FakeImages);

      var db = GetDatabase();
      var collection = db.GetCollection<LogEntry>("log_entries");
      var entries = collection.Find(x => x.ExperimentId == experimentId).Project<LogEntry>(fields).ToList();

      return entries;
    }

    [HttpGet("logentries/{experimentId}/{x}/{y}")]
    public IEnumerable<LogEntry> GetLogEntryDetails(string experimentId, int x, int y)
    {
      var db = GetDatabase();
      var collection = db.GetCollection<LogEntry>("log_entries");
      var entries = collection.Find(e => e.ExperimentId == experimentId && e.GridPosition.X == x && e.GridPosition.Y == y).ToList();

      return entries;
    }

    [HttpDelete("{experimentId}")]
    public IActionResult DeleteExperiment(string experimentId)
    {
      var db = GetDatabase();
      var collection = db.GetCollection<Experiment>("experiments");

      var filter = Builders<Experiment>.Filter.Eq(x => x.Id, experimentId);
      var update = Builders<Experiment>.Update.Set(x => x.IsDeleted, true);
      var result = collection.UpdateOne(filter, update);

      if (!result.IsAcknowledged || result.MatchedCount != 1)
        return BadRequest(result);
      return Ok(result);
    }

    [HttpGet("csv")]
    public IActionResult DumpAllEntriesAsCsv()
    {
      var db = GetDatabase();
      var collection = db.GetCollection<LogEntry>("log_entries");
      var entries = collection.Find(_ => true).ToList();

      return File(Encoding.ASCII.GetBytes(entries.ToCsv()), "text/csv", "all-experiments.csv");
    }

    [HttpGet("csv/{experimentId}")]
    public IActionResult DumpExperimentEntriesAsCsv(string experimentId)
    {
      var db = GetDatabase();
      var collection = db.GetCollection<LogEntry>("log_entries");
      var entries = collection.Find(x => x.ExperimentId == experimentId).ToList();

      return File(Encoding.ASCII.GetBytes(entries.ToCsv()), "text/csv", $"{experimentId}.csv");
    }

    private IMongoDatabase GetDatabase()
    {
      var client = new MongoClient(_configuration.GetSection("Database")["ConnectionString"]);
      return client.GetDatabase("lipizzaner_db");
    }
  }
}
