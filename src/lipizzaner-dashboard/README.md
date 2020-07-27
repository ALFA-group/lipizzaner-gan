# Lipizzaner Dashboard
A simple dashboard application that displays the logs Lipizzaner writes into it's MongoDB log server.

It uses ASP.NET Core for the backend and Angular for the frontend code.

## Running it
0. Setup a MongoDB server instance that the Lipizzaner dashboard can access, and set its connection string in `appsettings.json`.
   - To also connect the framework to this server, set the connection string in `../configuration/general.yml` as well.
1. Install .NET Core for your platform, as described [here](https://docs.microsoft.com/en-us/dotnet/core/get-started).
2. Install Node.js with NPM, as described [here](https://nodejs.org/en/).
3. Install the frontend dependencies
    1. `cd` into the `lipizzaner-client/ClientApp` directory.
    2. Run `npm install`.
5. Restore the backend dependencies
    1. `cd` into the `lipizzaner-client` directory (i.e. `cd ..`).
    2.  Run `dotnet restore`.
6. Depending on your OS (e.g. on Ubuntu), you may have to specify an environment variable:
  ```export ASPNETCORE_ENVIRONMENT=Development;```
  Running in development mode makes the process easier, as it automatically starts `ng serve`, which serves the frontend files.
8. Run the app with `dotnet run`

This application was built with the ASP.NET Core Angular template, details about it can be found [here](https://docs.microsoft.com/en-us/aspnet/core/spa/angular?view=aspnetcore-2.1&tabs=visual-studio).
