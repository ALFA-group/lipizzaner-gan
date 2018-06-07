# Adding a new data source to Lipizzaner
This tutorial describes that steps that are necessary to add a new data source and a new neural network topology to Lipizzaner (if the latter is required).
1) Add a new data loader class to `src/data`, inherit from `data.DataLoader` (or copy the structure from another data loader).
   1) The `n_input_neurons` property is the number of input neurons for the **discriminator** (i.e. how many dimensions your data has)
   2) You'll have to create your own dataset as well for custom data (it must inherit from `torch.utils.data.Dataset`). Look at `src/data/grid_toy_data_loader.py` for an example.
       Basically, the dataset provides the data, and the DataLoader only allows to iterate over it (a concept given by PyTorch)
2) Create your network factory, the class that creates your NN models, in `src/networks/network_factory.py` (or use a separate file; this should be split up in the future anyway).
   1) Inherit from `NetworkFactory`, `gen_input_size` is the number of input neurons of the **generator**. Generally, at least 100 input neurons should be used.
     
      You can basically copy this from another network factory. Just make sure you specify `self.gen_input_size` as the generator's input size, and `self.input_data_size` as the generator's output/discriminator's input size (good example: CircularProblemFactory). 
      **`self.input_data_size` must match the size of your data.**
3) Create class mappings for your new classes (both network factory and data loader) in `src/helpers/configuration_container.py`. These mappings will be used to create the class instances at runtime (pick any alphanumeric mapping name you want).
4) Create a configuration file in `src/configuration/lipizzaner-[w]gan`. You can copy a pre-existing one and just change the dataloader and network factory properties.

To run and debug the application locally, set your IP address in `src/configuration/general.yml`, select one port for a one-cell-grid (`5000`), and then run:

   ```
   python main.py --distributed --client
   python main.py --distributed --master -f configuration/lipizzaner-(w)gan/your_config.yml
   ```
