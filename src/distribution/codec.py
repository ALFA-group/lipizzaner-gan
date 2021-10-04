import tempfile
import codecs
import torch

from helpers.individual import Individual


class Codec:
    @staticmethod
    def _encode_state(state_dict):
        with tempfile.TemporaryFile() as tmp:
            torch.save(state_dict, tmp)
            tmp.flush()
            tmp.seek(0)
            return codecs.encode(tmp.read(), "base64").decode()

    @staticmethod
    def _decode_state(data):
        with tempfile.TemporaryFile() as tmp:
            tmp.write(codecs.decode(data.encode(), "base64"))
            tmp.flush()
            tmp.seek(0)
            return torch.load(tmp)

    @staticmethod
    def encode(individual):
        json_response = {
            "id": individual.id,
            "parameters": individual.genome.encoded_parameters,
            "learning_rate": individual.learning_rate,
            "optimizer_state": Codec.encode(individual.optimizer_state),
        }

        if individual.iteration is not None:
            json_response["iteration"] = individual.iteration

        if hasattr(individual.genome, "classification_layer"):
            json_response["classification_layer_parameters"] = individual.genome.encoded_classification_layer_parameters

        return json_response

    @staticmethod
    def decode(json, create_genome):
        individual = Individual.decode(
            create_genome,
            json["parameters"],
            is_local=False,
            learning_rate=json["learning_rate"],
            optimizer_state=Codec.decode(json["optimizer_state"]),
            source=json["source"],
            id=json["id"],
            iteration=json.get("iteration", None),
        )

        if hasattr(individual.genome, "classification_layer"):
            individual.genome.encoded_classification_layer_parameters = json["classification_layer_parameters"]

        return individual
