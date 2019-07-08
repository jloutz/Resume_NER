import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


def train_spacy_ner(training_data,
                    model=None,
                    n_iter=20,
                    debug=False):
    """Example of training spaCy's named entity recognizer, starting off with an
    existing model or a blank model.

    For more details, see the documentation:
    * Training: https://spacy.io/usage/training
    * NER: https://spacy.io/usage/linguistic-features#named-entities

    Compatible with: spaCy v2.0.0+
    Last tested with: v2.1.0
    """
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()

        ## gather up docs that throw exception (only in debug mode)
        baddocs = set()

        for itn in range(n_iter):
            random.shuffle(training_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            # singlebatch for debug
            singlebatch = 1
            compoundbatch = compounding(4.0, 32.0, 1.001)
            batchsize = singlebatch if debug else compoundbatch
            batches = minibatch(training_data, size=batchsize)
            for batch in batches:
                texts, annotations = zip(*batch)
                try:
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        losses=losses,
                    )
                except Exception:
                    if debug:
                        print("Exception thrown when processing doc:")
                        print(texts,annotations)
                        baddocs.add(batch[0][0])
                        continue
            print("Losses", losses)
    return nlp,baddocs


def load_spacy_model(model_dir):
    print("Loading from", model_dir)
    nlp = spacy.load(model_dir)
    return nlp

def persist_spacy_model(model, output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    model.to_disk(output_dir)
    print("Saved model to", output_dir)

