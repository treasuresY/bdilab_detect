from bdilab_detect.utils.missing_optional_dependency import import_optional

TransformerEmbedding = import_optional(
    'bdilab_detect.models.tensorflow.embedding',
    names=['TransformerEmbedding'])
