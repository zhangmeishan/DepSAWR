"""
The ``elmo`` subcommand allows you to make bulk ELMo predictions.

Given a pre-processed input text file, this command outputs the internal
layers used to compute ELMo representations to a single (potentially large) file.

The input file is previously tokenized, whitespace separated text, one sentence per line.
The output is a hdf5 file (<http://docs.h5py.org/en/latest/>) where, with the --all flag, each
sentence is a size (3, num_tokens, 1024) array with the biLM representations.

For information, see "Deep contextualized word representations", Peters et al 2018.
https://arxiv.org/abs/1802.05365

.. code-block:: console

   $ allennlp elmo --help
   usage: allennlp elmo [-h] (--all | --top | --average)
                        [--vocab-path VOCAB_PATH] [--options-file OPTIONS_FILE]
                        [--weight-file WEIGHT_FILE] [--batch-size BATCH_SIZE]
                        [--cuda-device CUDA_DEVICE] [--forget-sentences]
                        [--use-sentence-keys] [--include-package INCLUDE_PACKAGE]
                        input_file output_file

   Create word vectors using ELMo.

   positional arguments:
     input_file            The path to the input file.
     output_file           The path to the output file.

   optional arguments:
     -h, --help            show this help message and exit
     --all                 Output all three ELMo vectors.
     --top                 Output the top ELMo vector.
     --average             Output the average of the ELMo vectors.
     --vocab-path VOCAB_PATH
                           A path to a vocabulary file to generate.
     --options-file OPTIONS_FILE
                           The path to the ELMo options file.
     --weight-file WEIGHT_FILE
                           The path to the ELMo weight file.
     --batch-size BATCH_SIZE
                           The batch size to use.
     --cuda-device CUDA_DEVICE
                           The cuda_device to run on.
     --forget-sentences    If this flag is specified, and --use-sentence-keys is
                           not, remove the string serialized JSON dictionary that
                           associates sentences with their line number (its HDF5
                           key) that is normally placed in the
                           "sentence_to_index" HDF5 key.
     --use-sentence-keys   Normally a sentence's line number is used as the HDF5
                           key for its embedding. If this flag is specified, the
                           sentence itself will be used as the key.
     --include-package INCLUDE_PACKAGE
                           additional packages to include
"""

import argparse
import json
import logging
import os
from typing import IO, List, Iterable, Tuple
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import numpy
import torch

#from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of, prepare_global_logging
from allennlp.common.checks import ConfigurationError
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long
DEFAULT_BATCH_SIZE = 64


def empty_embedding() -> numpy.ndarray:
    return numpy.zeros((3, 0, 1024))

class ElmoEmbedder():
    def __init__(self,
                 options_file: str = DEFAULT_OPTIONS_FILE,
                 weight_file: str = DEFAULT_WEIGHT_FILE,
                 cuda_device: int = -1) -> None:
        """
        Parameters
        ----------
        options_file : ``str``, optional
            A path or URL to an ELMo options file.
        weight_file : ``str``, optional
            A path or URL to an ELMo weights file.
        cuda_device : ``int``, optional, (default=-1)
            The GPU device to run on.
        """
        self.indexer = ELMoTokenCharactersIndexer()

        logger.info("Initializing ELMo.")
        self.elmo_bilm = _ElmoBiLm(options_file, weight_file)
        if cuda_device >= 0:
            self.elmo_bilm = self.elmo_bilm.cuda(device=cuda_device)

        self.cuda_device = cuda_device

    def batch_to_embeddings(self, batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        Returns
        -------
            A tuple of tensors, the first representing activations (batch_size, 3, num_timesteps, 1024) and
        the second a mask (batch_size, num_timesteps).
        """
        character_ids = batch_to_ids(batch)
        if self.cuda_device >= 0:
            character_ids = character_ids.cuda(device=self.cuda_device)

        bilm_output = self.elmo_bilm(character_ids)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        # without_bos_eos is a 3 element list of (activation, mask) tensor pairs,
        # each with size (batch_size, num_timesteps, dim and (batch_size, num_timesteps)
        # respectively.
        without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos)
                           for layer in layer_activations]
        # Converts a list of pairs (activation, mask) tensors to a single tensor of activations.
        activations = torch.cat([ele[0].unsqueeze(1) for ele in without_bos_eos], dim=1)
        # The mask is the same for each ELMo vector, so just take the first.
        mask = without_bos_eos[0][1]

        return activations, mask

    def embed_sentence(self, sentence: List[str]) -> numpy.ndarray:
        """
        Computes the ELMo embeddings for a single tokenized sentence.

        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.

        Parameters
        ----------
        sentence : ``List[str]``, required
            A tokenized sentence.

        Returns
        -------
        A tensor containing the ELMo vectors.
        """

        return self.embed_batch([sentence])[0]

    def embed_batch(self, batch: List[List[str]]) -> List[numpy.ndarray]:
        """
        Computes the ELMo embeddings for a batch of tokenized sentences.

        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.

        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        elmo_embeddings = []

        # Batches with only an empty sentence will throw an exception inside AllenNLP, so we handle this case
        # and return an empty embedding instead.
        if batch == [[]]:
            elmo_embeddings.append(empty_embedding())
        else:
            embeddings, mask = self.batch_to_embeddings(batch)
            for i in range(len(batch)):
                length = int(mask[i, :].sum())
                # Slicing the embedding :0 throws an exception so we need to special case for empty sentences.
                if length == 0:
                    elmo_embeddings.append(empty_embedding())
                else:
                    elmo_embeddings.append(embeddings[i, :, :length, :].detach().cpu().numpy())

        return elmo_embeddings

    def embed_sentences(self,
                        sentences: Iterable[List[str]],
                        batch_size: int = DEFAULT_BATCH_SIZE) -> Iterable[numpy.ndarray]:
        """
        Computes the ELMo embeddings for a iterable of sentences.

        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.

        Parameters
        ----------
        sentences : ``Iterable[List[str]]``, required
            An iterable of tokenized sentences.
        batch_size : ``int``, required
            The number of sentences ELMo should process at once.

        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        for batch in lazy_groups_of(iter(sentences), batch_size):
            yield from self.embed_batch(batch)

