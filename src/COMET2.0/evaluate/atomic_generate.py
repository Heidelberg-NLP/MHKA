import time
import torch

import src.evaluate.generate as base_generate
import src.evaluate.sampler as sampling
import utils.utils as utils
import src.data.config as cfg


def make_generator(opt, *args):
    return AtomicGenerator(opt, *args)


class AtomicGenerator(base_generate.Generator):
    def __init__(self, opt, model, data_loader):
        self.opt = opt

        self.model = model
        self.data_loader = data_loader

        self.sampler = sampling.make_sampler(
            opt.eval.sample, opt, data_loader)

    def reset_sequences(self):
        return []

    def generate(self, split="dev"):
        print("Generating Sequences")

        # Set evaluation mode
        self.model.eval()

        # Reset evaluation set for dataset split
        self.data_loader.reset_offsets(splits=split, shuffle=False)

        start = time.time()
        count = 0
        sequences = None

        # Reset generated sequence buffer
        sequences = self.reset_sequences()

        # Initialize progress bar
        bar = utils.set_progress_bar(
            self.data_loader.total_size[split] / 2)

        reset = False

        with torch.no_grad():
            # Cycle through development set
            while not reset:

                start = len(sequences)
                # Generate a single batch
                reset = self.generate_batch(sequences, split, bs=1)

                end = len(sequences)

                if not reset:
                    bar.update(end - start)
                else:
                    print(end)

                count += 1

                if cfg.toy and count > 10:
                    break
                #if (self.opt.eval.gs != "full" and (count > opt.eval.gs)):
                #    break

        torch.cuda.synchronize()
        print("{} generations completed in: {} s".format(
            split, time.time() - start))

        avg_scores, indiv_scores = None, None

        return sequences, avg_scores, indiv_scores

    def generate_batch(self, sequences, split, verbose=False, bs=1):
        # Sample batch from data loader
        batch, reset = self.data_loader.sample_batch(
            split, bs=bs)

        start_idx = self.data_loader.max_event+1
        max_end_len = self.data_loader.max_effect

        context = batch["sequences"][:, :start_idx]
        reference = batch["sequences"][:, start_idx:]
        init = "".join([self.data_loader.vocab_decoder[i].replace(
            '</w>', ' ') for i in context[:, :self.data_loader.max_event].squeeze().tolist() if i]).strip()

        start = self.data_loader.max_event
        end = self.data_loader.max_event+1

        attr = "".join([self.data_loader.vocab_decoder[i].replace(
            '</w>', ' ') for i in context[:, start:end].squeeze(0).tolist() if i]).strip()

        # Decode sequence
        sampling_result = self.sampler.generate_sequence(
            batch, self.model, self.data_loader, start_idx, max_end_len)

        sampling_result["key"] = batch["key"]
        sampling_result["e1"] = init
        sampling_result["r"] = attr
        sequences.append(sampling_result)

        return reset
