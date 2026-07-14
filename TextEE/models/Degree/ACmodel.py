import torch

from .EAEmodel import DegreeEAEModel


class DegreeACModel(DegreeEAEModel):
    """AC variant of Degree's EAE generative model.

    Inherits the BART encoder-decoder pipeline from DegreeEAEModel
    (process_data / forward / generate). The only override is `predict`,
    which plumbs `length_penalty` through to HF generate so per-prompt-group
    decoding hyperparameters can be applied at inference time.
    """

    def predict(self, batch, num_beams=4, max_length=50, length_penalty=1.0):
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs = self.process_data(batch)
        self.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=enc_idxs,
                attention_mask=enc_attn,
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=float(length_penalty),
            )
        final_output = []
        for bid in range(enc_idxs.size(0)):
            final_output.append(self.tokenizer.decode(
                outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        self.train()
        return final_output
