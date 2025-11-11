from transformers import Trainer

class LantErnSFTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the custom_loss function
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        For LantErn, we also need to compute the distance between the predicted and ground truth latents
        """
        import pdb; pdb.set_trace()
        outputs = model(
            **inputs,
            #output_hidden_states=True,
            return_dict=True,
        )
        # clear
        torch.cuda.empty_cache()
        # compute the distance between the predicted and ground truth latents
        import pdb; pdb.set_trace()

        pass

    