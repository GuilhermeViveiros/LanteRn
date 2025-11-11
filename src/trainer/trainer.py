from transformers import SFTTrainer, GRPOTrainer

class LantErnSFTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the custom_loss function
    def custom_loss(self, *args, **kwargs):
        print("custom_loss")
        import pdb; pdb.set_trace()

        pass