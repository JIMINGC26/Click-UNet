from inference.predictors import Predictor
from models.Unet_id import UNet2D_Test

def get_predictor(net, device, prob_tresh=0.49):
    predictor = Predictor(net, device)
    return predictor