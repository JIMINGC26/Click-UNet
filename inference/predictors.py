import sys
sys.path.append('../')

import torch
import torch.nn.functional as F
from torchvision import transforms
from dataloaders.BaseTransform import SigmoidForPred

class Predictor(object):
    def __init__(self,model, device="cuda:0", net_clicks_limit=None):
        self.device = device
        self.net_clicks_limit = net_clicks_limit
        self.net = model
        self.original_image = None
        self.prev_prediction = None

        self.to_tensor = transforms.ToTensor()
        self.transforms = SigmoidForPred()

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        
        self.transforms.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def get_predictions_ori(self, prev_mask=None):
    
        input_img = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_img, []
        )
        pred_logits = self._get_prediction_ori(image_nd, clicks_lists,prev_mask,  is_image_changed)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        
        prediction = self.transforms.inv_transform(prediction)

        self.prev_prediction = prediction
        return prediction.cpu().detach().numpy()[0, 0]

    def get_predictions_refine(self, clicker, prev_mask=None):
        clicks_list = clicker.get_clicks()

        input_img = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_img, [clicks_list]
        )
        pred_logits = self._get_prediction_re(image_nd, clicks_lists, prev_mask, is_image_changed)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        prediction = self.transforms.inv_transform(prediction)

        self.prev_prediction = prediction
        return prediction.cpu().detach().numpy()[0, 0]

    # TODO:refine=True, point=, 区分是否添加点击     
    def _get_prediction_re(self, image_nd, clicks_lists, prev_mask, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, len(clicks_lists), point=points_nd, pred=prev_mask)
       
    def _get_prediction_ori(self, image_nd, clicks_lists, prev_mask, is_image_changed):
        points_nd = torch.ones(1, 20, 3) * -1
        points_nd.to(self.device)
        print(image_nd.shape, type(image_nd))
        return self.net(image_nd, 0, points_nd, prev_mask) 
        
    
    def _get_transform_states(self):
        return self.transforms

    def _set_transform_states(self, states):
        # assert len(states) == len(self.transforms)
        # for state, transform in zip(states, self.transforms):
        # transforms.set_state(states)
        return

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        
        image_nd, clicks_lists = self.transforms.transform(image_nd, clicks_lists)
        is_image_changed |= self.transforms.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
