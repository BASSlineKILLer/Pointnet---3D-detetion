from .detector3d_template import Detector3DTemplate


class PointRCNN_before(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        print("[DEBUG] Starting forward pass of PointRCNN_before")
        for i, cur_module in enumerate(self.module_list):
            print(f"[DEBUG] Running module {i}: {cur_module.__class__.__name__}")
            batch_dict = cur_module(batch_dict)
        if self.training:
            print("[DEBUG] Calculating training loss...")
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            print("[DEBUG] Running post-processing...")
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
