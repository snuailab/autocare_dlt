import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, **loss_params) -> None:
        super().__init__()

        self.criterion = nn.MSELoss()

        # === Hyperparams for loss === #
        # self.use_target_weight = loss_params.get("use_target_weight")
        self.scaled_loss = loss_params.get("scaled_loss")

        if not isinstance(self.scaled_loss, bool):
            raise KeyError(
                f"scaled_loss: {self.scaled_loss} must be a boolean."
            )

        # === Hardware cfgs === #
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device

    def forward(self, output: torch.Tensor, target: list) -> dict:

        if output.shape[0] != len(target):  # batch sizes
            raise ValueError(
                f"the number of prediction ({output.shape[0]}) and the number of targets ({len(target)} are not the same)"
            )

        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_gt = torch.zeros_like(output)
        for i, t in enumerate(target):
            heatmaps_gt[i] = t["heatmap"]

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(
            1, 1
        )
        heatmaps_gt = heatmaps_gt.reshape((batch_size, num_joints, -1)).split(
            1, 1
        )

        loss = torch.zeros(1, device=self.device)

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            # if self.use_target_weight:
            #     loss += 0.5 * self.criterion(
            #         heatmap_pred.mul(target_weight[:, idx]),
            #         heatmap_gt.mul(target_weight[:, idx]),
            #     )
            # else:
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        loss /= num_joints
        loss_dict = {"joint_mse_loss": loss}

        return loss_dict
